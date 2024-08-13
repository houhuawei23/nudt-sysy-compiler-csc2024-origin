// #define NDEBUG
// #define DEBUG
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include "pass/analysis/ControlFlowGraph.hpp"
#include "pass/analysis/indvar.hpp"
// #include "pass/optimize/Loop/LoopParallel.hpp"
#include "pass/optimize/Loop/LoopBodyExtract.hpp"
#include "pass/optimize/Utils/BlockUtils.hpp"

#include <cassert>
#include <unordered_map>
#include <iostream>
#include <vector>

using namespace ir;
namespace pass {

void LoopBodyInfo::print(std::ostream& os) const {
  os << "LoopBodyInfo: " << std::endl;
  std::cout << "callInst: ";
  callInst->print(os);
  os << std::endl;
  std::cout << "indVar: ";
  indVar->print(os);
  std::cout << std::endl;
  std::cout << "preHeader: ";
  preHeader->dumpAsOpernd(os);
  os << std::endl;
  std::cout << "header: ";
  header->dumpAsOpernd(os);
  os << std::endl;
  std::cout << "body: ";
  body->dumpAsOpernd(os);
  os << std::endl;
  std::cout << "latch: ";
  latch->dumpAsOpernd(os);
  os << std::endl;
}
void LoopBodyExtract::run(Function* func, TopAnalysisInfoManager* tp) {
  runImpl(func, tp);
}

bool hasCall(Loop* loop) {
  for (auto block : loop->blocks()) {
    for (auto inst : block->insts()) {
      if (auto call = inst->dynCast<CallInst>()) {
        return true;
      }
    }
  }
  return false;
}

bool LoopBodyExtract::runImpl(Function* func, TopAnalysisInfoManager* tp) {
  auto sideEffectInfo = tp->getSideEffectInfo();
  CFGAnalysisHHW().run(func, tp);  // refresh CFG

  auto lpctx = tp->getLoopInfo(func);
  auto indVarInfo = tp->getIndVarInfo(func);
  bool modified = false;
#ifdef DEBUG
  func->rename();
  // func->print(std::cerr);
#endif
  auto loops = lpctx->sortedLoops();

  std::unordered_set<Loop*> extractedLoops;
  const auto isBlocked = [&](Loop* lp) {
    for (auto extracted : extractedLoops) {
      if (extracted->blocks().count(lp->header())) {
#ifdef DEBUG
        lp->header()->dumpAsOpernd(std::cerr);
        std::cerr << "is sub of ";
        extracted->header()->dumpAsOpernd(std::cerr);
        std::cerr << std::endl;
#endif
        return true;
      }
    }
    return false;
  };
  for (auto loop : loops) {
    if (isBlocked(loop)) continue;
    if (hasCall(loop)) continue;
    const auto indVar = indVarInfo->getIndvar(loop);
    const auto step = indVar->getStep()->i32();

    if (step != 1) continue;  // only support step = 1

    LoopBodyInfo loopBodyInfo;
    if (not extractLoopBody(func, *loop, indVar, tp, loopBodyInfo /* ret */)) continue;
    modified = true;
    extractedLoops.insert(loop);
#ifdef DEBUG
    std::cerr << "extracted loop body: " << loopBodyInfo.callInst->callee()->name() << std::endl;
#endif
    // break;
  }
  tp->CallChange();
  tp->CFGChange(func);
  tp->IndVarChange(func);
  // fix cfg
  CFGAnalysisHHW().run(func, tp);  // refresh CFG
  return modified;
}

/**
      other
        |
        v
  |-> loop header --> loop next
  |     |
  |     v
  |   loop body
  |     |
  |     v
  --- loop latch

- loop.header:
  - phi = phi [v1, other], [i.next, loop.latch] ; phi inst (for indvar),
  - cond = imcp op phi, endVar
  - br cond, loop.body, loop.next

- loop.body:
  - real body of the loop

- loop.latch:
  - i.next = i + step
  - br loop.header

==> after extractLoopBody:

      other
        |
        v
  --> loop header --> loop.next
  |     |
  |     v
  |  callBlock
  |     |
  |     v
  --  loop latch


newLoop:
  - i = phi [i0, other], [i.next, newLoop]
  -
 */
// need iterInst in loop.latch
static auto getUniqueID() {
  static size_t id = 0;
  const auto base = "sysyc_loop_body";
  return base + std::to_string(id++);
}

bool moveNext2NewLatch(Function* func, Loop* loop, IndVar* indVar, TopAnalysisInfoManager* tp) {
  assert(loop->latchs().size() == 1);
  const auto next = indVar->iterInst();
  auto oldLatch = loop->getUniqueLatch();
  auto newLatch = func->newBlock();
  newLatch->set_name("new_latch");
  newLatch->set_idx(func->blocks().size());
  bool finded = false;
  for (auto block : func->blocks()) {
    for (auto inst : block->insts()) {
      if (inst == next) {
        block->move_inst(inst);
        newLatch->emplace_back_inst(inst);
        finded = true;
        break;
      }
    }
    if (finded) break;
  }
  IRBuilder builder;
  oldLatch->insts().pop_back();  // pop jump to header
  builder.set_pos(oldLatch, oldLatch->insts().end());
  builder.makeInst<BranchInst>(newLatch);
  builder.set_pos(newLatch, newLatch->insts().end());
  builder.makeInst<BranchInst>(loop->header());
  loop->setLatch(newLatch);
  loop->blocks().insert(newLatch);
  CFGAnalysisHHW().run(func, tp);
  // loop->getUniqueLatch()->dumpAsOpernd(std::cerr);
  // fix phi
  for (auto inst : loop->header()->insts()) {
    if (auto phiInst = inst->dynCast<PhiInst>()) {
      phiInst->replaceoldtonew(oldLatch, newLatch);
    }
  }
  return true;
}

bool extractLoopBody(Function* func,
                     Loop& loop,
                     IndVar* indVar,
                     TopAnalysisInfoManager* tp,
                     LoopBodyInfo& info) {
#ifdef DEBUG
  std::cerr << "extract loop body for: " << func->name() << std::endl;
  func->rename();
#endif
  if (func->attribute().hasAttr(FunctionAttribute::LoopBody)) {
    return false;
  }
  if (hasCall(&loop)) {
    return false;
  }
  // make sure loop is correct
  auto oldLatch = loop.getUniqueLatch();
#ifdef DEBUG
  loop.print(std::cerr);
  std::cerr << "old latch: ";
  loop.getUniqueLatch()->dumpAsOpernd(std::cerr);
  std::cerr << std::endl;
  indVar->print(std::cerr);
#endif

  moveNext2NewLatch(func, &loop, indVar, tp);

#ifdef DEBUG
  loop.print(std::cerr);
  std::cerr << "new latch: ";
  loop.getUniqueLatch()->dumpAsOpernd(std::cerr);
  std::cerr << std::endl;

  indVar->print(std::cerr);
#endif
  assert((loop.latchs().size() == 1) && "Loop must have exactly one latch");
  if (loop.header() == loop.getUniqueLatch() and loop.exits().size() != 1) {
    // header == latch, no loop body
    // only support loop with one exit
    return false;
  }
  // only support 2 phi insts: 1 for indvar, 1 for giv
  size_t phiCount = 0;
  for (auto inst : loop.header()->insts()) {
    if (inst->isa<PhiInst>()) {
      phiCount++;
    }
  }
  if (phiCount > 2) return false;

  for (auto block : loop.blocks()) {
    // if (block == loop.getUniqueLatch()) continue; cmmc
    if (block == loop.header()) continue;

    for (auto next : block->next_blocks()) {
      if (not loop.contains(next)) {
        // std::cerr << block->name() << "->" << next->name() << " is not in loop" << std::endl;
        return false;
      }
    }
  }

  // first phi inst != loop.inductionVar, giv = that phi inst
  // global induction var, such as n
  PhiInst* giv = nullptr;
  for (auto inst : loop.header()->insts()) {
    if (inst->isa<PhiInst>() and inst != indVar->phiinst()) {
      giv = inst->dynCast<PhiInst>();
    } else {
      break;
    }
  }
  // if giv

  // not giv

  std::unordered_set<Value*> allowedToBeUsedByOuter;

  allowedToBeUsedByOuter.insert(indVar->phiinst());
  // allowedToBeUsedByOuter.insert(loop.next)?

  // only indvar, next, giv allowed to be used by outer
  // other inst in loop should not be used by outer
  for (auto block : loop.blocks()) {
    for (auto inst : block->insts()) {
      if (allowedToBeUsedByOuter.count(inst)) continue;
      for (auto user_use : inst->uses()) {
        auto userInst = user_use->user()->dynCast<Instruction>();
        if (loop.blocks().count(userInst->block())) {
          continue;
        } else {
          return false;
        }
      }
    }
  }

  // independent
  // std::unordered_map<Value*, uint32_t> loadStoreMap;
  // for (auto block : loop.blocks()) {
  //   for (auto inst : block->insts()) {
  //     if (inst->isTerminator()) continue;
  //     if (auto loadInst = inst->dynCast<LoadInst>()) {
  //       const auto ptr = loadInst->ptr();
  //     } else if (auto storeInst = inst->dynCast<StoreInst>()) {
  //       const auto ptr = storeInst->ptr();
  //     }
  //     // TODO:
  //   }
  // }
  // std::vector<std::pair<Instruction*, Instruction*>> workList;
  // for (auto [k, v] : loadStoreMap) {
  //   if (v == 3) {
  //     // TODO:
  //   }
  // }

  auto funcType = FunctionType::gen(Type::void_type(), {});

  auto bodyFunc = func->module()->addFunction(funcType, getUniqueID());
  bodyFunc->attribute().addAttr(FunctionAttribute::LoopBody);
  // some operand used in loop must be passed by function arg, add to val2arg
  std::unordered_map<Value*, Value*> val2arg;

  // indvar phi -> body func first arg
  val2arg.emplace(indVar->phiinst(), bodyFunc->new_arg(indVar->phiinst()->type(), "indvar_arg"));

  // if giv, giv -> body func second arg
  if (giv) {
    val2arg.emplace(giv, bodyFunc->new_arg(giv->type()));
  }

  // duplicate cmp, true
  // for (auto block : loop.blocks()) {
  //   // TODO
  //   auto branchInst = block->terminator()->dynCast<BranchInst>();
  //   assert(branchInst);
  //   if (branchInst->is_cond()) {
  //     auto cond = branchInst->cond()->dynCast<Instruction>();
  //     // if cond is in loop, skip
  //     if (loop.blocks().count(cond->block())) continue;
  //     // TODO:
  //   }
  // }
  // std::cerr << "for all operands in loop, add to val2arg or pass by function arg" << std::endl;
  for (auto block : loop.blocks()) {
#ifdef DEBUG
    block->dumpAsOpernd(std::cerr);
    std::cerr << std::endl;
#endif
    if (block == loop.header() or block == loop.getUniqueLatch()) continue;
    for (auto inst : block->insts()) {
#ifdef DEBUG
      inst->print(std::cerr);
      std::cerr << std::endl;
#endif
      for (auto opuse : inst->operands()) {
        auto op = opuse->value();
#ifdef DEBUG
        op->dumpAsOpernd(std::cerr);
        std::cerr << std::endl;
#endif
        if (op->type()->isLabel()) continue;
        if (val2arg.count(op)) continue;  // already mapped
        if (op->dynCast<ConstantValue>() or op->dynCast<GlobalVariable>()) {
          continue;  // constants and global variables can be used directly
        }
        if (auto opInst = op->dynCast<Instruction>()) {
          if (loop.blocks().count(opInst->block())) continue;
        }
        // else, this op must pass by function arg, add to val2arg
        val2arg.emplace(op, bodyFunc->new_arg(op->type(), op->name() + "_arg"));
      }
    }
  }
#ifdef DEBUG
  for (auto [val, arg] : val2arg) {
    std::cerr << "val: ";
    val->dumpAsOpernd(std::cerr);
    std::cerr << " -> ";
    std::cerr << "arg: ";
    arg->dumpAsOpernd(std::cerr);
    std::cerr << std::endl;
  }
#endif
  bodyFunc->updateTypeFromArgs();

  std::unordered_map<Value*, Value*> arg2val;

  // replace operands used in loop with corresponding args
  // update use
  // std::cerr << "replace operands used in loop with corresponding args" << std::endl;
  for (auto [val, arg] : val2arg) {
    arg2val.emplace(arg, val);
    auto uses = val->uses();  // avoid invalidating use iterator
#ifdef DEBUG
    std::cerr << "val: ";
    val->dumpAsOpernd(std::cerr);
    std::cerr << ", with uses size: " << uses.size() << std::endl;
#endif
    for (auto use : uses) {
      const auto userInst = use->user()->dynCast<Instruction>();
#ifdef DEBUG
      userInst->print(std::cerr);
      std::cerr << std::endl;
#endif
      // exclude head and iterInst
      if (userInst->block() == loop.header() or userInst->block() == loop.getUniqueLatch()) continue;
      if (userInst == indVar->iterInst()) continue;
      if (loop.blocks().count(userInst->block())) {
#ifdef DEBUG
        std::cerr << "replace operand " << val->name() << " with arg " << arg->name() << std::endl;
#endif
        // userInst is in loop, replace operand with arg
        const auto idx = use->index();
        userInst->setOperand(idx, arg);
      }
      // std::cerr << "after replace, val: " << val->uses().size() << std::endl;
    }
  }

  // realArgs used by call inst
  std::vector<Value*> callRealArgs;
  for (auto arg : bodyFunc->args()) {
    callRealArgs.push_back(arg2val[arg]);
  }

  // construct bodyFunc blocks
  // push header.next as loop_body's entry
  std::unordered_set<BasicBlock*> removeWorkList;
  for (auto next : loop.header()->next_blocks()) {
    if (loop.contains(next)) {
      bodyFunc->setEntry(next);
      bodyFunc->blocks().push_back(next);
      removeWorkList.insert(next);
      break;
    }
  }
  // other blocks in loop
  for (auto block : loop.blocks()) {
    // exclue head and latch
    if (block == loop.header() or block == loop.getUniqueLatch()) continue;
    if (block != bodyFunc->entry()) {
      block->set_parent(bodyFunc);
      bodyFunc->blocks().push_back(block);
      removeWorkList.insert(block);
    }
  }
  assert(bodyFunc->blocks().size() == (loop.blocks().size() - 2));
  // remove loop blocks from func
  func->blocks().remove_if([&](BasicBlock* block) { return removeWorkList.count(block); });

  assert(oldLatch->terminator()->isa<BranchInst>());

  IRBuilder builder;
  // oldLatch now is the new loop_body's exit
  oldLatch->insts().pop_back();
  // just return, caller will call next iter
  builder.set_pos(oldLatch, oldLatch->insts().end());
  builder.makeInst<ReturnInst>();
  bodyFunc->setExit(oldLatch);

  // header -> callBlock -> latch
  // fix branch relation
  auto callBlock = func->newBlock();
  auto headerBranch = loop.header()->terminator()->dynCast<BranchInst>();
  assert(loop.contains(headerBranch->iftrue()));  // true is jump in loop
  headerBranch->set_iftrue(callBlock);

  // buid callBlock: call loop_body + jump to latch
  builder.set_pos(callBlock, callBlock->insts().end());
  const auto callInst = builder.makeInst<CallInst>(bodyFunc, callRealArgs);
  builder.makeInst<BranchInst>(loop.getUniqueLatch());

  // fix constraints on entry and exit
  fixAllocaInEntry(*bodyFunc);

  tp->CallChange();
  tp->CFGChange(func);
  tp->IndVarChange(func);

  // fix cfg
  const auto fixFunction = [&](Function* function) {
    CFGAnalysisHHW().run(function, tp);
    blockSortDFS(*function, tp);
    // function->rename();
    // function->print(std::cerr);
  };

  std::cerr << "after extractLoopBody, func: " << func->name() << std::endl;
  fixFunction(func);
  fixFunction(bodyFunc);

  {
    // return LoopBodyInfo
    info.callInst = callInst;
    info.indVar = indVar;
    info.header = loop.header();
    info.body = callBlock;
    info.latch = loop.getUniqueLatch();
    info.preHeader = loop.getLoopPreheader();
  }
  return true;
}

}  // namespace pass