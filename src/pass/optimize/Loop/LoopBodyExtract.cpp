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
void LoopBodyExtract::run(Function* func, TopAnalysisInfoManager* tp) {
  runImpl(func, tp);
}

void LoopBodyExtract::runImpl(Function* func, TopAnalysisInfoManager* tp) {
  CFGAnalysisHHW().run(func, tp);  // refresh CFG

  auto lpctx = tp->getLoopInfo(func);
  auto indVarInfo = tp->getIndVarInfo(func);

  for (auto loop : lpctx->loops()) {
    loop->print(std::cerr);
    const auto indVar = indVarInfo->getIndvar(loop);
    const auto step = indVar->getStep()->i32();
    indVar->print(std::cerr);

    if (step != 1) continue;  // only support step = 1

    LoopBodyFuncInfo loopBodyInfo;
    if (not extractLoopBody(func, *loop, indVar, tp, loopBodyInfo /* ret */)) continue;

    // std::cerr << "extracted loop body: " << loopBodyInfo.bodyFunc->name() << std::endl;
  }
  tp->CallChange();
  tp->CFGChange(func);
  tp->IndVarChange(func);
  // fix cfg
  CFGAnalysisHHW().run(func, tp);  // refresh CFG
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
  --> newLoop --> loop.next
  |     |
  |     v
  -- callBlock


newLoop:
  - i = phi [i0, other], [i.next, newLoop]
  -
 */
// need iterInst in loop.latch
bool extractLoopBody(Function* func,
                     Loop& loop,
                     IndVar* indVar,
                     TopAnalysisInfoManager* tp,
                     LoopBodyFuncInfo& info) {
  // make sure loop is correct
  auto oldLatch = loop.getLoopLatch();
  const auto splitLatch = [&]() {
    assert(oldLatch->insts().size() >= 2);
    auto newLatch = func->newBlock();
    newLatch->set_name("new_latch");
    newLatch->set_idx(func->blocks().size());
    auto lastIter = std::prev(oldLatch->insts().end());
    auto lastButOneIter = std::prev(lastIter);
    assert(*lastButOneIter == indVar->iterInst());
    oldLatch->move_inst(*lastButOneIter);
    newLatch->emplace_back_inst(*lastButOneIter);
    oldLatch->move_inst(*lastIter);
    newLatch->emplace_back_inst(*lastIter);
    IRBuilder builder;
    builder.set_pos(oldLatch, oldLatch->insts().end());
    builder.makeInst<BranchInst>(newLatch);
    loop.setLatch(newLatch);
    loop.blocks().insert(newLatch);
    CFGAnalysisHHW().run(func, tp);
    // fix phi
    for (auto inst : loop.header()->insts()) {
      if (auto phiInst = inst->dynCast<PhiInst>()) {
        phiInst->replaceoldtonew(oldLatch, newLatch);
      }
    }
  };
  splitLatch();
  loop.print(std::cerr);
  std::cerr << "new latch: ";
  loop.getLoopLatch()->dumpAsOpernd(std::cerr);
  std::cerr << std::endl;
  assert((loop.latchs().size() == 1) && "Loop must have exactly one latch");
  if (loop.header() == loop.getLoopLatch() and loop.exits().size() != 1) {
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
    // if (block == loop.getLoopLatch()) continue; cmmc
    if (block == loop.header()) continue;

    for (auto next : block->next_blocks()) {
      if (not loop.contains(next)) {
        std::cerr << block->name() << "->" << next->name() << " is not in loop" << std::endl;
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

  auto bodyFunc = func->module()->addFunction(funcType, "loop_body");

  // some operand used in loop must be passed by function arg, add to val2arg
  std::unordered_map<Value*, Value*> val2arg;

  // indvar phi -> body func first arg
  val2arg.emplace(indVar->phiinst(), bodyFunc->new_arg(indVar->phiinst()->type()));

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
  std::cerr << "for all operands in loop, add to val2arg or pass by function arg" << std::endl;
  for (auto block : loop.blocks()) {
    block->dumpAsOpernd(std::cerr);
    if (block == loop.header() or block == loop.getLoopLatch()) continue;
    for (auto inst : block->insts()) {
      inst->print(std::cerr);
      std::cerr << std::endl;
      for (auto opuse : inst->operands()) {
        auto op = opuse->value();
        op->dumpAsOpernd(std::cerr);
        std::cerr << std::endl;
        if (op->type()->isLabel()) continue;
        if (val2arg.count(op)) continue;  // already mapped
        if (op->dynCast<ConstantValue>() or op->dynCast<GlobalVariable>()) {
          continue;  // constants and global variables can be used directly
        }
        if (loop.blocks().count(op->dynCast<Instruction>()->block())) continue;
        // else, this op must pass by function arg, add to val2arg
        val2arg.emplace(op, bodyFunc->new_arg(op->type()));
      }
    }
  }

  bodyFunc->updateTypeFromArgs();

  std::unordered_map<Value*, Value*> arg2val;

  // replace operands used in loop with corresponding args
  // update use
  std::cerr << "replace operands used in loop with corresponding args" << std::endl;
  for (auto [val, arg] : val2arg) {
    val->dumpAsOpernd(std::cerr);
    std::cerr << std::endl;
    arg2val.emplace(arg, val);
    auto uses = val->uses();  // avoid invalidating use iterator
    for (auto use : uses) {
      const auto userInst = use->user()->dynCast<Instruction>();
      userInst->print(std::cerr);
      std::cerr << std::endl;
      // exclude head and iterInst
      if (userInst->block() == loop.header() or userInst->block() == loop.getLoopLatch()) continue;
      if (userInst == indVar->iterInst()) continue;
      if (loop.blocks().count(userInst->block())) {
        // userInst is in loop, replace operand with arg
        std::cerr << "replace operand " << val->name() << " with arg " << arg->name() << std::endl;
        // use->set_value(arg);
        const auto idx = use->index();
        userInst->setOperand(idx, arg);
      }
      std::cerr << "after replace, val: " << val->uses().size() << std::endl;
    }
  }

  // realArgs used by call inst
  std::vector<Value*> callRealArgs;
  for (auto arg : bodyFunc->args()) {
    callRealArgs.push_back(arg2val[arg]);
  }

  // construct bodyFunc blocks
  // push original loop header to bodyFunc blocks, as entry
  // bodyFunc->blocks().push_back(loop.header());
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
    if (block == loop.header() or block == loop.getLoopLatch()) continue;
    if (block != bodyFunc->entry()) {
      block->set_parent(bodyFunc);
      bodyFunc->blocks().push_back(block);
      removeWorkList.insert(block);
    }
  }
  // remove loop blocks from func
  func->blocks().remove_if([&](BasicBlock* block) { return removeWorkList.count(block); });

  assert(oldLatch->terminator()->isa<BranchInst>());

  oldLatch->insts().pop_back();

  IRBuilder builder;
  // make return inst?
  builder.set_pos(oldLatch, oldLatch->insts().end());
  builder.makeInst<ReturnInst>();
  bodyFunc->setExit(oldLatch);

  auto newLoop = func->newBlock();
  auto headerBranch = loop.header()->terminator()->dynCast<BranchInst>();
  assert(loop.contains(headerBranch->iftrue()));
  headerBranch->set_iftrue(newLoop);
  // builder.set_pos(loop.header(), loop.header()->insts().end());
  // builder.makeInst<BranchInst>(newLoop);

  // buid newLoop
  builder.set_pos(newLoop, newLoop->insts().end());
  const auto callInst = builder.makeInst<CallInst>(bodyFunc, callRealArgs);
  builder.makeInst<BranchInst>(loop.getLoopLatch());

  // fix constraints on entry and exit

  fixAllocaInEntry(*bodyFunc);

  tp->CallChange();
  tp->CFGChange(func);
  tp->IndVarChange(func);

  // fix cfg
  CFGAnalysisHHW().run(func, tp);
  CFGAnalysisHHW().run(bodyFunc, tp);

  std::cerr << "after extractLoopBody, func: " << std::endl;
  func->rename();
  func->print(std::cerr);
  std::cerr << std::endl;
  std::cerr << "bodyFunc: " << std::endl;
  bodyFunc->rename();
  bodyFunc->print(std::cerr);
  // return
  info.callInst = callInst;
  info.indVar = indVar;
  info.header = loop.header();
  info.body = newLoop;
  info.latch = loop.getLoopLatch();
  info.preHeader = loop.getLoopPreheader();
  // update loop info:
  return true;
}

}  // namespace pass