#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include "pass/analysis/ControlFlowGraph.hpp"
#include "pass/analysis/indvar.hpp"
#include "pass/optimize/LoopParallel.hpp"

#include <cassert>
#include <unordered_map>
#include <iostream>
#include <vector>
namespace pass {
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
bool extractLoopBody(ir::Function* func,
                     ir::Loop& loop,
                     ir::indVar* indVar,
                     TopAnalysisInfoManager* tp,
                     LoopBodyFuncInfo& info) {
  assert((loop.latchs().size() == 1) && "Loop must have exactly one latch");
  if (loop.header() == loop.getLoopLatch() and loop.exits().size() != 1) {
    // header == latch, no loop body
    // only support loop with one exit
    return false;
  }
  // only support 2 phi insts: 1 for indvar, 1 for giv
  size_t phiCount = 0;
  for (auto inst : loop.header()->insts()) {
    if (inst->isa<ir::PhiInst>()) {
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
  ir::PhiInst* giv = nullptr;
  for (auto inst : loop.header()->insts()) {
    if (inst->isa<ir::PhiInst>() and inst != indVar->phiinst()) {
      giv = inst->dynCast<ir::PhiInst>();
    } else {
      break;
    }
  }
  // if giv

  // not giv

  std::unordered_set<ir::Value*> allowedToBeUsedByOuter;

  allowedToBeUsedByOuter.insert(indVar->phiinst());
  // allowedToBeUsedByOuter.insert(loop.next)?

  // only indvar, next, giv allowed to be used by outer
  // other inst in loop should not be used by outer
  for (auto block : loop.blocks()) {
    for (auto inst : block->insts()) {
      if (allowedToBeUsedByOuter.count(inst)) continue;
      for (auto user_use : inst->uses()) {
        auto userInst = user_use->user()->dynCast<ir::Instruction>();
        if (loop.blocks().count(userInst->block())) {
          continue;
        } else {
          return false;
        }
      }
    }
  }

  // independent
  std::unordered_map<ir::Value*, uint32_t> loadStoreMap;
  for (auto block : loop.blocks()) {
    for (auto inst : block->insts()) {
      if (inst->isTerminator()) continue;
      if (auto loadInst = inst->dynCast<ir::LoadInst>()) {
        const auto ptr = loadInst->ptr();
      } else if (auto storeInst = inst->dynCast<ir::StoreInst>()) {
        const auto ptr = storeInst->ptr();
      }
      // TODO:
    }
  }
  std::vector<std::pair<ir::Instruction*, ir::Instruction*>> workList;
  for (auto [k, v] : loadStoreMap) {
    if (v == 3) {
      // TODO:
    }
  }

  auto funcType = ir::FunctionType::gen(ir::Type::void_type(), {});

  auto bodyFunc = func->module()->addFunction(funcType, "loop_body");

  // some operand used in loop must be passed by function arg, add to val2arg
  std::unordered_map<ir::Value*, ir::Value*> val2arg;

  // indvar phi -> body func first arg
  val2arg.emplace(indVar->phiinst(), bodyFunc->new_arg(indVar->phiinst()->type()));

  // if giv, giv -> body func second arg
  if (giv) {
    val2arg.emplace(giv, bodyFunc->new_arg(giv->type()));
  }

  // duplicate cmp, true
  for (auto block : loop.blocks()) {
    // TODO
    auto branchInst = block->terminator()->dynCast<ir::BranchInst>();
    assert(branchInst);
    if (branchInst->is_cond()) {
      auto cond = branchInst->cond()->dynCast<ir::Instruction>();
      // if cond is in loop, skip
      if (loop.blocks().count(cond->block())) continue;
      // TODO:
    }
  }

  for (auto block : loop.blocks()) {
    for (auto inst : block->insts()) {
      for (auto opuse : inst->operands()) {
        auto op = opuse->value();
        if (val2arg.count(op)) continue;  // already mapped
        if (op->dynCast<ir::ConstantValue>() or op->dynCast<ir::GlobalVariable>()) {
          continue;  // constants and global variables can be used directly
        }
        if (loop.blocks().count(op->dynCast<ir::Instruction>()->block())) continue;
        // else, this op must pass by function arg, add to val2arg
        val2arg.emplace(op, bodyFunc->new_arg(op->type()));
      }
    }
  }

  bodyFunc->updateTypeFromArgs();

  std::unordered_map<ir::Value*, ir::Value*> arg2val;

  // replace operands used in loop with corresponding args
  // update use
  for (auto [val, arg] : val2arg) {
    arg2val.emplace(arg, val);
    for (auto use : val->uses()) {
      const auto userInst = use->user()->dynCast<ir::Instruction>();
      if (loop.blocks().count(userInst->block())) {
        // userInst is in loop, replace operand with arg
        use->set_value(arg);
      }
    }
  }

  // realArgs used by call inst
  std::vector<ir::Value*> callRealArgs;
  for (auto arg : bodyFunc->args()) {
    callRealArgs.push_back(arg2val[arg]);
  }

  // construct bodyFunc blocks
  // push original loop header to bodyFunc blocks, as entry
  bodyFunc->blocks().push_back(loop.header());
  // other blocks in loop
  for (auto block : loop.blocks()) {
    block->set_parent(bodyFunc);
    if (block != loop.header()) {
      bodyFunc->blocks().push_back(block);
    }
  }
  // remove loop blocks from func
  func->blocks().remove_if([&](ir::BasicBlock* block) { return loop.blocks().count(block); });

  ir::IRBuilder builder;

  builder.set_pos(loop.getLoopLatch());

  // make return inst?

  auto newLoop = func->newBlock();
  builder.set_pos(newLoop);
  for (auto inst : loop.header()->insts()) {
    if (inst->isa<ir::PhiInst>()) {
      newLoop->emplace_first_inst(inst);
      // remove from loop header?
    } else
      break;
  }
  const auto callInst = builder.makeInst<ir::CallInst>(bodyFunc, callRealArgs);
  // const auto

  return true;
}

}  // namespace pass