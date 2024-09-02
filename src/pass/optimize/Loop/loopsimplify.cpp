#include "pass/optimize/optimize.hpp"
#include "pass/optimize/loopsimplify.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>

namespace pass {
ir::BasicBlock* loopsimplify::insertUniqueBackedgeBlock(
    ir::Loop* L,
    ir::BasicBlock* preheader,
    TopAnalysisInfoManager* tp) {
  ir::BasicBlock* header = L->header();
  ir::Function* F = header->function();

  if (!preheader) return nullptr;

  std::vector<ir::BasicBlock*> BackedgeBBs;
  for (ir::BasicBlock* BB : header->pre_blocks()) {
    if (BB != preheader)  // L->contains(BB)
      BackedgeBBs.push_back(BB);
  }

  ir::BasicBlock* BEBB = F->newBlock();
  ir::BranchInst* jmp = new ir::BranchInst(header, BEBB);
  BEBB->emplace_back_inst(jmp);
  ir::BasicBlock::block_link(BEBB, header);

  for (auto& inst : header->insts()) {
    if (ir::PhiInst* phiinst = dyn_cast<ir::PhiInst>(inst)) {
      ir::PhiInst* BEphi = new ir::PhiInst(BEBB, phiinst->type());
      BEBB->emplace_first_inst(BEphi);
      bool hasuniqueval = true;
      ir::Value* uniqueval = nullptr;
      for (ir::BasicBlock* BB : BackedgeBBs) {
        ir::Value* val = phiinst->getvalfromBB(BB);
        BEphi->addIncoming(val, BB);
        if (hasuniqueval) {
          if (!uniqueval) {
            uniqueval = val;
          } else if (uniqueval != val) {
            hasuniqueval = false;
          }
        }
      }
      for (ir::BasicBlock* BB : BackedgeBBs) {
        phiinst->delBlock(BB);
      }
      phiinst->addIncoming(BEphi, BEBB);
      if (hasuniqueval) {
        BEphi->replaceAllUseWith(uniqueval);
        BEBB->delete_inst(BEphi);
      }
    }
  }

  for (ir::BasicBlock* BB : BackedgeBBs) {
    auto inst = BB->insts().back();
    ir::BasicBlock::delete_block_link(BB, header);
    ir::BasicBlock::block_link(BB, BEBB);
    if (ir::BranchInst* brinst = dyn_cast<ir::BranchInst>(inst)) {
      brinst->replaceDest(header, BEBB);
    }
  }
  L->blocks().insert(BEBB);
  return BEBB;
}

ir::BasicBlock* loopsimplify::insertUniquePreheader(
    ir::Loop* L,
    TopAnalysisInfoManager* tp) {
  ir::BasicBlock* header = L->header();
  ir::Function* F = header->function();
  ir::BasicBlock* preheader = L->getloopPredecessor();
  if (!preheader) {  // 有多个循环外的preheader，则插入一个新的汇合块
    std::vector<ir::BasicBlock*> preBBs;
    for (ir::BasicBlock* BB : header->pre_blocks()) {
      if (!L->contains(BB)) preBBs.push_back(BB);
    }
    ir::BasicBlock* BEBB = F->newBlock();
    ir::BranchInst* jmp = new ir::BranchInst(header, BEBB);
    BEBB->emplace_back_inst(jmp);
    ir::BasicBlock::block_link(BEBB, header);
    for (auto& inst : header->insts()) {
      if (ir::PhiInst* phiinst = dyn_cast<ir::PhiInst>(inst)) {
        ir::PhiInst* BEphi = new ir::PhiInst(BEBB, phiinst->type());
        BEBB->emplace_first_inst(BEphi);
        bool hasuniqueval = true;
        ir::Value* uniqueval = nullptr;
        for (ir::BasicBlock* BB : preBBs) {
          ir::Value* val = phiinst->getvalfromBB(BB);
          BEphi->addIncoming(val, BB);
          // if (hasuniqueval) {
          //   if (!uniqueval) {
          //     uniqueval = val;
          //   } else if (uniqueval != val) {
          //     hasuniqueval = false;
          //   }
          // }
        }
        for (ir::BasicBlock* BB : preBBs) {
          phiinst->delBlock(BB);
        }
        phiinst->addIncoming(BEphi, BEBB);
        // if (hasuniqueval) {
        //   BEphi->replaceAllUseWith(uniqueval);
        //   BEBB->delete_inst(BEphi);
        // }
      }
    }
    for (ir::BasicBlock* BB : preBBs) {
      auto inst = BB->insts().back();
      ir::BasicBlock::delete_block_link(BB, header);
      ir::BasicBlock::block_link(BB, BEBB);
      if (ir::BranchInst* brinst = dyn_cast<ir::BranchInst>(inst)) {
        brinst->replaceDest(header, BEBB);
      }
    }
    L->blocks().insert(BEBB);
    return BEBB;
  }

  ir::BasicBlock* newpre = F->newBlock();
  ir::BranchInst* jmp = new ir::BranchInst(header, newpre);
  newpre->emplace_back_inst(jmp);
  ir::BranchInst* br = dyn_cast<ir::BranchInst>(preheader->insts().back());
  br->replaceDest(header, newpre);
  ir::BasicBlock::delete_block_link(preheader, header);
  ir::BasicBlock::block_link(newpre, header);
  ir::BasicBlock::block_link(preheader, newpre);

  for (auto inst : header->insts()) {
    if (ir::PhiInst* phiinst = dyn_cast<ir::PhiInst>(inst)) {
      phiinst->replaceoldtonew(preheader, newpre);
    }
  }
  L->blocks().insert(newpre);
  return newpre;
}

void loopsimplify::insertUniqueExitBlock(ir::Loop* L,
                                         TopAnalysisInfoManager* tp) {
  ir::Function* F = L->header()->function();
  std::vector<ir::BasicBlock*> InLoopPred;
  for (ir::BasicBlock* exit : L->exits()) {
    if (exit->pre_blocks().size() > 1) {
      InLoopPred.clear();
      for (ir::BasicBlock* pred : exit->pre_blocks()) {
        if (L->contains(pred)) {
          InLoopPred.push_back(pred);
        }
      }

      for (ir::BasicBlock* pred : InLoopPred) {
        ir::BasicBlock* newBB = F->newBlock();
        ir::BranchInst* jmp = new ir::BranchInst(exit, newBB);
        newBB->emplace_back_inst(jmp);
        ir::BranchInst* br = dyn_cast<ir::BranchInst>(pred->insts().back());
        br->replaceDest(exit, newBB);
        ir::BasicBlock::delete_block_link(pred, exit);
        ir::BasicBlock::block_link(pred, newBB);
        ir::BasicBlock::block_link(newBB, exit);
        for (auto inst : exit->insts()) {
          if (ir::PhiInst* phiinst = dyn_cast<ir::PhiInst>(inst)) {
            phiinst->replaceoldtonew(pred, newBB);
          }
        }
      }
      // 需要更新_exits
    }
  }
  return;
}

bool loopsimplify::simplifyOneLoop(ir::Loop* L, TopAnalysisInfoManager* tp) {
  bool changed = false;
  // if (L->isLoopSimplifyForm()) return false;
  // 如果有多条回边
  //preheader不能为F的entry
  ir::BasicBlock* entry = L->header()->function()->entry();
  ir::BasicBlock* preheader = L->getLoopPreheader();
  ir::BasicBlock* LoopLatch = L->getLoopLatch();
  if (!preheader || preheader == entry) {
    preheader = insertUniquePreheader(L, tp);
    if (preheader) changed = true;
  }

  if (!LoopLatch) {
    LoopLatch = insertUniqueBackedgeBlock(L, preheader, tp);
    if (LoopLatch) changed = true;
  }

  if (!L->hasDedicatedExits()) {
    insertUniqueExitBlock(L, tp);
    changed = true;
  }

  return changed;
}

void loopsimplify::run(ir::Function* func, TopAnalysisInfoManager* tp) {
  if (func->isOnlyDeclare()) return;
    // func->rename();
  // func->print(std::cerr);
  LoopInfo* LI = tp->getLoopInfo(func);
  func->rename();
      // func->rename();
  // func->print(std::cerr);
  auto loops = LI->loops();
  bool changed = false;
  for (auto L : loops) {
    changed |= simplifyOneLoop(L, tp);
    if (!L->isLoopSimplifyForm()) {
      assert("loop is not in simplify form");
    }
  }
  if (changed) {
    // update loopinfo
    tp->CFGChange(func);
  }
  // func->rename();
  // func->print(std::cerr);

  return;
}
}  // namespace pass