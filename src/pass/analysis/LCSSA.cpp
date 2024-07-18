#include "pass/analysis/LCSSA.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>
namespace pass {
bool LCSSA::isUseout(ir::Instruction* inst, ir::Loop* L) {
  for (auto use : inst->uses()) {
    ir::Instruction* userinst = dyn_cast<ir::Instruction>(use->user());
    ir::BasicBlock* userbb = userinst->block();
    std::vector<ir::BasicBlock*>
        userbbVector;  // phi指令对同一个val的使用可能来自不同的bb
    userbbVector.push_back(userbb);
    if (auto userphi = dyn_cast<ir::PhiInst>(userinst)) {
      userbbVector.pop_back();
      for (size_t i = 0; i < userphi->getsize(); i++) {
        auto phival = userphi->getValue(i);
        auto phibb = userphi->getBlock(i);
        if (phival == inst) userbbVector.push_back(phibb);
      }
    }
    for (auto bb : userbbVector) {
      if (!L->contains(bb)) {
        return true;
      }
    }
  }
  return false;
}

void LCSSA::addDef(std::set<ir::Instruction*>& definsts,
                   ir::Instruction* inst) {
  if (definsts.find(inst) == definsts.end()) {
    definsts.insert(inst);
    if (auto phiinst = dyn_cast<ir::PhiInst>(inst)) {
      for (size_t i = 0; i < phiinst->getsize(); i++) {
        if (auto phival = dyn_cast<ir::Instruction>(phiinst->getValue(i))) {
          definsts.insert(phival);
        }
      }
    }
  }
}

void LCSSA::makeExitPhi(ir::Instruction* inst,
                        ir::BasicBlock* exit,
                        ir::Loop* L) {
  ir::PhiInst* exitphi = utils::make<ir::PhiInst>(exit, inst->type());
  exit->emplace_first_inst(exitphi);
  for (auto bb : exit->pre_blocks()) {
    exitphi->addIncoming(inst, bb);
  }

  std::map<ir::Instruction*, int> useinstmap;
  std::set<ir::Instruction*> definsts;
  std::stack<ir::Value*> valstack;
  addDef(definsts, exitphi);
  addDef(definsts, inst);
  int idx = 0;
  for (auto use : inst->uses()) {
    ir::Instruction* userinst = dyn_cast<ir::Instruction>(use->user());
    ir::BasicBlock* userbb = userinst->block();
    std::vector<ir::BasicBlock*>
        userbbVector;  // phi指令对同一个val的使用可能来自不同的bb
    userbbVector.push_back(userbb);
    if (auto userphi = dyn_cast<ir::PhiInst>(userinst)) {
      userbbVector.pop_back();
      for (size_t i = 0; i < userphi->getsize(); i++) {
        auto phival = userphi->getValue(i);
        auto phibb = userphi->getBlock(i);
        if (phival == inst) userbbVector.push_back(phibb);
      }
    }
    for (auto bb : userbbVector) {
      if (!L->contains(bb)) {
        useinstmap[userinst] = idx;  //?
      }
    }
    idx++;
  }
  useinstmap[exitphi] = -1;
  rename(valstack, exit, useinstmap, definsts);
}

ir::Value* LCSSA::stack_pop(std::stack<ir::Value*>& stack) {
  if (stack.empty()) return ir::Constant::gen_undefine();
  return stack.top();
}

void LCSSA::rename(std::stack<ir::Value*>& stack,
                   ir::BasicBlock* bb,
                   std::map<ir::Instruction*, int> useInstrMap,
                   std::set<ir::Instruction*> defInstrs) {
  int cnt = 0;
  for (auto inst : bb->insts()) {
    if (auto phi = dyn_cast<ir::PhiInst>(inst) &&
                   useInstrMap.find(inst) != useInstrMap.end()) {
      inst->adjustuse(stack_pop(stack), useInstrMap[inst]);
    }
    if (defInstrs.find(inst) != defInstrs.end()) {
      stack.push(inst);
      cnt++;
    }
  }
  int idx = 0;
  for (auto succ : bb->next_blocks()) {
    for (auto succinst : succ->insts()) {
      if (!dyn_cast<ir::PhiInst>(succinst)) {
        break;
      }
      if (useInstrMap.find(succinst) != useInstrMap.end()) {
        if (useInstrMap[succinst] == -1) {
          succinst->adjustuse(stack_pop(stack), idx);
        } else {
          succinst->adjustuse(stack_pop(stack), useInstrMap[succinst]);
        }
      }
    }
    idx++;
  }

  for (ir::BasicBlock* domson : DT->domson(bb)) {
    rename(stack, domson, useInstrMap, defInstrs);
  }

  for (int i = 0; i < cnt; i++)
    stack.pop();
}

bool LCSSA::isLCSSAform(ir::Loop* L) {
  for (auto bb : L->blocks())
    for (auto inst : bb->insts())
      for (auto use : inst->uses()) {
        ir::Instruction* userinst = dyn_cast<ir::Instruction>(use->user());
        ir::BasicBlock* userbb = userinst->block();
        std::vector<ir::BasicBlock*> userbbVector;
        userbbVector.push_back(userbb);
        if (auto userphi = dyn_cast<ir::PhiInst>(userinst)) {
          userbbVector.pop_back();
          for (size_t i = 0; i < userphi->getsize(); i++) {
            auto phival = userphi->getValue(i);
            auto phibb = userphi->getBlock(i);
            if (phival == inst) userbbVector.push_back(phibb);
          }
        }
        for (auto userbb : userbbVector) {
          if (userbb != bb && !L->contains(userbb)) return false;
        }
      }

  return true;
}
void LCSSA::runonloop(ir::Loop* L, loopInfo* LI, topAnalysisInfoManager* tp) {
  for (ir::BasicBlock* BB : L->blocks()) {
    for (ir::Instruction* inst : BB->insts()) {
      if (isUseout(inst, L)) {
        for (ir::BasicBlock* exit : L->exits()) {
          makeExitPhi(inst, exit, L);
        }
      }
    }
  }
}

void LCSSA::run(ir::Function* F, topAnalysisInfoManager* tp) {
  loopInfo* LI = tp->getLoopInfo(F);
  DT = tp->getDomTree(F);
  LI->refresh();
  auto loops = LI->loops();
  for (auto loop : loops) {
    runonloop(loop, LI, tp);
    assert(!isLCSSAform(loop)); 
  }
  return;
}
}  // namespace pass