#include "pass/analysis/LCSSA.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>
namespace pass {
ir::Value* LCSSA::stack_pop(std::stack<ir::Value*>& stack) {
  if (stack.empty()) return ir::Constant::gen_undefine();
  return stack.top();
}
bool LCSSA::isUseout(ir::Instruction* inst, ir::Loop* L) {
  for (auto use : inst->uses()) {
    ir::Instruction* userinst = dyn_cast<ir::Instruction>(use->user());
    ir::BasicBlock* userbb = userinst->block();
    if (auto userphi = dyn_cast<ir::PhiInst>(userinst)) {
      userbb = userphi->getBlock(use->index() / 2);
    }

    if (!L->contains(userbb)) {
      return true;
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

  std::map<ir::Instruction*, int>
      useinstmap;  // 记录在循环外使用与其useidx的映射
  std::set<ir::Instruction*> definsts;  // 记录对inst的定义
  std::stack<ir::Value*> valstack;
  addDef(definsts, exitphi);
  addDef(definsts, inst);
  int idx = -1;
  for (auto use : inst->uses()) {
    idx++;
    ir::Instruction* userinst = dyn_cast<ir::Instruction>(use->user());
    ir::BasicBlock* userbb = userinst->block();
    if (L->contains(userbb)) continue;
    if (auto userphi = dyn_cast<ir::PhiInst>(userinst)) {
      userbb = userphi->getBlock(use->index() / 2);
    }

    if (!L->contains(userbb)) {
      useinstmap[userinst] = idx;
    }
  }
  useinstmap[exitphi] = -1;
  rename(valstack, exit, useinstmap, definsts, inst);
}

void LCSSA::rename(std::stack<ir::Value*>& stack, ir::BasicBlock* bb, std::map<ir::Instruction*, int> useInstrMap, std::set<ir::Instruction*> defInsts, ir::Value* old) {
  int cnt = 0;
  for (ir::Instruction* inst : bb->insts()) {
    
    // std::cerr << inst->isa<ir::PhiInst>() << std::endl;
    // std::cerr << (useInstrMap.find(inst) != useInstrMap.end()) << std::endl;
    if (!dyn_cast<ir::PhiInst>(inst) && (useInstrMap.find(inst) != useInstrMap.end())) {  // 说明phi是inst的使用者
      old->replaceUseWith(stack_pop(stack), useInstrMap[inst]);
      // std::cerr << "in" << std::endl;
    }
    if (defInsts.find(inst) != defInsts.end()) {
      stack.push(inst);  // 更新到达定义
      cnt++;
    }
  }

  //处理succbb
  int idx = 0;
  for (auto succ : bb->next_blocks()) {  // 更新succ的phi
    for (auto succinst : succ->insts()) {
      if (!dyn_cast<ir::PhiInst>(succinst)) {  // 没有phi就直接break
        break;
      }
      if (useInstrMap.find(succinst) != useInstrMap.end()) {
        if (useInstrMap[succinst] == -1) {
          old->replaceUseWith(stack_pop(stack), idx);
        } else {
          old->replaceUseWith(stack_pop(stack), useInstrMap[succinst]);
        }
      }
    }
    idx++;
  }

  for (ir::BasicBlock* domson : DT->domson(bb)) {
    rename(stack, domson, useInstrMap, defInsts, old);
  }

  for (int i = 0; i < cnt; i++)
    stack.pop();
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
    // assert(isLCSSAform(loop));
  }
  return;
}

// 判断循环是不是lcssa形式，遍历循环的所有指令，检查指令的所有use
// 如果use-user不是phi指令，则直接判断userbb是否在循环内
// 如果是phi指令，则判断phi指令使用inst对应的前驱块是否在循环中
bool LCSSA::isLCSSAform(ir::Loop* L) {
  for (auto bb : L->blocks())
    for (auto inst : bb->insts())
      for (auto use : inst->uses()) {
        ir::Instruction* userinst = dyn_cast<ir::Instruction>(use->user());
        ir::BasicBlock* userbb = userinst->block();
        if (auto userphi = dyn_cast<ir::PhiInst>(userinst)) {
          userbb = userphi->getBlock(use->index() / 2);
        }
        if (userbb != bb && !L->contains(userbb)) return false;
      }

  return true;
}
}  // namespace pass