#include "pass/analysis/LCSSA.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>
namespace pass {
std::vector<ir::PhiInst*> LCSSA::insertphi(ir::Instruction* inst, ir::Loop* L, std::vector<ir::BasicBlock*>& DefsBlock, std::vector<ir::Instruction*>& Defs) {
  std::set<ir::BasicBlock*> Phiset;
  std::vector<ir::BasicBlock*> W;
  ir::PhiInst* newphi;
  ir::BasicBlock* x;
  std::vector<ir::PhiInst*> newphis;

  for (ir::BasicBlock* BB : DefsBlock){
    W.push_back(BB);
  }
  while (!W.empty()){
    x = W.back();
    W.pop_back();
    for (ir::BasicBlock* Y : DT->domfrontier(x)){//迭代支配边界插入newphi
      if (Phiset.find(Y) == Phiset.end()){
        // newphi = utils::make<ir::PhiInst>(Y, inst->type());
        newphi = new ir::PhiInst(Y, inst->type());
        Y->emplace_first_inst(newphi);
        Defs.push_back(newphi);
        newphis.push_back(newphi);
        DefsBlock.push_back(Y);
        Phiset.insert(Y);
        if (find(DefsBlock.begin(), DefsBlock.end(), Y) == DefsBlock.end() && (!L->contains(Y)))
          W.push_back(Y);
      }  
    }
  }
}

void LCSSA::rename(ir::Instruction* usee ,ir::Loop* L, ir::Function* F, std::vector<ir::BasicBlock*> DefsBlock, std::vector<ir::Instruction*> Defs, std::vector<ir::PhiInst*> newphis){
  std::stack<std::pair<ir::BasicBlock*, ir::Value*>> Worklist;
  std::set<ir::BasicBlock*> VisitedSet;
  ir::BasicBlock *BB;
  ir::Value* incoming;
  Worklist.push({F->entry(), ir::Constant::gen_undefine()});
  
  while(!Worklist.empty()){
    BB = Worklist.top().first;
    incoming = Worklist.top().second;
    Worklist.pop();
    bool inloop = L->contains(BB);
    if (VisitedSet.find(BB) != VisitedSet.end()) 
      continue;
    else  
      VisitedSet.insert(BB);

    for (auto inst : BB->insts()){
      if (find(Defs.begin(), Defs.end(), inst) != Defs.end() || (usee == inst)){//如果是定值，则更新incoming
        incoming = inst;
      }
      // else{//遍历指令的operand，检查是否有usee，如果有则用incoming替换，setoperand
      //   if(!inloop && (!inst->dynCast<ir::PhiInst>()) ){
      //     for (auto op : inst->operands()){
      //       if (op->value()->dynCast<ir::Instruction>() == usee){
      //         if (!incoming->type()->isUndef())
      //           inst->setOperand(op->index(), incoming);
      //       }
      //     }
      //   }
        
      // }
    }

    for (auto SuccBB : BB->next_blocks()){
      if (ir::PhiInst* phi = SuccBB->insts().back()->dynCast<ir::PhiInst>()){
        if (std::find(newphis.begin(), newphis.end(), phi) != newphis.end()){
          phi->addIncoming(incoming, BB);
        }
      }
      // if (!L->contains(SuccBB)){
      //   for (auto sinst : SuccBB->insts()){
      //     if (sinst->dynCast<ir::PhiInst>()){
      //       for (auto op : sinst->operands()){
      //         if (op->value()->dynCast<ir::Instruction>() == usee){
      //           if (!incoming->type()->isUndef())
      //             sinst->setOperand(op->index(), incoming);
      //         }
      //       }
      //     }
      //   }
      // }
    }

    for (auto dombb : DT->domson(BB)){
      Worklist.push({dombb, incoming});
    }
  }
}

void LCSSA::promoteLCSSA(ir::Instruction* usee,ir::Loop* L) {
  std::vector<ir::BasicBlock*> DefsBlock;
  std::vector<ir::Instruction*> Defs;

  for (auto exitbb : L->exits()){//为每个exitbb插入一个phi，更新defs和defbb
      ir::PhiInst* exitphi = utils::make<ir::PhiInst>(exitbb, usee->type());
      exitbb->emplace_first_inst(exitphi);
      for (auto prebb : exitbb->pre_blocks()){
        exitphi->addIncoming(usee, prebb);
      }
      DefsBlock.push_back(exitbb);
      Defs.push_back(exitphi);
  }
  

  ir::Function* F = usee->block()->function();
  std::vector<ir::PhiInst*> newphis = insertphi(usee, L, DefsBlock, Defs);
  rename(usee, L, F, DefsBlock, Defs, newphis);


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

void LCSSA::runonloop(ir::Loop* L, loopInfo* LI, topAnalysisInfoManager* tp) {
  // for (ir::Loop* subloop : L->subLoops()) {
  //   runonloop(subloop, LI, tp);
  // }
  for (ir::BasicBlock* BB : L->blocks()) 
    // if (L->getloopfor(BB)) 
      for (ir::Instruction* inst : BB->insts()) 
        if (isUseout(inst, L)) 
          // for (ir::BasicBlock* exit : L->exits()) 
          promoteLCSSA(inst, L);
        
      
    
  
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