#include "pass/optimize/ADCE.hpp"

static std::queue<ir::Instruction*>workList;
static std::map<ir::BasicBlock*,bool>liveBB;
static std::map<ir::Instruction*,bool>liveInst;

namespace pass{
    
    void ADCE::run(ir::Function* func, topAnalysisInfoManager* tp){
        if(not func->entry())return ;
        pdctx=tp->getPDomTree(func);
        pdctx->refresh();
        //初始化所有的inst和BB的live信息
        for(auto bb:func->blocks()){
            liveBB[bb]=false;
            for(auto inst:bb->insts()){
                liveInst[inst]=false;
                if(inst->isAggressiveAlive()){
                    workList.push(inst);
                }
            }
        }
        //工作表算法
        while(not workList.empty()){
            auto curInst=workList.front();
            auto curBB=curInst->block();
            workList.pop();
            if(liveInst[curInst])continue;
            //设置当前的inst为活, 以及其块
            liveInst[curInst]=true;
            liveBB[curBB]=true;
            auto curInstPhi=dyn_cast<ir::PhiInst>(curInst);
            //如果是phi,就要将其所有前驱BB的terminal置为活
            if(curInstPhi){
                for(int idx=0;idx<curInstPhi->getsize();idx++){
                    auto phibb=curInstPhi->getBlock(idx);
                    auto phibbTerminator=phibb->terminator();
                    if(phibbTerminator and not liveInst[phibbTerminator]){
                        workList.push(phibbTerminator);
                        liveBB[phibb]=true;
                    }
                }
            }
            for(auto cdgpredBB:pdctx->pdomfrontier(curBB)){//curBB->pdomFrontier
                auto cdgpredBBTerminator=cdgpredBB->terminator();
                if(cdgpredBBTerminator and not liveInst[cdgpredBBTerminator]){
                    workList.push(cdgpredBBTerminator);
                }
            }
            for(auto op:curInst->operands()){
                auto opInst=dyn_cast<ir::Instruction>(op->value());
                if(opInst and not liveInst[opInst])
                    workList.push(opInst);
            }
        }

    }
}