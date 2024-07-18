#include "pass/optimize/ADCE.hpp"

static std::queue<ir::Instruction*>workList;
static std::map<ir::BasicBlock*,bool>liveBB;
static std::map<ir::Instruction*,bool>liveInst;

namespace pass{
    
    void ADCE::run(ir::Function* func, topAnalysisInfoManager* tp){
        bool isCFGChange=false;
        bool isCallGraphChange=false;
        if(func->isOnlyDeclare())return ;
        pdctx=tp->getPDomTree(func);
        pdctx->refresh();
        liveBB.clear();
        liveInst.clear();
        assert(workList.empty() and "ADCE WorkList not empty before running!");
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
        //delete useless insts
        for(auto bb:func->blocks()){
            for(auto instIter=bb->insts().begin();instIter!=bb->insts().end();){
                auto inst=*instIter;
                instIter++;
                if(not liveInst[inst] and not dyn_cast<ir::BranchInst>(inst)){
                    if(dyn_cast<ir::CallInst>(inst))isCallGraphChange=true;
                    bb->force_delete_inst(inst);
                }
                    
            }
        }
        //delete bb
        for(auto bbIter=func->blocks().begin();bbIter!=func->blocks().end();){
            auto bb=*bbIter;
            bbIter++;
            if(not liveBB[bb] and bb!=func->entry()){
                func->forceDelBlock(bb);
                isCFGChange=true;
            }
        }
        //rebuild jmp
        //unnecessary to rebuild phi, because:
        //if a phi inst is alive, all its incoming bbs are alive
        for(auto bb:func->blocks()){
            auto terInst=dyn_cast<ir::BranchInst>(bb->terminator());
            if(terInst==nullptr)continue;
            if(terInst->is_cond()){
                // std::cerr<<"isCond:"<<terInst->is_cond()<<std::endl;
                // std::cerr<<"Operands size:"<<terInst->operands().size()<<std::endl;
                auto trueTarget=terInst->iftrue();
                auto falseTarget=terInst->iffalse();
                if(not liveBB[trueTarget]){
                    auto newTarget=getTargetBB(trueTarget);
                    assert(newTarget!=nullptr);
                    terInst->set_iftrue(newTarget);
                    ir::BasicBlock::block_link(bb,newTarget);
                }
                if(not liveBB[terInst->iffalse()]){
                    auto newTarget=getTargetBB(falseTarget);
                    assert(newTarget!=nullptr);
                    terInst->set_iffalse(newTarget);
                    ir::BasicBlock::block_link(bb,newTarget);
                }
                if(terInst->iffalse()==terInst->iftrue()){
                    auto newBr=new ir::BranchInst(terInst->iftrue());
                    newBr->setBlock(bb);
                    bb->force_delete_inst(terInst);
                    bb->emplace_back_inst(newBr);
                }
            }
            else{
                // std::cerr<<"isCond:"<<terInst->is_cond()<<std::endl;
                // std::cerr<<"Operands size:"<<terInst->operands().size()<<std::endl;
                auto jmpTarget=terInst->dest();
                if(not liveBB[jmpTarget]){
                    auto newTarget=getTargetBB(jmpTarget);
                    assert(newTarget!=nullptr);
                    terInst->set_dest(newTarget);
                    ir::BasicBlock::block_link(bb,newTarget);
                }
            }
        }
        if(isCFGChange)tp->CFGChange(func);
        if(isCallGraphChange)tp->CallChange();
    }

    ir::BasicBlock* ADCE::getTargetBB(ir::BasicBlock* bb){
        auto targetBB=bb;
        while(not liveBB[targetBB])targetBB=pdctx->ipdom(targetBB);
        return targetBB;
    }

    void ADCE::ADCEInfoCheck(ir::Function* func){
        using namespace std;
        cout<<"In Function "<<func->name()<<" :"<<endl;
        for(auto bb:func->blocks()){
            cout<<bb->name()<<" alive: "<<liveBB[bb]<<endl;
        }
    }
}