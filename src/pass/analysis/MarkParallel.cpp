#include "pass/analysis/MarkParallel.hpp"
#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
using namespace pass;

std::string markParallel::name() const {
    return "mark parallel";
}

void markParallel::run(ir::Function* func,TopAnalysisInfoManager* tp){
    dpctx=tp->getDepInfo(func);
    domctx=tp->getDomTreeWithoutRefresh(func);
    lpctx=tp->getLoopInfoWithoutRefresh(func);
    cgctx=tp->getCallGraphWithoutRefresh();
    sectx=tp->getSideEffectInfoWithoutRefresh();
    idvctx=tp->getIndVarInfoWithoutRefresh(func);
    parctx=tp->getParallelInfo(func);
    for(auto loop:lpctx->loops()){
        runOnLoop(loop);
    }
    printParallelInfo(func);
}

void markParallel::runOnLoop(ir::Loop* lp){
    auto lpDepInfo=dpctx->getLoopDependenceInfo(lp);
    bool isParallelConcerningArray=lpDepInfo->getIsParallel();
    auto defaultIdv=idvctx->getIndvar(lp);
    if(isParallelConcerningArray==false){
        parctx->setIsParallel(lp->header(),false);
        return;
    }
    for(auto bb:lp->blocks()){
        for(auto inst:bb->insts()){
            if(auto callInst=inst->dynCast<ir::CallInst>()){
                auto callee=callInst->callee();
                if(not sectx->isPureFunc(callee)){
                    parctx->setIsParallel(lp->header(),false);
                    return;
                }
            }
            else if(auto storeInst=inst->dynCast<ir::StoreInst>()){
                auto ptr=storeInst->ptr();
                if(ptr->dynCast<ir::GlobalVariable>()){
                    parctx->setIsParallel(lp->header(),false);
                    return;
                }
            }
        }
    }
    if(lp->header()->phi_insts().size()>1){//接下来就开始处理每一个phi
        for(auto pi:lp->header()->phi_insts()){
            auto phi=pi->dynCast<ir::PhiInst>();
            if(phi==defaultIdv->phiinst())continue;
            auto res=getResPhi(phi,lp);
            if(res==nullptr){
                parctx->setIsParallel(lp->header(),false);
                return;
            }
            parctx->setPhi(phi,res->isAdd,res->isSub,res->isMul,res->mod);
        }
    }
    parctx->setIsParallel(lp->header(),true);
    return;
}

void markParallel::printParallelInfo(ir::Function* func){
    std::cerr<<"In Function \""<<func->name()<<"\":"<<std::endl;
    for(auto lp:lpctx->loops()){
        using namespace std;
        cerr<<"Parallize Loop whose header is \""<<lp->header()->name()<<"\" :";
        if(parctx->getIsParallel(lp->header())){
            cerr<<"YES";
        }
        else{
            cerr<<"NO";
        }
        cerr<<endl;
    }

}

resPhi* markParallel::getResPhi(ir::PhiInst* phi,ir::Loop* lp){
    assert(phi->block()==lp->header());
    if(phi->isFloat32())return nullptr;
    auto lpPreheader=lp->getLoopPreheader();
    auto preheaderIncoming=phi->getvalfromBB(lpPreheader);
    if(lp->latchs().size()>1)return nullptr;
    auto latchIncoming=phi->getvalfromBB(*lp->latchs().begin());
    if(auto binaryLatchIncoming=latchIncoming->dynCast<ir::BinaryInst>()){
        auto pnewResPhi=new resPhi;
        pnewResPhi->phi=phi;
        pnewResPhi->isAdd=false;
        pnewResPhi->isSub=false;
        pnewResPhi->isMul=false;
        pnewResPhi->isModulo=false;
        auto binaryLatchIncomingInstId=binaryLatchIncoming->valueId();
        ir::Instruction* curInst;
        if(binaryLatchIncomingInstId==ir::vADD or binaryLatchIncomingInstId==ir::vMUL or binaryLatchIncomingInstId==ir::vSUB){
            pnewResPhi->isModulo=false;
            pnewResPhi->mod=nullptr;
            curInst=binaryLatchIncoming->dynCast<ir::Instruction>();
            if(binaryLatchIncomingInstId==ir::vADD){
                if(binaryLatchIncoming->lValue()==phi or binaryLatchIncoming->rValue()==phi){
                    pnewResPhi->isAdd=true;
                    return pnewResPhi;
                }
                return nullptr;
            }
            if(binaryLatchIncomingInstId==ir::vMUL){
                if(binaryLatchIncoming->lValue()==phi or binaryLatchIncoming->rValue()==phi){
                    pnewResPhi->isMul=true;
                    return pnewResPhi;
                }
                return nullptr;
            }
            if(binaryLatchIncomingInstId==ir::vSUB){
                if(binaryLatchIncoming->lValue()==phi){
                    pnewResPhi->isSub=true;
                    return pnewResPhi;
                }
                return nullptr;
            }
        }
        else if(binaryLatchIncomingInstId==ir::vSREM){
            pnewResPhi->isModulo=true;
            pnewResPhi->mod=binaryLatchIncoming->rValue();
            curInst=binaryLatchIncoming->lValue()->dynCast<ir::Instruction>();
            auto curInstId=curInst->valueId();
            if(curInstId==ir::vADD){
                if(binaryLatchIncoming->lValue()==phi or binaryLatchIncoming->rValue()==phi){
                    pnewResPhi->isAdd=true;
                    return pnewResPhi;
                }
                return nullptr;
            }
            if(curInstId==ir::vMUL){
                if(binaryLatchIncoming->lValue()==phi or binaryLatchIncoming->rValue()==phi){
                    pnewResPhi->isMul=true;
                    return pnewResPhi;
                }
                return nullptr;
            }
            if(curInstId==ir::vSUB){
                if(binaryLatchIncoming->lValue()==phi){
                    pnewResPhi->isSub=true;
                    return pnewResPhi;
                }
                return nullptr;
            }
            return nullptr;

        }
        else{
            delete pnewResPhi;
            return nullptr;
        }
    }
    else {
        return nullptr;
    }
    return nullptr;
}

bool markParallel::isSimplyLpInvariant(ir::Loop* lp,ir::Value* val){
    if(auto constVal=val->dynCast<ir::ConstantInteger>()){
        return true;
    }
    if(auto arg=val->dynCast<ir::Argument>()){
        return true;
    }
    if(auto inst=val->dynCast<ir::Instruction>()){
        return domctx->dominate(inst->block(),lp->header()) and inst->block()!=lp->header();
    }
    return false;
}