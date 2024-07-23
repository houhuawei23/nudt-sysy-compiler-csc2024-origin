#include "pass/analysis/indvar.hpp"

using namespace pass;

void indVarAnalysis::run(ir::Function* func,topAnalysisInfoManager* tp) {
    if(func->isOnlyDeclare())return;
    lpctx=tp->getLoopInfo(func);
    lpctx->refresh();
    ivctx=tp->getIndVarInfo(func);
    ivctx->clearAll();
    func->rename();
    for(auto lp:lpctx->loops()){
        auto lpHeader=lp->header();
        auto lpPreHeader=lp->getLoopPreheader();
        auto lpHeaderTerminator=dyn_cast<ir::BranchInst>(lpHeader->terminator());
        if(lpHeaderTerminator==nullptr)continue;//header's terminator must be brcond
        auto lpCond=lpHeaderTerminator->cond();
        auto lpCondScid=lpCond->valueId();
        ir::PhiInst* keyPhiInst;
        ir::Value* mEndVar;
        if(lpCondScid>=ir::vICMP_BEGIN and lpCondScid<=ir::vICMP_END){
            auto lpCondIcmp=dyn_cast<ir::ICmpInst>(lpCond);
            if(lpCondIcmp->lhs()->valueId()==ir::vPHI){
                keyPhiInst=dyn_cast<ir::PhiInst>(lpCondIcmp->lhs());
                mEndVar=lpCondIcmp->rhs();
            }
            else if(lpCondIcmp->rhs()->valueId()==ir::vPHI){
                keyPhiInst=dyn_cast<ir::PhiInst>(lpCondIcmp->rhs());
                mEndVar=lpCondIcmp->lhs();
            }
        }
        else if(lpCondScid>=ir::vFCMP_BEGIN and lpCondScid<=ir::vFCMP_END){
            auto lpCondFcmp=dyn_cast<ir::FCmpInst>(lpCond);
            if(lpCondFcmp->lhs()->valueId()==ir::vPHI){
                keyPhiInst=dyn_cast<ir::PhiInst>(lpCondFcmp->lhs());
                mEndVar=lpCondFcmp->rhs();
            }
            else if(lpCondFcmp->rhs()->valueId()==ir::vPHI){
                keyPhiInst=dyn_cast<ir::PhiInst>(lpCondFcmp->rhs());
                mEndVar=lpCondFcmp->lhs();
            }
        }
        else
            continue;
        auto mBeginVar=dyn_cast<ir::Constant>(keyPhiInst->getvalfromBB(lpPreHeader));
        auto iterInst=keyPhiInst->getValue(0)==mBeginVar?keyPhiInst->getValue(1):keyPhiInst->getValue(0);
        auto iterInstScid=iterInst->valueId();
        ir::Constant* mstepVar;
        if(not(iterInstScid==ir::vADD or iterInstScid==ir::vFADD or 
            iterInstScid==ir::vSUB or iterInstScid==ir::vFSUB or 
            iterInstScid==ir::vMUL or iterInstScid==ir::vFMUL))continue;
        auto iterInstBinary=dyn_cast<ir::BinaryInst>(iterInst);
        if(iterInstBinary->lValue()->valueId()==ir::vPHI){
            if(dyn_cast<ir::PhiInst>(iterInstBinary->lValue())!=keyPhiInst)continue;
            mstepVar=dyn_cast<ir::Constant>(iterInstBinary->rValue());
        }
        else if(iterInstBinary->rValue()->valueId()==ir::vPHI){
            if(dyn_cast<ir::PhiInst>(iterInstBinary->rValue())!=keyPhiInst)continue;
            mstepVar=dyn_cast<ir::Constant>(iterInstBinary->lValue());
        }
        else 
            continue;
        addIndVar(lp,mBeginVar,mstepVar,mEndVar,iterInstBinary,dyn_cast<ir::Instruction>(lpCond));
    }
}

void indVarAnalysis::addIndVar(ir::Loop* lp, ir::Constant* mbegin, ir::Constant* mstep, ir::Value* mend, ir::BinaryInst* iterinst, ir::Instruction* cmpinst){
    auto pnewIdv=new ir::indVar(mbegin,mend,mstep,iterinst,cmpinst);
    ivctx->addIndVar(lp,pnewIdv);
}

void indVarInfoCheck::run(ir::Function* func, topAnalysisInfoManager* tp){
    if(func->isOnlyDeclare())return;
    lpctx=tp->getLoopInfo(func);
    lpctx->refresh();
    ivctx=tp->getIndVarInfo(func);
    using namespace std;
    for(auto lp:lpctx->loops()){
        cerr<<"In loop whose header is "<<lp->header()->name()<<":"<<endl;
        auto idv=ivctx->getIndvar(lp);
        if(idv==nullptr){
            cerr<<"No indvar."<<endl;
        }
        else{
            cerr<<"BeginVar:\t"<<idv->getBeginI32()<<endl;
            cerr<<"StepVar:\t"<<idv->getStepI32()<<endl;
        }
    }
}