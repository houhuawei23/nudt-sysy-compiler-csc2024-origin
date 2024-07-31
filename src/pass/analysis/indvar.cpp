#include "pass/analysis/indvar.hpp"

using namespace pass;


void indVarAnalysis::run(ir::Function* func, TopAnalysisInfoManager* tp) {
    if (func->isOnlyDeclare()) return;
    lpctx = tp->getLoopInfo(func);
    // lpctx->setOff();
    lpctx->refresh();
    ivctx = tp->getIndVarInfo(func);
    ivctx->clearAll();
    func->rename();
    for (auto lp : lpctx->loops()) {
        auto lpHeader = lp->header();
        std::cerr<<lp->header()->name()<<std::endl;
        if(not lp->isLoopSimplifyForm())continue;
        auto lpPreHeader = lp->getLoopPreheader();
        auto lpHeaderTerminator = dyn_cast<ir::BranchInst>(lpHeader->terminator());
        if (lpHeaderTerminator == nullptr) continue;  // header's terminator must be brcond
        if (not lpHeaderTerminator->is_cond()) continue;
        if (lp->exits().size()>1) continue;
        auto lpCond = lpHeaderTerminator->cond();
        auto lpCondScid = lpCond->valueId();
        ir::PhiInst* keyPhiInst;
        ir::Value* mEndVar;
        if (lpCondScid >= ir::vICMP_BEGIN and lpCondScid <= ir::vICMP_END) {
            auto lpCondIcmp = dyn_cast<ir::ICmpInst>(lpCond);
            auto lpCondIcmpLHSPhi=lpCondIcmp->lhs()->dynCast<ir::PhiInst>();
            auto lpCondIcmpRHSPhi=lpCondIcmp->rhs()->dynCast<ir::PhiInst>();
            if(lpCondIcmpLHSPhi!=nullptr and lpCondIcmpLHSPhi->block()==lpHeader){
                keyPhiInst=lpCondIcmpLHSPhi;
                mEndVar=lpCondIcmp->rhs();
            }
            else if(lpCondIcmpRHSPhi!=nullptr and lpCondIcmpRHSPhi->block()==lpHeader){
                keyPhiInst=lpCondIcmpRHSPhi;
                mEndVar=lpCondIcmp->lhs();
            }
            else 
                continue;
        } 
        else
            continue;
        auto mBeginVar = dyn_cast<ir::Constant>(keyPhiInst->getvalfromBB(lpPreHeader));
        if(mBeginVar==nullptr){//考虑内层循环嵌套问题
            if(lpctx->looplevel(lpHeader)==1 or lpctx->looplevel(lpHeader)==0)continue;//如果这时本来就是最外层循环，那么就不适合分析indvar
            auto mBeginVarPhi=dyn_cast<ir::PhiInst>(keyPhiInst->getvalfromBB(lpPreHeader));
            if(mBeginVarPhi==nullptr)continue;
            mBeginVar=getConstantBeginVarFromPhi(mBeginVarPhi,lp->parent());
        }
        if(mBeginVar==nullptr)continue;
        auto iterInst=keyPhiInst->getValue(0)==keyPhiInst->getvalfromBB(lp->getlooppPredecessor())?
            keyPhiInst->getValue(1):keyPhiInst->getValue(0);
        auto iterInstScid = iterInst->valueId();
        ir::Constant* mstepVar;
        if (not(iterInstScid == ir::vADD or iterInstScid == ir::vFADD or iterInstScid == ir::vSUB or
                iterInstScid == ir::vFSUB or iterInstScid == ir::vMUL or iterInstScid == ir::vFMUL))
            continue;
        auto iterInstBinary = dyn_cast<ir::BinaryInst>(iterInst);
        if (iterInstBinary->lValue()->valueId() == ir::vPHI) {
            if (dyn_cast<ir::PhiInst>(iterInstBinary->lValue()) != keyPhiInst) continue;
            mstepVar = dyn_cast<ir::Constant>(iterInstBinary->rValue());
        } else if (iterInstBinary->rValue()->valueId() == ir::vPHI) {
            if (dyn_cast<ir::PhiInst>(iterInstBinary->rValue()) != keyPhiInst) continue;
            mstepVar = dyn_cast<ir::Constant>(iterInstBinary->lValue());
        } else
            continue;
        addIndVar(lp, mBeginVar, mstepVar, mEndVar, iterInstBinary,
                  dyn_cast<ir::Instruction>(lpCond),keyPhiInst);
    }
}

void indVarAnalysis::addIndVar(ir::Loop* lp,
                               ir::Constant* mbegin,
                               ir::Constant* mstep,
                               ir::Value* mend,
                               ir::BinaryInst* iterinst,
                               ir::Instruction* cmpinst,
                               ir::PhiInst* phiinst) {
    auto pnewIdv = new ir::indVar(mbegin, mend, mstep, iterinst, cmpinst,phiinst);
    ivctx->addIndVar(lp, pnewIdv);
}

void indVarInfoCheck::run(ir::Function* func, TopAnalysisInfoManager* tp) {
    if (func->isOnlyDeclare()) return;
    lpctx = tp->getLoopInfo(func);
    lpctx->refresh();
    ivctx = tp->getIndVarInfo(func);
    using namespace std;
    for (auto lp : lpctx->loops()) {
        cerr << "In loop whose header is " << lp->header()->name() << ":" << endl;
        auto idv = ivctx->getIndvar(lp);
        if (idv == nullptr) {
            cerr << "No indvar." << endl;
        } else {
            cerr << "BeginVar:\t" << idv->getBeginI32() << endl;
            cerr << "StepVar :\t" << idv->getStepI32() << endl;
            if(idv->isEndVarConst())
            cerr << "EndVar  :\t" << idv->getEndVarI32() << endl;
        }
    }
}

ir::Constant* indVarAnalysis::getConstantBeginVarFromPhi(ir::PhiInst* phiinst,ir::Loop* lp){
    if(not lp->isLoopSimplifyForm())return nullptr;
    if(phiinst->block()!=lp->header())return nullptr;
    auto constVal=phiinst->getvalfromBB(lp->getLoopPreheader())->dynCast<ir::Constant>();
    if(constVal!=nullptr)return constVal;
    auto phiVal=phiinst->getvalfromBB(lp->getLoopPreheader())->dynCast<ir::PhiInst>();
    if(phiVal==nullptr)return nullptr;
    auto outerLp=lp->parent();
    if(outerLp==nullptr)return nullptr;
    return getConstantBeginVarFromPhi(phiVal,outerLp);
}