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

        } else if (lpCondScid >= ir::vFCMP_BEGIN and lpCondScid <= ir::vFCMP_END) {
            // auto lpCondFcmp = dyn_cast<ir::FCmpInst>(lpCond);
            // if (lpCondFcmp->lhs()->valueId() == ir::vPHI) {
            //     keyPhiInst = dyn_cast<ir::PhiInst>(lpCondFcmp->lhs());
            //     mEndVar = lpCondFcmp->rhs();
            // } else if (lpCondFcmp->rhs()->valueId() == ir::vPHI) {
            //     keyPhiInst = dyn_cast<ir::PhiInst>(lpCondFcmp->rhs());
            //     mEndVar = lpCondFcmp->lhs();
            // }
            // else
            continue;//暂不支持浮点indvar
        } else
            continue;
        auto mBeginVar = dyn_cast<ir::Constant>(keyPhiInst->getvalfromBB(lpPreHeader));
        if(mBeginVar==nullptr){//考虑内层循环嵌套问题
            if(lpctx->looplevel(lpHeader)==0)continue;//如果这时本来就是最外层循环，那么就不适合分析indvar
            auto curLoop=lp;
            auto outerLoop=lp->parent();
            auto mBeginVarValue=keyPhiInst->getvalfromBB(lpHeader);
            bool canFound=true;
            while(outerLoop!=nullptr){
                if(not outerLoop->isLoopSimplifyForm()){
                    canFound=false;
                    break;
                }
                auto phiinst=mBeginVarValue->dynCast<ir::PhiInst>();
                if(phiinst==nullptr or phiinst->block()!=outerLoop->header()){
                    canFound=false;
                    break;
                }
                
            }
            if(not canFound)continue;
        }
        auto iterInst =
          keyPhiInst->getValue(0) == mBeginVar ? keyPhiInst->getValue(1) : keyPhiInst->getValue(0);
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