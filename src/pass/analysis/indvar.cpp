#include "pass/analysis/indvar.hpp"

using namespace pass;

void indVarAnalysis::run(ir::Function* func,topAnalysisInfoManager* tp) {
    if(func->isOnlyDeclare())return;
    lpctx=tp->getLoopInfo(func);
    lpctx->refresh();
    ivctx=tp->getIndVarInfo(func);
    ivctx->clearAll();
    ivctx->initialize();
    for(auto lp:lpctx->loops()){
        auto lpHeader=lp->header();
        auto lpPreHeader=lp->getLoopPreheader();
        assert(lpPreHeader!=nullptr);
        for(auto pinst:lpHeader->phi_insts()){
            auto phiinst=dyn_cast<ir::PhiInst>(pinst);
            // 这里假设phi的incoming只有两个(经过了simplify loop)
            assert(phiinst->getsize()==2 and "Run simplifyloop pass before indvar!");
            auto mbegin=phiinst->getvalfromBB(lpPreHeader);//作为初始值
            auto constMBegin=dyn_cast<ir::Constant>(mbegin);
            if(constMBegin==nullptr)continue;//从preheader到达的定值不是常数则显然不可能是indvar
            auto anotherIncoming=phiinst->getValue(0)==mbegin?phiinst->getValue(1):phiinst->getValue(0);
            if(anotherIncoming->valueId()==ir::vADD){
                auto anotherIncomingAddInst=dyn_cast<ir::BinaryInst>(anotherIncoming);
                if(anotherIncomingAddInst->lValue()->valueId()==ir::vCONSTANT){
                    auto addInstRValue=dyn_cast<ir::PhiInst>(anotherIncomingAddInst->rValue());
                    
                }
                
            }
        }
    }
}

void addIndVar(ir::Loop* lp, ir::Constant* mbegin, ir::Constant* mstep, ir::Value* mend){

}