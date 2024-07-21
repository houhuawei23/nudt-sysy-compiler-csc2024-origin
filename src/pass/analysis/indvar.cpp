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
        
    }
}

void addIndVar(ir::Loop* lp, ir::Constant* mbegin, ir::Constant* mstep, ir::Value* mend){

}