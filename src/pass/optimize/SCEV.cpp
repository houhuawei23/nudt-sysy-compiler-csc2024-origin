#include "pass/optimize/SCEV.hpp"
using namespace pass;

void SCEV::run(ir::Function* func,TopAnalysisInfoManager* tp){
    if(func->isOnlyDeclare())return;
    lpctx=tp->getLoopInfo(func);
    idvctx=tp->getIndVarInfo(func);
    sectx=tp->getSideEffectInfo();
    lpctx->refresh();
    idvctx->setOff();
    idvctx->refresh();
    sectx->setOff();
    sectx->refresh();
    for(auto lp:lpctx->loops()){
        runOnLoop(lp);
    }
}

void SCEV::runOnLoop(ir::Loop* lp){
    if(lp->exits().size()>1)return;//不处理多出口
    
}