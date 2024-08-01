#include "pass/optimize/SCEV.hpp"
using namespace pass;

void SCEV::run(ir::Function* func,TopAnalysisInfoManager* tp){
    if(func->isOnlyDeclare())return;
    lpctx=tp->getLoopInfo(func);
    idvctx=tp->getIndVarInfo(func);
    sectx=tp->getSideEffectInfo();
    domctx=tp->getDomTree(func);
    lpctx->refresh();
    idvctx->setOff();
    idvctx->refresh();
    sectx->setOff();
    sectx->refresh();
    domctx->refresh();
    for(auto lp:lpctx->loops()){
        if(lp->parent()!=nullptr)continue;//只处理顶层循环，底层循环通过顶层循环向下分析
        runOnLoop(lp);
    }
}

void SCEV::runOnLoop(ir::Loop* lp){
    if(lp->exits().size()>1)return;//不处理多出口
    auto defaultIdv=idvctx->getIndvar(lp);
    // if(defaultIdv==nullptr)return;//必须有基础indvar
    if(not lp->isLoopSimplifyForm())return;
    for(auto subLp:lp->subLoops()){
        runOnLoop(subLp);
    }
    if(defaultIdv==nullptr)return;
}

bool SCEV::isSimplyLoopInvariant(ir::Loop* lp,ir::Value* val){
    if(auto inst=val->dynCast<ir::Instruction>()){
        auto instBB=inst->block();
        if(domctx->dominate(instBB,lp->header()))return true;
    }
    if(auto conVal=val->dynCast<ir::Constant>())return true;
    return false;
}