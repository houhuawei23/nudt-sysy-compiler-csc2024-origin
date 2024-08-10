#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/analysis/dependenceAnalysis/dpaUtils.hpp"
using namespace pass;

void dependenceAnalysis::run(ir::Function* func,TopAnalysisInfoManager* tp){
    domctx=tp->getDomTree(func);
    lpctx=tp->getLoopInfo(func);
    idvctx=tp->getIndVarInfo(func);
    sectx=tp->getSideEffectInfo();
    dpctx=tp->getDepInfoWithoutRefresh(func);

    for(auto lp:lpctx->loops()){
        if(lp->parent()==nullptr)
            runOnLoop(lp);
    }
}

void dependenceAnalysis::runOnLoop(ir::Loop* lp){
    for(auto subLp:lp->subLoops()){//先处理子循环
        runOnLoop(subLp);
    }
    auto depInfoForLp=std::any_cast<LoopDependenceInfo*>(dpctx->getLoopDependenceInfo(lp));
    if(depInfoForLp==nullptr){
        depInfoForLp=new LoopDependenceInfo();
        dpctx->setDepInfoLp(lp,depInfoForLp);
    }
    depInfoForLp->makeLoopDepInfo(lp);

}   