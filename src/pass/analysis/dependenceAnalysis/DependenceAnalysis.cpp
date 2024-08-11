#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/analysis/dependenceAnalysis/dpaUtils.hpp"
using namespace pass;

void dependenceAnalysis::run(ir::Function* func,TopAnalysisInfoManager* tp){
    domctx=tp->getDomTree(func);
    lpctx=tp->getLoopInfo(func);
    idvctx=tp->getIndVarInfo(func);
    sectx=tp->getSideEffectInfo();
    cgctx=tp->getCallGraph();
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
    auto func=lp->header()->function();
    //分析所有的inst
    depInfoForLp->makeLoopDepInfo(lp);
    //别名分析测试
    bool isSame=false;
    for(auto setIter=depInfoForLp->getBaseAddrs().begin();setIter!=depInfoForLp->getBaseAddrs().end();setIter++){
        for(auto setIter2=depInfoForLp->getBaseAddrs().begin();setIter2!=setIter;setIter++){
            if(isTwoBaseAddrPossiblySame(*setIter,*setIter2,func,cgctx)){
                isSame=true;
                break;
            }
        }
    }
    if(isSame){
        depInfoForLp->setIsBaseAddrPossiblySame(isSame);
        assert(false and "No alias is allowed in dependenceAnalysis::runOnLoop()");
    }
    //为并行设计的依赖关系分析
    depInfoForLp->print(std::cerr);
}   