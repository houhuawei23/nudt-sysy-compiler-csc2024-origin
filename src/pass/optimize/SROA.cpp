#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/optimize/SROA.hpp"
using namespace pass;

static std::set<ir::GetElementPtrInst*>processedSubAddrs;

std::string SROA::name() const {
    return "SROA";
}

//SROA基本逻辑：
/*
处理的是对于循环内的变量
对于所有的循环内内存存取
1. 只考虑数组，全局由AG2L来考虑
2. 对于循环不变量，直接创建临时变量进行修改(从外到内，尽可能在外层做，将load-store紧贴在对应gep之后)
3. 对于当前循环内的量，只要没有possiblySame，都可以进行替换，从外到内即可
*/

void SROA::run(ir::Function* func,TopAnalysisInfoManager* tp){
    dpctx=tp->getDepInfo(func);
    domctx=tp->getDomTreeWithoutRefresh(func);
    idvctx=tp->getIndVarInfoWithoutRefresh(func);
    lpctx=tp->getLoopInfoWithoutRefresh(func);
    sectx=tp->getSideEffectInfoWithoutRefresh();
    processedSubAddrs.clear();
    for(auto lp:lpctx->loops()){
        if(lp->parentloop()!=nullptr)continue;
        runOnLoop(lp);
    }
}

void SROA::runOnLoop(ir::Loop* lp){
    depLpInfo=dpctx->getLoopDependenceInfo(lp);
    
    //从外到内
    for(auto subLoop:lp->subLoops()){
        runOnLoop(subLoop);
    }
}