#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/optimize/SROA.hpp"
using namespace pass;

static std::set<ir::GetElementPtrInst*>processedSubAddrs;

std::string SROA::name() const {
    return "SROA";
}

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
    if(not depLpInfo->getIsBaseAddrPossiblySame()){
        for(auto baseaddr:depLpInfo->getBaseAddrs()){
            if(not depLpInfo->getIsBaseAddrWrite(baseaddr)){//如果当前地址纯读取就直接替换
                for(auto subaddr:depLpInfo->baseAddrToSubAddrSet(baseaddr)){
                    if(processedSubAddrs.count(subaddr)!=0)continue;//处理过的不再处理
                    processedSubAddrs.insert(subaddr);
                    

                    
                }
            }
        }
    }
    for(auto subLoop:lp->subLoops()){
        runOnLoop(subLoop);
    }
}

void SROA::replaceOnlyLoadSubAddr(ir::GetElementPtrInst* gep,ir::Loop* lp){
    auto gepidx=depLpInfo->getGepIdx(gep);
    //经过LICM，如果循环没有副作用函数，就不可能出现idx全为LpI的情况而没有在外层被处理的
    
}