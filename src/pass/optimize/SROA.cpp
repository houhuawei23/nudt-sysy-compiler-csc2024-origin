#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
#include "pass/optimize/SROA.hpp"
using namespace pass;

std::string SROA::name() const {
    return "SROA";
}

void SROA::run(ir::Function* func,TopAnalysisInfoManager* tp){
    dpctx=tp->getDepInfo(func);
    domctx=tp->getDomTreeWithoutRefresh(func);
    idvctx=tp->getIndVarInfoWithoutRefresh(func);
    lpctx=tp->getLoopInfoWithoutRefresh(func);
    sectx=tp->getSideEffectInfoWithoutRefresh();
}