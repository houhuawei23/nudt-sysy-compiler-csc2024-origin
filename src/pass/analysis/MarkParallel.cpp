#include "pass/analysis/MarkParallel.hpp"
#include "pass/analysis/dependenceAnalysis/DependenceAnalysis.hpp"
using namespace pass;

std::string markParallel::name() const {
    return "mark parallel";
}

void markParallel::run(ir::Function* func,TopAnalysisInfoManager* tp){
    dpctx=tp->getDepInfo(func);
    domctx=tp->getDomTreeWithoutRefresh(func);
    lpctx=tp->getLoopInfoWithoutRefresh(func);
    cgctx=tp->getCallGraphWithoutRefresh();
    sectx=tp->getSideEffectInfoWithoutRefresh();
    idvctx=tp->getIndVarInfoWithoutRefresh(func);
    parctx=tp->getParallelInfo(func);
    for(auto loop:lpctx->loops()){
        runOnLoop(loop);
    }
    // printParallelInfo(func);
}

void markParallel::runOnLoop(ir::Loop* lp){
    auto lpDepInfo=dpctx->getLoopDependenceInfo(lp);
    bool isParallelConcerningArray=lpDepInfo->getIsParallel();
    if(isParallelConcerningArray==false){
        parctx->setIsParallel(lp->header(),false);
        return;
    }
    if(lp->header()->phi_insts().size()>1){
        parctx->setIsParallel(lp->header(),false);
        return;
    }
    for(auto bb:lp->blocks()){
        for(auto inst:bb->insts()){
            if(auto callInst=inst->dynCast<ir::CallInst>()){
                auto callee=callInst->callee();
                if(not sectx->isPureFunc(callee)){
                    parctx->setIsParallel(lp->header(),false);
                    return;
                }
            }
            else if(auto storeInst=inst->dynCast<ir::StoreInst>()){
                auto ptr=storeInst->ptr();
                if(ptr->dynCast<ir::GlobalVariable>()){
                    parctx->setIsParallel(lp->header(),false);
                    return;
                }
            }
        }
    }
    parctx->setIsParallel(lp->header(),true);
    return;
}

void markParallel::printParallelInfo(ir::Function* func){
    std::cerr<<"In Function \""<<func->name()<<"\":"<<std::endl;
    for(auto lp:lpctx->loops()){
        using namespace std;
        cerr<<"Parallize Loop whose header is \""<<lp->header()->name()<<"\" :";
        if(parctx->getIsParallel(lp->header())){
            cerr<<"YES";
        }
        else{
            cerr<<"NO";
        }
        cerr<<endl;
    }

}