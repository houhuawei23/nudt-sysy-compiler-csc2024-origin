#include "pass/analysis/sideEffectAnalysis.hpp"
using namespace pass;

static std::set<ir::Function*>worklist;
static std::set<ir::Function*>hasSideEffectFunctions;
static std::set<ir::Function*>vis;


void sideEffectAnalysis::run(ir::Module* md,TopAnalysisInfoManager* tp){
    /*
    这样的函数有副作用：
    1. 对全局量做存
    2. 传入指针
    3. 调用有副作用的函数
    4. 库函数，因为优化掉没有用的库函数会导致io出现错误
    */
    
    worklist.clear();
    vis.clear();
    hasSideEffectFunctions.clear();
    sectx=tp->getSideEffectInfo();
    cgctx=tp->getCallGraph();
    cgctx->refresh();
    sectx->clearAll();

    for(auto func:md->funcs()){
        sectx->setFunc(func,false);
        if(cgctx->isLib(func)){//cond 4
            sectx->setFunc(func,true);
            hasSideEffectFunctions.insert(func);
        }
        else{
            for(auto arg:func->args()){//cond 2
                if(arg->isPointer()){
                    sectx->setFunc(func,true);
                    hasSideEffectFunctions.insert(func);
                    worklist.insert(func);
                    break;
                }
            }
            if(sectx->hasSideEffect(func))continue;
            for(auto bb:func->blocks()){
                for(auto inst:bb->insts()){
                    if(auto storeInst=inst->dynCast<ir::StoreInst>()){//cond 1
                        auto stptr=storeInst->ptr();
                        if(auto gv=md->findGlobalVariable(stptr->name())){
                            sectx->setFunc(func,true);
                            hasSideEffectFunctions.insert(func);
                            worklist.insert(func);
                            break;
                        }
                        else if(auto gep=stptr->dynCast<ir::GetElementPtrInst>()){
                            if(isGlobal(gep)){
                                sectx->setFunc(func,true);
                                hasSideEffectFunctions.insert(func);
                                worklist.insert(func);
                                break;
                            }
                        }
                    }
                }
                if(sectx->hasSideEffect(func))break;
            }
        }
    }
    //cond 3
    while(not worklist.empty()){
        
        auto func=*worklist.begin();
        worklist.erase(func);
        if(not vis.count(func))
            vis.insert(func);
        else   
            continue;
        for(auto callerFunc:cgctx->callers(func)){
            if(not hasSideEffectFunctions.count(callerFunc)){
                sectx->setFunc(func,true);
                hasSideEffectFunctions.insert(func);
                worklist.insert(func);
            }
        }
    }
    sectx->setOn();
    // infoCheck(md);
}

bool sideEffectAnalysis::isGlobal(ir::GetElementPtrInst* gep){
    if(auto gvbasePtr=gep->value()->dynCast<ir::GlobalVariable>())return true;
    if(auto gepbasePtr=gep->value()->dynCast<ir::GetElementPtrInst>())return isGlobal(gepbasePtr);
    return false;
}

void sideEffectAnalysis::infoCheck(ir::Module* md){
    for(auto func:md->funcs()){
        using namespace std;
        // if(func->isOnlyDeclare())continue;
        cerr<<func->name()<<":"<<sectx->hasSideEffect(func)<<endl;
    }
}