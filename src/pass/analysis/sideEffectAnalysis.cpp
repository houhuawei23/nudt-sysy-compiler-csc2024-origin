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
    // std::cerr<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
    
    worklist.clear();
    vis.clear();
    hasSideEffectFunctions.clear();
    sectx=tp->getSideEffectInfoWithoutRefresh();
    cgctx=tp->getCallGraph();
    sectx->clearAll();

    for(auto func:md->funcs()){
        sectx->setFuncSideEffect(func,false);
        sectx->setFuncGVUse(func,false);
        if(cgctx->isLib(func)){//cond 4
            sectx->setFuncSideEffect(func,true);
            sectx->setFuncGVUse(func,false);
            hasSideEffectFunctions.insert(func);
            worklist.insert(func);
        }
        else{
            for(auto arg:func->args()){//cond 2
                if(arg->isPointer()){
                    sectx->setFuncSideEffect(func,true);
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
                        if(auto gv=stptr->dynCast<ir::GlobalVariable>()){
                            // std::cerr<<"In function \""<<func->name()<<"\", we got a store to gv\" "<<gv->name()<<"\""<<std::endl;
                            sectx->setFuncSideEffect(func,true);
                            hasSideEffectFunctions.insert(func);
                            worklist.insert(func);
                            break;
                        }
                        else if(auto gep=stptr->dynCast<ir::GetElementPtrInst>()){
                            if(isGlobal(gep)){
                                sectx->setFuncSideEffect(func,true);
                                hasSideEffectFunctions.insert(func);
                                worklist.insert(func);
                                break;
                            }
                        }
                    }
                    else if(auto loadInst=inst->dynCast<ir::LoadInst>()){
                        auto loadPtr=loadInst->ptr();
                        if(auto gv=loadPtr->dynCast<ir::GlobalVariable>()){
                            sectx->setFuncGVUse(func,true);
                        }
                        else if(auto gep=loadPtr->dynCast<ir::GetElementPtrInst>()){
                            if(isGlobal(gep)){
                                sectx->setFuncGVUse(func,true);
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
        // std::cerr<<func->name()<<std::endl;
        // infoCheck(md);
        for(auto callerFunc:cgctx->callers(func)){
            // std::cerr<<"call "<<callerFunc->name()<<std::endl;
            if(not hasSideEffectFunctions.count(callerFunc)){
                sectx->setFuncSideEffect(callerFunc,true);
                hasSideEffectFunctions.insert(callerFunc);
                worklist.insert(callerFunc);
                // std::cerr<<"Side Effect "<<callerFunc->name()<<std::endl;
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