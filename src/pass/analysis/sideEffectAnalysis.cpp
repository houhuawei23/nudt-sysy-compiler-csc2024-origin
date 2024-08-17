#include "pass/analysis/sideEffectAnalysis.hpp"
using namespace pass;

static std::set<ir::Function*>worklist;



void sideEffectAnalysis::run(ir::Module* md,TopAnalysisInfoManager* tp){
    /*
    这样的函数有副作用：
    1. 对全局量做存
    2. 传入指针
    3. 调用有副作用的函数
    4. 库函数，因为优化掉没有用的库函数会导致io出现错误
    */
    // std::cerr<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
    topmana=tp;
    worklist.clear();
    sectx=tp->getSideEffectInfoWithoutRefresh();
    cgctx=tp->getCallGraph();
    sectx->clearAll();
    
    for(auto func:md->funcs()){
        sectx->functionInit(func);
        //isLib or not
        if(cgctx->isLib(func))
            sectx->setFuncIsLIb(func,true);
        else sectx->setFuncIsLIb(func,false);
        //use Gv and arg
        for(auto bb:func->blocks()){
            for(auto inst:bb->insts()){
                ir::Value* ptr;
                if(auto ldInst=inst->dynCast<ir::LoadInst>()){
                    ptr=getBaseAddr(ldInst->ptr());
                    if(auto allocainst=ptr->dynCast<ir::AllocaInst>())continue;
                    if(auto arg=ptr->dynCast<ir::Argument>())sectx->setArgRead(arg,true);
                    if(auto gv=ptr->dynCast<ir::GlobalVariable>())sectx->funcReadGlobals(func).insert(gv);
                }
                else if(auto stInst=inst->dynCast<ir::StoreInst>()){
                    ptr=getBaseAddr(stInst->ptr());
                    if(auto allocainst=ptr->dynCast<ir::AllocaInst>())continue;
                    if(auto arg=ptr->dynCast<ir::Argument>())sectx->setArgWrite(arg,true);
                    if(auto gv=ptr->dynCast<ir::GlobalVariable>())sectx->funcWriteGlobals(func).insert(gv);
                }
                else if(auto callInst=inst->dynCast<ir::CallInst>()){
                    auto calleeFunc=callInst->callee();
                    if(not cgctx->isLib(calleeFunc))continue;
                    if(calleeFunc->name()=="putarray"){
                        auto rarg=callInst->rargs()[1];
                        ptr=getBaseAddr(rarg->value());
                        if(auto allocainst=ptr->dynCast<ir::AllocaInst>())continue;
                        if(auto arg=ptr->dynCast<ir::Argument>())sectx->setArgRead(arg,true);
                        if(auto gv=ptr->dynCast<ir::GlobalVariable>())sectx->funcReadGlobals(func).insert(gv);
                    }
                    if(calleeFunc->name()=="putfarray"){
                        auto rarg=callInst->rargs()[1];
                        ptr=getBaseAddr(rarg->value());
                        if(auto allocainst=ptr->dynCast<ir::AllocaInst>())continue;
                        if(auto arg=ptr->dynCast<ir::Argument>())sectx->setArgRead(arg,true);
                        if(auto gv=ptr->dynCast<ir::GlobalVariable>())sectx->funcReadGlobals(func).insert(gv);
                    }
                    if(calleeFunc->name()=="getarray"){
                        auto rarg=callInst->rargs()[0];
                        ptr=getBaseAddr(rarg->value());
                        if(auto allocainst=ptr->dynCast<ir::AllocaInst>())continue;
                        if(auto arg=ptr->dynCast<ir::Argument>())sectx->setArgWrite(arg,true);
                        if(auto gv=ptr->dynCast<ir::GlobalVariable>())sectx->funcWriteGlobals(func).insert(gv);
                    }
                    if(calleeFunc->name()=="getfarray"){
                        auto rarg=callInst->rargs()[0];
                        ptr=getBaseAddr(rarg->value());
                        if(auto allocainst=ptr->dynCast<ir::AllocaInst>())continue;
                        if(auto arg=ptr->dynCast<ir::Argument>())sectx->setArgWrite(arg,true);
                        if(auto gv=ptr->dynCast<ir::GlobalVariable>())sectx->funcWriteGlobals(func).insert(gv);
                    }
                }
                
            }
        }
    }

    //propagate based on callgraph
    bool isChange=false;
    do{
        isChange=propogateSideEffect(md);
    }
    while(isChange);
    
    sectx->setOn();
    infoCheck(md);
}


ir::Value* sideEffectAnalysis::getBaseAddr(ir::Value* subAddr){
    if(auto allocainst=subAddr->dynCast<ir::AllocaInst>())return allocainst;
    if(auto gv=subAddr->dynCast<ir::GlobalVariable>())return gv;
    if(auto arg=subAddr->dynCast<ir::Argument>())return arg;
    if(auto gep=subAddr->dynCast<ir::GetElementPtrInst>())return getBaseAddr(gep->value());
    if(auto phi=subAddr->dynCast<ir::PhiInst>()){
        auto func=phi->block()->function();
        auto lpctx=topmana->getLoopInfo(func);
        auto lp=lpctx->head2loop(phi->block());
        auto preHeaderVal=phi->getvalfromBB(lp->getLoopPreheader());
        return getBaseAddr(preHeaderVal);
    }
    assert("Error! invalid type of input in function \"getBaseAddr\"!"&&false);
    return nullptr;
}

bool sideEffectAnalysis::propogateSideEffect(ir::Module* md){
    bool isChange=false;
    for(auto func:md->funcs()){
        for(auto calleeFunc:cgctx->callees(func)){
            //对于所有当前函数调用的函数
            if(calleeFunc==func)continue;//全局使用，其递归无影响
            //将被调用函数使用的全局变量添加到调用函数使用的全局变量中去
            for(auto gv:sectx->funcReadGlobals(calleeFunc)){
                auto resPair=sectx->funcReadGlobals(func).insert(gv);
                isChange=isChange or resPair.second;
            }
            for(auto gv:sectx->funcWriteGlobals(calleeFunc)){
                auto resPair=sectx->funcWriteGlobals(func).insert(gv);
                isChange=isChange or resPair.second;
            }
            if(cgctx->isLib(calleeFunc) and not sectx->getIsCallLib(func)){
                sectx->setFuncIsCallLib(func,true);
                isChange=true;
            }
            if(sectx->getIsCallLib(calleeFunc) and not sectx->getIsCallLib(func)){
                sectx->setFuncIsCallLib(func,true);
                isChange=true;
            }
        }
        //calleeInst,探讨具体对于对应gv引起的修改
        for(auto calleeInst:cgctx->calleeCallInsts(func)){
            auto calleeFunc=calleeInst->callee();
            for(auto pointerArg:sectx->funcArgSet(calleeFunc)){
                auto pointerArgIdx=pointerArg->index();
                auto pointerRArg=calleeInst->rargs()[pointerArgIdx];
                auto pointerRArgBaseAddr=getBaseAddr(pointerRArg->value());
                if(pointerRArgBaseAddr->dynCast<ir::AllocaInst>())continue;
                if(auto gv=pointerRArgBaseAddr->dynCast<ir::GlobalVariable>()){
                    if(sectx->getArgRead(pointerArg)){
                        auto resPair=sectx->funcReadGlobals(func).insert(gv);
                        isChange=isChange or resPair.second;
                    }
                    if(sectx->getArgWrite(pointerArg)){
                        auto resPair=sectx->funcWriteGlobals(func).insert(gv);
                        isChange=isChange or resPair.second;
                    }
                }
                if(auto arg=pointerRArgBaseAddr->dynCast<ir::Argument>()){
                    if(sectx->getArgRead(pointerArg)){
                        if(not sectx->getArgRead(arg))isChange=true;
                        sectx->setArgRead(arg,true);
                    }
                    if(sectx->getArgWrite(pointerArg)){
                        if(not sectx->getArgWrite(arg))isChange=true;
                        sectx->setArgWrite(arg,true);
                    }
                }
            }
        }
    }
    return isChange;
}

void sideEffectAnalysis::infoCheck(ir::Module* md){
    for(auto func:md->funcs()){
        using namespace std;
        if(func->isOnlyDeclare())continue;
        cerr<<"In Function \""<<func->name()<<"\":"<<endl;
        cerr<<"Read Global Variables:"<<endl;
        for(auto gv:sectx->funcReadGlobals(func)){
            cerr<<gv->name()<<"\t";
        }
        cerr<<endl;
        cerr<<"Write Global Variables:"<<endl;
        for(auto gv:sectx->funcWriteGlobals(func)){
            cerr<<gv->name()<<"\t";
        }
        cerr<<endl;
        cerr<<"Function Pointer Args:"<<endl;
        for(auto arg:sectx->funcArgSet(func)){
            cerr<<"Arg "<<arg->name()<<": ";
            if(sectx->getArgRead(arg))
                cerr<<"\tRead";
            if(sectx->getArgWrite(arg))
                cerr<<"\tWrite";
            cerr<<endl;
        }
        cerr<<endl;
    }
    for(auto func:md->funcs()){
        using namespace std;
        cerr<<"Function \""<<func->name()<<"\" side effect: ";
        if(sectx->hasSideEffect(func))cerr<<"YES";
        else cerr<<"NO";
        cerr<<"\tpure func: ";
        if(sectx->isPureFunc(func))cerr<<"YES";
        else cerr<<"NO";
        cerr<<endl;
    }
}