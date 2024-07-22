#include "pass/optimize/GlobalToLocal.hpp"

using namespace pass;
static std::unordered_map<ir::GlobalVariable*,std::set<ir::Function*>>globalDirectUsedFunc;//对该全局变量直接使用的函数
static std::unordered_map<ir::Function*,std::set<ir::GlobalVariable*>>funcDirectUseGlobal;//该函数直接使用的全局变量
static std::unordered_map<ir::Function*,std::set<ir::GlobalVariable*>>funcIndirectUseGlobal;//该函数间接使用的全局变量
static std::unordered_map<ir::GlobalVariable*,bool>globalHasStore;

void global2local::run(ir::Module* md,topAnalysisInfoManager* tp){
    cgctx=tp->getCallGraph();
    cgctx->refresh();
    globalCallAnalysis(md);
    for(auto gv:md->globalVars()){
        processGlobalVariables(gv,md);
    }
}

void global2local::globalCallAnalysis(ir::Module* md){
    /*
    这里进行分析, 判断每一个函数是否调用了对应的全局变量
    如果func A B C存在这样的调用链：A call B call C,
    而C调用了global var a,那么A和B也要视为调用了这个global var
    需要得到的信息：
    1. 每一个全局变量分别被使用了几次
    2. 每一个函数分别使用了（或者间接使用）哪些全局变量
    */
    //清理对应信息
    globalDirectUsedFunc.clear();
    funcDirectUseGlobal.clear();
    funcIndirectUseGlobal.clear();
    globalHasStore.clear();
    for(auto func:md->funcs()){
        funcDirectUseGlobal[func]=std::set<ir::GlobalVariable*>();
        funcIndirectUseGlobal[func]=std::set<ir::GlobalVariable*>();
    }
    for(auto gv:md->globalVars()){
        globalHasStore[gv]=false;
        globalDirectUsedFunc[gv]=std::set<ir::Function*>();
        for(auto puse:gv->uses()){
            auto gvUser=puse->user();
            auto gvUserInst=dyn_cast<ir::Instruction>(gvUser);
            if(gvUserInst){
                auto directUseFunc=gvUserInst->block()->function();
                globalDirectUsedFunc[gv].insert(directUseFunc);
                funcDirectUseGlobal[directUseFunc].insert(gv);
            }
            auto gvUserStoreInst=dyn_cast<ir::StoreInst>(gvUser);
            if(gvUserStoreInst!=nullptr)
                globalHasStore[gv]=true;

        }
    }


}

//对于间接调用结果，需要基于直接调用进行传播
void global2local::addIndirectGlobalUseFunc(ir::GlobalVariable* gv, ir::Function* func){
    funcIndirectUseGlobal[func].insert(gv);
    for(auto callerfunc:cgctx->callers(func)){
        if(funcIndirectUseGlobal[func].count(gv)==0)
            addIndirectGlobalUseFunc(gv,callerfunc);
    }
}

/*
对于所有的全局变量，分为以下三种情况：
1. 没有被使用过的global
2. 只在一个函数中被使用的global
3. 在多个函数中被使用的global
*/
void global2local::processGlobalVariables(ir::GlobalVariable* gv,ir::Module* md){
    auto gvUseFuncSize=globalDirectUsedFunc[gv].size();
    if(gv->isArray())return;
    if(not globalHasStore[gv]){
        //如果一个gv没有store,那么所有的值都可以被初始值直接替换！
        for(auto puseIter=gv->uses().begin();puseIter!=gv->uses().end();){
            auto puse=*puseIter;
            puseIter++;
            auto userLdInst=dyn_cast<ir::LoadInst>(puse->user());
            assert(userLdInst!=nullptr);//这里假设所有的对全局的使用都是load
            userLdInst->replaceAllUseWith(gv->init(0));
            userLdInst->block()->delete_inst(userLdInst);
        }
        return;
    }
    //如果对应的gv没有被使用过一次，那么就直接删除了
    if(gvUseFuncSize==0){
        md->delGlobalVariable(gv);
    }
    
}