#include "pass/optimize/aggressiveG2L.hpp"

using namespace pass;

/*
激进的global2local（仅仅对标量进行替换）
对函数内部的global变量进行local变化，以试图消去冗余的loadstore，将全局变量以尽可能长的时间留在虚拟寄存器中
把全局变量分为三种：
1 只读的和常数的--直接替换
2 只写的--删掉
3 又读又写的
其中第三种再细分，又分为：
a 只是被main函数使用的
b 被一个其他的函数使用的
c 被多个函数使用的

在结束之后内部调用ADCE
*/

void aggressiveG2L::run(ir::Module* md,TopAnalysisInfoManager* tp){
    //1. 分析读写情况;
    std::unordered_map<ir::GlobalVariable*,std::set<ir::Function*>>funcGvRead;
    std::unordered_map<ir::GlobalVariable*,std::set<ir::Function*>>funcGvWrite;
    std::set<ir::GlobalVariable*>readAndWriteGvs;
    std::set<ir::GlobalVariable*>readOnlyGvs;
    std::set<ir::GlobalVariable*>writeOnlyGvs;
    for(auto gv:md->globalVars()){
        if(gv->isArray())continue;
        funcGvRead[gv]=std::set<ir::Function*>();
        funcGvWrite[gv]=std::set<ir::Function*>();
    }
    sectx=tp->getSideEffectInfo();
    for(auto func:md->funcs()){
        for(auto readGv:sectx->funcReadGlobals(func)){
            if(readGv->isArray())continue;
            funcGvRead[readGv].insert(func);
        }
        for(auto writeGv:sectx->funcWriteGlobals(func)){
            if(writeGv->isArray())continue;
            funcGvWrite[writeGv].insert(func);
        }
    }
    for(auto gv:md->globalVars()){
        if(gv->isArray())continue;
        if(funcGvRead[gv].size() and not funcGvWrite[gv].size())
            readOnlyGvs.insert(gv);
        else if(funcGvWrite[gv].size() and not funcGvRead[gv].size())
            writeOnlyGvs.insert(gv);
        else
            readAndWriteGvs.insert(gv);
    }
    for(auto ROGv:readOnlyGvs)replaceReadOnlyGv(ROGv);
    for(auto WOGv:writeOnlyGvs)deleteWriteOnlyGv(WOGv);
    //处理cond 3
    std::set<ir::GlobalVariable*>multipleRWGvs;
    for(auto rwGv:readAndWriteGvs){
        auto& readfnset=funcGvRead[rwGv];
        auto& writefnset=funcGvWrite[rwGv];
        //cond a and cond b
        if(readfnset.size()==1 and writefnset.size()==1){
            
        }
        else{//cond c

        }


    }
}

void aggressiveG2L::replaceReadOnlyGv(ir::GlobalVariable* gv){
    auto gvInitVal=gv->init(0);
    for(auto puse:gv->uses()){
        auto puser=puse->user();
        if(auto ldinst=puser->dynCast<ir::LoadInst>()){
            ldinst->replaceAllUseWith(gvInitVal);
        }
        if(auto stInst=puser->dynCast<ir::StoreInst>()){
            assert(false and "Trying to replace a gv with value while it's not Read only!");
        }
    }
}

void aggressiveG2L::deleteWriteOnlyGv(ir::GlobalVariable* gv){
    for(auto puse:gv->uses()){
        auto puser=puse->user();
        if(auto stInst=puser->dynCast<ir::StoreInst>()){
            stInst->block()->delete_inst(stInst);
        }
        if(auto ldinst=puser->dynCast<ir::LoadInst>()){
            assert(false and "Trying to delete a non-WirteOnly gv!");
        }
    }
}