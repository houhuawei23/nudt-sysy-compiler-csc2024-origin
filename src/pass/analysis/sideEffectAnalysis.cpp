#include "pass/analysis/sideEffectAnalysis.hpp"
using namespace pass;

static std::set<ir::Function*>worklist;
static std::set<ir::Function*>hasSideEffectFunctions;

void sideEffectAnalysis::run(ir::Module* md,TopAnalysisInfoManager* tp){
    
    worklist.clear();
    hasSideEffectFunctions.clear();
    sectx=tp->getSideEffectInfo();
    cgctx=tp->getCallGraph();
    cgctx->refresh();
    sectx->clearAll();
    // assert(md->findFunction("sysy_starttime")->isOnlyDeclare());

    sectx->setFunc(md->findFunction("getint"),true);
    sectx->setFunc(md->findFunction("getch"),true);
    sectx->setFunc(md->findFunction("getfloat"),true);
    sectx->setFunc(md->findFunction("putint"),true);
    sectx->setFunc(md->findFunction("putch"),true);
    sectx->setFunc(md->findFunction("putfloat"),true);
    sectx->setFunc(md->findFunction("putarray"),true);
    sectx->setFunc(md->findFunction("putfarray"),true);
    sectx->setFunc(md->findFunction("putf"),true);
    sectx->setFunc(md->findFunction("starttime"),true);
    sectx->setFunc(md->findFunction("stoptime"),true);

    sectx->setFunc(md->findFunction("getarray"),true);
    sectx->setFunc(md->findFunction("getfarray"),true);
    hasSideEffectFunctions.insert(md->findFunction("getfarray"));
    hasSideEffectFunctions.insert(md->findFunction("getarray"));


    for(auto func:md->funcs()){
        if(func->isOnlyDeclare())continue;
        sectx->setFunc(func,false);

        
        for(auto bb:func->blocks()){
            for(auto inst:bb->insts()){
                if(inst->hasSideEffect()){
                    sectx->setFunc(func,true);
                    break;
                }
            }
            if(sectx->hasSideEffect(func))break;
        }
        if(sectx->hasSideEffect(func)){
            hasSideEffectFunctions.insert(func);
            worklist.insert(func);
        }
            
    }

    while(not worklist.empty()){
        auto func=*worklist.begin();
        worklist.erase(func);
        for(auto funcCaller:cgctx->callers(func)){
            if(not hasSideEffectFunctions.count(funcCaller)){
                hasSideEffectFunctions.insert(funcCaller);
                sectx->setFunc(funcCaller,true);
                worklist.insert(func);
            }
        }
    }

    sectx->setOn();
    sectx->setFunc(md->findFunction("sysy_starttime"),true);
    sectx->setFunc(md->findFunction("sysy_stopttime"),true);
    // infoCheck(md);
}

void sideEffectAnalysis::infoCheck(ir::Module* md){
    for(auto func:md->funcs()){
        using namespace std;
        // if(func->isOnlyDeclare())continue;
        cerr<<func->name()<<":"<<sectx->hasSideEffect(func)<<endl;
    }
}