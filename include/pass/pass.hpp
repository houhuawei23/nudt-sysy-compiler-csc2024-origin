#pragma once
#include <vector>
#include "ir/ir.hpp"
#include "pass/analysisinfo.hpp"


namespace pass {

//! Pass Template
template <typename PassUnit>
class Pass {
   public:
    // pure virtual function, define the api
    virtual void run(PassUnit* pass_unit,topAnalysisInfoManager* tp) = 0;
};

// Instantiate Pass Class for Module, Function and BB
using ModulePass = Pass<ir::Module>;
using FunctionPass = Pass<ir::Function>;
using BasicBlockPass = Pass<ir::BasicBlock>;


class PassManager{
    ir::Module* _irModule;
    pass::topAnalysisInfoManager* tAIM;
    public:
        PassManager(ir::Module* pm,topAnalysisInfoManager* tp){
            _irModule=pm;
            tAIM=tp;
        }
        void run(ModulePass* mp){
            mp->run(_irModule,tAIM);
        }
        void run(FunctionPass* fp){
            for(auto func:_irModule->funcs()){
                fp->run(func,tAIM);
            }
        }
         void run(BasicBlockPass* bp){
            for(auto func:_irModule->funcs()){
                for(auto bb:func->blocks()){
                    bp->run(bb,tAIM);
                }
            }
        }
};

class topAnalysisInfoManager{
    private:
        ir::Module* mModule;
        // ir::Module info
        callGraph* mCallGraph;
        // ir::Function info
        std::unordered_map<ir::Function*,domTree*>mDomTree;
        std::unordered_map<ir::Function*,pdomTree*>mPDomTree;
        std::unordered_map<ir::Function*,loopInfo*>mLoopInfo;
        // bb info
    public:
        topAnalysisInfoManager(ir::Module* pm):mModule(pm),mCallGraph(nullptr){}
        domTree* getDomTree(ir::Function* func){return mDomTree[func];}
        pdomTree* getPDomTree(ir::Function* func){return mPDomTree[func];}
        loopInfo* getLoopInfo(ir::Function* func){return mLoopInfo[func];}
        callGraph* getCallGraph(){return mCallGraph;}
        void initialize();
        void CFGChange(ir::Function* func){
            mDomTree[func]->setOff();
            mPDomTree[func]->setOff();
            mLoopInfo[func]->setOff();
        }
        void CallChange(){
            mCallGraph->setOff();
        }
};


}  // namespace pass