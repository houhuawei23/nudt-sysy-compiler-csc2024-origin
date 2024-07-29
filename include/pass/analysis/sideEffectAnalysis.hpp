#pragma once
#include "pass/pass.hpp"
using namespace pass;

class sideEffectAnalysis:public ModulePass{
    public:
        void run(ir::Module* md,TopAnalysisInfoManager* tp)override;
        std::string name() const override;
    
    private:
        callGraph* cgctx;
        sideEffectInfo* sectx;
        void infoCheck(ir::Module* md);
        bool isGlobal(ir::GetElementPtrInst* gep);
};
