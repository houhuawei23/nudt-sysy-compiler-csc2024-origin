#pragma once
#include "pass/pass.hpp"
using namespace pass;

class sideEffectAnalysis:public ModulePass{
    public:
    void run(ir::Module* md,TopAnalysisInfoManager* tp)override;
    void infoCheck(ir::Module* md);
    private:
    callGraph* cgctx;
    sideEffectInfo* sectx;
};
