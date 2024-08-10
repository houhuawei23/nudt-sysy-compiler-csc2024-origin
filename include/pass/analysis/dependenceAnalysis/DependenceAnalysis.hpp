#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass{

    class dependenceAnalysis;
    // class dependenceAnalysisInfoCheck;

    class dependenceAnalysis:public FunctionPass{
        public:
            void run(ir::Function* func,TopAnalysisInfoManager* tp)override;
            std::string name() const override { return "dependence analysis"; }
        private:
            domTree* domctx;
            loopInfo* lpctx;
            indVarInfo* idvctx;
            sideEffectInfo* sectx;
            dependenceInfo* dpctx;
            void runOnLoop(ir::Loop* lp);
    };
    
};