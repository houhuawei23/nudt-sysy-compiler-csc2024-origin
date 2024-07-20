#pragma once
#include "pass/pass.hpp"

namespace pass{
    class indVarAnalysis:public FunctionPass{
        public:
            void run(ir::Function* func,topAnalysisInfoManager *tp)override;
        private:
            loopInfo* lpctx;
    };

    class indVarInfoCheck:public FunctionPass{
        public:   
            void run(ir::Function* func,topAnalysisInfoManager* tp)override;
        private:
            loopInfo* lpctx;
    };
}