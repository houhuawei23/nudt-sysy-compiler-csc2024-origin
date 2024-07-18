#pragma once
#include "pass/pass.hpp"

namespace pass{
    class loopAnalysis:public FunctionPass{
        public:
            void run(ir::Function* func,topAnalysisInfoManager *tp)override;
        private:
            void addLoopBlocks(ir::Function*func,ir::BasicBlock* header,ir::BasicBlock* tail);
            void loopGetExits(ir::Loop* plp);
            loopInfo* lpctx;
            domTree* domctx;
    };

    class loopInfoCheck:public FunctionPass{
        public:   
            void run(ir::Function* func,topAnalysisInfoManager* tp)override;
        private:
            loopInfo* lpctx;
    };
}
