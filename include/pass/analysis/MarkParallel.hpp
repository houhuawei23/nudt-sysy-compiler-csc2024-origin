#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include <vector>
#include <set>

namespace pass{
    class markParallel:public FunctionPass{
        public:
            void run(ir::Function* func,TopAnalysisInfoManager* tp)override;
            std::string name() const override;
        private:
            dependenceInfo* dpctx;
            domTree* domctx;
            loopInfo* lpctx;
            sideEffectInfo* sectx;
            callGraph* cgctx;
            indVarInfo* idvctx;
            parallelInfo* parctx;
            void runOnLoop(ir::Loop* lp);
            void printParallelInfo(ir::Function* func);
    };
};