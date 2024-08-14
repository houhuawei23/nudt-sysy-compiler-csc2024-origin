#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass{
    class SROA:public FunctionPass{
        public:
            void run(ir::Function* func,TopAnalysisInfoManager* tp)override;
            std::string name() const override;
        private:
            domTree* domctx;
            loopInfo* lpctx;
            sideEffectInfo* sectx;
            dependenceInfo* dpctx;
            indVarInfo* idvctx;
            void runOnLoop(ir::Loop* lp);
    };
}