#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass{
    class SCEV;
    class SCEVValue;

    class SCEV:public FunctionPass{
        public:
            void run(ir::Function* func,TopAnalysisInfoManager* tp)override;
        private:
            loopInfo* lpctx;
            indVarInfo* idvctx;
            sideEffectInfo* sectx;
            void runOnLoop(ir::Loop* lp);

    };
    enum SCEVOperator{
        addi,
        subi,
        muli,
        addf,
        subf,
        mulf
    };
    class SCEVValue{
    private:    
        SCEVOperator scevOp;
        ir::Value* mBegin;
        ir::Value* mEnd;
        ir::Value* mStep;

    };
}