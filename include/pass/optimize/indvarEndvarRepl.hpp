#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass{
    class idvEdvRepl:public FunctionPass{
        public:
            std::string name() const override { return "idvEdvRepl"; }
            void run(ir::Function* func,TopAnalysisInfoManager* tp)override;
        private:
            loopInfo* lpctx;
            indVarInfo* idvctx;
            domTree* domctx;
            sideEffectInfo* sectx;
            void runOnLoop(ir::Loop* lp);
            int getConstantEndvarIndVarIterCnt(ir::Loop* lp,ir::IndVar* idv);
            void normalizeIcmpAndBr(ir::Loop* lp,ir::IndVar* idv);
            void exchangeIcmpOp(ir::ICmpInst* icmpInst);
            void reverseIcmpOp(ir::ICmpInst* icmpInst);
            void exchangeBrDest(ir::BranchInst* brInst);
            bool isSimplyNotInLoop(ir::Loop* lp,ir::Value* val);
            bool isSimplyLoopInvariant(ir::Loop* lp,ir::Value* val);
            void replaceIndvarAfterLoop(ir::Loop* lp,ir::IndVar* idv,ir::Value* finalVar);
            ir::Value* addFinalVarInstInLatchSub1(ir::Value* edv,ir::Loop* lp);
            ir::Value* addFinalVarInstInLatchAdd1(ir::Value* edv,ir::Loop* lp);
    };
}