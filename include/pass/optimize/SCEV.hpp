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
        std::string name() const override{return "scev";}
    private:
        loopInfo* lpctx;
        indVarInfo* idvctx;
        sideEffectInfo* sectx;
        domTree* domctx;
        void runOnLoop(ir::Loop* lp);
        bool isSimplyLoopInvariant(ir::Loop* lp,ir::Value* val);
        bool isUsedOutsideLoop(ir::Loop* lp,ir::Value* val);
        int getConstantEndvarIndVarIterCnt(ir::Loop* lp,ir::indVar* idv);
        void addCalcIterCntInstructions(ir::Loop* lp,ir::indVar* idv);
        void normalizeIcmpAndBr(ir::Loop* lp,ir::indVar* idv);
        void exchangeIcmpOp(ir::ICmpInst* icmpInst);
        void reverseIcmpOp(ir::ICmpInst* icmpInst);
        void exchangeBrDest(ir::BranchInst* brInst);

};
}