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
struct SCEVValue{
    ir::Value* initVal;
    std::vector<ir::Value*>addsteps;
    std::vector<ir::Value*>substeps;
    ir::PhiInst* phiinst;
    bool isFloat = false;
};



class SCEV : public FunctionPass{
    public:
        void run(ir::Function* func,TopAnalysisInfoManager* tp)override;
        std::string name() const override{return "scev";}
    private:
        LoopInfo* lpctx;
        IndVarInfo* idvctx;
        sideEffectInfo* sectx;
        DomTree* domctx;
        void runOnLoop(ir::Loop* lp,TopAnalysisInfoManager* tp);
        bool isSimplyLoopInvariant(ir::Loop* lp,ir::Value* val);
        bool isSimplyNotInLoop(ir::Loop* lp,ir::Value* val);
        bool isUsedOutsideLoop(ir::Loop* lp,ir::Value* val);
        int getConstantEndvarIndVarIterCnt(ir::Loop* lp,ir::IndVar* idv);
        ir::Value* addCalcIterCntInstructions(ir::Loop* lp,ir::IndVar* idv,ir::IRBuilder& builder);
        void normalizeIcmpAndBr(ir::Loop* lp,ir::IndVar* idv);
        void exchangeIcmpOp(ir::ICmpInst* icmpInst);
        void reverseIcmpOp(ir::ICmpInst* icmpInst);
        void exchangeBrDest(ir::BranchInst* brInst);
        void visitPhi(ir::Loop* lp,ir::PhiInst* phiinst);
        int findAddSubChain(ir::Loop* lp,ir::PhiInst* phiinst,ir::BinaryInst* nowInst);
        void getSCEVValue(ir::Loop* lp,ir::PhiInst* phiinst,
            std::vector<ir::BinaryInst*>&instsChain);
        void SCEVReduceInstr(ir::Loop* lp,SCEVValue* scevVal,ir::Value* itercnt,ir::IRBuilder& builder);
};
}