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
            void runOnLoop(ir::Loop* lp,TopAnalysisInfoManager* tp);
            domTree* domctx;
            loopInfo* lpctx;
            indVarInfo* idvctx;
            sideEffectInfo* sectx;
            dependenceInfoForLoops* dpctx;
            memRW* makeMemRW(ir::Value* ptr,memOp memop,ir::Instruction* inst);
            subAddrIdx* getSubAddrIdx(ir::GetElementPtrInst* gep,ir::Loop* lp,ir::Value* baseptr);
            idxType getIdxType(ir::Value* idxVal,ir::Loop* lp,ir::indVar* idv);
            bool isSimplyLpInvariant(ir::Value* val,ir::Loop* lp);
    };

    class dependenceAnalysisInfoCheck:public FunctionPass{
        public:
            void run(ir::Function* func,TopAnalysisInfoManager* tp)override;
            std::string name() const override { return "dependence analysis infocheck"; }
        private:
            dependenceInfoForLoops* dpctx;
            loopInfo* lpctx;
    };

};