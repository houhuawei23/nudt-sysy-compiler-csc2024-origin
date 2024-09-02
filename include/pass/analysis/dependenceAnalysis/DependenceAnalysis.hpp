#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include "pass/analysis/dependenceAnalysis/dpaUtils.hpp"

namespace pass{

    class DependenceAnalysis;
    // class dependenceAnalysisInfoCheck;

    class DependenceAnalysis:public FunctionPass{
        public:
            void run(ir::Function* func,TopAnalysisInfoManager* tp)override;
            std::string name() const override { return "dependence analysis"; }
        private:
            TopAnalysisInfoManager* topmana;
            DomTree* domctx;
            LoopInfo* lpctx;
            IndVarInfo* idvctx;
            sideEffectInfo* sectx;
            CallGraph* cgctx;
            dependenceInfo* dpctx;
            void runOnLoop(ir::Loop* lp);
            void makeGepIdx(ir::Loop* lp,ir::IndVar* idv,gepIdx* gepidx);
            bool isSimplyLoopInvariant(ir::Loop* lp,ir::Value* val);
            bool isIDVPLUSMINUSFORMULA(ir::IndVar* idv,ir::Value* val,ir::Loop* lp);
            int isTwoGepIdxPossiblySame(gepIdx*gepidx1,gepIdx*gepidx2,ir::Loop* lp,ir::IndVar* idv);
            int isTwoIdxPossiblySame(ir::Value* val1,ir::Value* val2,idxType type1,idxType type2,ir::Loop* lp,ir::IndVar* idv);
    };
    
};