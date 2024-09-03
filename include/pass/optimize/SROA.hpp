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
            DomTree* domctx;
            LoopInfo* lpctx;
            sideEffectInfo* sectx;
            dependenceInfo* dpctx;
            IndVarInfo* idvctx;
            LoopDependenceInfo* depInfoForLp;
            void runOnLoop(ir::Loop* lp);
            ir::AllocaInst* createNewLocal(ir::Type* allocaType,ir::Function* func);
            bool replaceAllUseInLpIdv(ir::GetElementPtrInst* gep,ir::Loop* lp,ir::AllocaInst* newAlloca,bool isOnlyRead,bool isOnlyWrite);
            bool replaceAllUseInLpForLpI(ir::GetElementPtrInst* gep,ir::Loop* lp,ir::AllocaInst* newAlloca,bool isOnlyRead,bool isOnlyWrite);
            int isTwoGepIdxPossiblySame(GepIdx* gepidx1,GepIdx* gepidx2,ir::Loop* lp,ir::IndVar* idv);
            int isTwoIdxPossiblySame(ir::Value* val1,ir::Value* val2,IdxType type1,IdxType type2,ir::Loop* lp,ir::IndVar* idv);
            bool isSimplyLoopInvariant(ir::Loop* lp,ir::Value* val);
    };
}