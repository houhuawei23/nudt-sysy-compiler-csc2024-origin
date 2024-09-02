#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include <vector>
#include <set>

namespace pass{
    class markParallel;
    struct resPhi;
    struct resPhi{
        ir::PhiInst* phi;//对应的phi指令指针
        bool isAdd;//表示最后的结果汇合将使用+
        bool isMul;//表示最后的结果汇合将使用*
        bool isSub;//表示最后的结果汇合将使用preVal-gv1-gv2-gv3-gv4
        bool isModulo;//在+的基础上，最后的结果每次汇合需要mod
        ir::Value* mod;//if isModulo,使用这个值进行
    };

    class markParallel:public FunctionPass{
        public:
            void run(ir::Function* func,TopAnalysisInfoManager* tp)override;
            std::string name() const override;
        private:
            TopAnalysisInfoManager* topmana;
            dependenceInfo* dpctx;
            DomTree* domctx;
            LoopInfo* lpctx;
            sideEffectInfo* sectx;
            CallGraph* cgctx;
            IndVarInfo* idvctx;
            parallelInfo* parctx;
            void runOnLoop(ir::Loop* lp);
            void printParallelInfo(ir::Function* func);
            resPhi* getResPhi(ir::PhiInst* phi,ir::Loop* lp);
            bool isSimplyLpInvariant(ir::Loop* lp,ir::Value* val);
            bool isFuncParallel(ir::Loop* lp,ir::CallInst* callinst);
            ir::Value* getBaseAddr(ir::Value* ptr);
    };
};