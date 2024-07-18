#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"
// 稀疏常量传播: SCCP (Sparse Conditional Constant Propagation)
namespace pass{
    class mySCCP:public FunctionPass{
        public:
            void run(ir::Function* func, topAnalysisInfoManager* tp)override;
            
        private:
            bool getExecutableFlag(ir::BasicBlock* a, ir::BasicBlock* b);
            void addConstFlod(ir::Instruction*inst);
            bool getDeadFlag(ir::BasicBlock* a);
            void branchInstFlod(ir::BranchInst* brInst);
            void deleteDeadBlock(ir::BasicBlock* bb);
    };
}