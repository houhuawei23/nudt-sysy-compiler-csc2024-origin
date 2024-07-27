#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class loopsimplify : public FunctionPass {
   private:
    
   public:
    ir::BasicBlock* insertUniqueBackedgeBlock(ir::Loop* L, ir::BasicBlock* preheader,TopAnalysisInfoManager* tp);
    ir::BasicBlock* insertUniquePreheader(ir::Loop* L,TopAnalysisInfoManager* tp);
    void insertUniqueExitBlock(ir::Loop* L,TopAnalysisInfoManager* tp);
    bool simplifyOneLoop(ir::Loop* L,TopAnalysisInfoManager* tp);
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;
    
};
}  // namespace pass