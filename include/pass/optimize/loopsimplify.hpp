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
    ir::BasicBlock* insertUniqueBackedgeBlock(ir::Loop* L, ir::BasicBlock* preheader,topAnalysisInfoManager* tp);
    ir::BasicBlock* insertUniquePreheader(ir::Loop* L,topAnalysisInfoManager* tp);
    void insertUniqueExitBlock(ir::Loop* L,topAnalysisInfoManager* tp);
    bool simplifyOneLoop(ir::Loop* L,topAnalysisInfoManager* tp);
    void run(ir::Function* func, topAnalysisInfoManager* tp) override;
    
};
}  // namespace pass