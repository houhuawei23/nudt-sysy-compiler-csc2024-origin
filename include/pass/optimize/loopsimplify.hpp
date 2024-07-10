#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class Inline : public FunctionPass {
   private:
    
   public:
    void run(ir::Function* func, topAnalysisInfoManager* tp) override;
    
};
}  // namespace pass