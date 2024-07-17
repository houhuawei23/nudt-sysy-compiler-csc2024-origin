#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class Inline : public ModulePass {
   private:
    callGraph* cgctx;
   public:
    void run(ir::Module* module, topAnalysisInfoManager* tp) override;
    void callinline(ir::CallInst* call);
    std::vector<ir::CallInst*> getcall(ir::Module* module,ir::Function* function);//找出调用了function的call指令
    std::vector<ir::Function*> getinlineFunc(ir::Module* module);
};
}  // namespace pass