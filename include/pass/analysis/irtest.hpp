#pragma once
#include "pass/pass.hpp"

namespace pass {
class irCheck : public ModulePass {
  public:
    void run(ir::Module* ctx, TopAnalysisInfoManager* tp) override;

  private:
    bool runDefUseTest(ir::Function* func);
    bool runPhiTest(ir::Function* func);
    bool runCFGTest(ir::Function* func);
    bool checkDefUse(ir::Value* val);
    bool checkPhi(ir::PhiInst* phi);
    bool checkFuncInfo(ir::Function* func);
    bool checkAllocaOnlyInEntry(ir::Function* func);
};
}  // namespace pass