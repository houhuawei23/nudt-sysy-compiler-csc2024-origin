#pragma once
#include "../../../include/ir/ir.hpp"
#include "../../../include/pass/pass.hpp"
#include <set>
#include <queue>

namespace pass {
class ADCE : public FunctionPass {
  public:
    void run(ir::Function* func, topAnalysisInfoManager* tp) override;

  private:
    pdomTree* pdctx;
    void ADCEInfoCheck(ir::Function* func);
    ir::BasicBlock* getTargetBB(ir::BasicBlock* bb);
};

}  // namespace pass
