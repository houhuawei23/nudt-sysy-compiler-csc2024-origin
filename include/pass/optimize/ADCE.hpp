#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include <set>
#include <queue>

namespace pass {
class ADCE : public FunctionPass {
  public:
    std::string name() const override { return "ADCE"; }
    void run(ir::Function* func, TopAnalysisInfoManager* tp) override;

  private:
    pdomTree* pdctx;
    void ADCEInfoCheck(ir::Function* func);
    ir::BasicBlock* getTargetBB(ir::BasicBlock* bb);
};

}  // namespace pass
