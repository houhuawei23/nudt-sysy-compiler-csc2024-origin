#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class SCP : public FunctionPass {
  public:
    void run(ir::Function* func, topAnalysisInfoManager* tp) override;

  private:
    void addConstFlod(ir::Instruction* inst);
};
}  // namespace pass