#pragma once
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {
class SCCP : public FunctionPass {
  public:
    void run(ir::Function* func, topAnalysisInfoManager* tp) override;

  private:
    bool cleanCFG(ir::Function* func);
    bool addConstFlod(ir::Instruction* inst);
    bool SCPrun(ir::Function* func, topAnalysisInfoManager* tp);
    void searchCFG(ir::BasicBlock* bb);
};
}  // namespace pass