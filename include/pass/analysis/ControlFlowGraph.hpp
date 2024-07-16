#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"
#include <iostream>
namespace pass {
class CFGAnalysis : public FunctionPass {
  bool check(std::ostream& os, ir::Function* func) const;

 public:
  void run(ir::Function* func, topAnalysisInfoManager* tp) override;
};

}  // namespace pass