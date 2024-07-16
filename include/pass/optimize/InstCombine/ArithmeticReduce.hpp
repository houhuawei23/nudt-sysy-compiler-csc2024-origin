#pragma once
#include "ir/ir.hpp"
#include "pass/pass.hpp"

namespace pass {

class ArithmeticReduce final : public FunctionPass {
  bool runOnBlock(ir::IRBuilder& builder, ir::BasicBlock& block);

 public:
  void run(ir::Function* func, topAnalysisInfoManager* tp) override {
    ir::IRBuilder builder;

    for (auto block : func->blocks()) {
      runOnBlock(builder, *block);
    }
  }
};
};  // namespace pass