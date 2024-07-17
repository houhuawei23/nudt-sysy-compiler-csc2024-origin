#include "pass/optimize/InstCombine/ArithmeticReduce.hpp"
#include "pass/optimize/Utils/PatternMatch.hpp"
#include "pass/optimize/Utils/BlockUtils.hpp"

// #include "support/"
#include <iostream>
#include <cassert>
#include <algorithm>
using namespace ir;
namespace pass {

bool ArithmeticReduce::runOnBlock(ir::IRBuilder& builder,
                                  ir::BasicBlock& block) {
  bool debug = false;
  bool modified = false;
  auto reducer = [&](Instruction* inst) -> Value* {
    if(debug) {
      std::cerr << "Checking: " << inst->valueId() << std::endl;
    }
    // commutative c x -> commutative x c
    auto isConst = [](Value* v) { return v->isa<Constant>(); };

    if (const auto biInst = inst->dynCast<BinaryInst>()) {
      if (biInst->isCommutative()) {
        if (biInst->operand(0)->isa<Constant>() and
            !biInst->operand(1)->isa<Constant>()) {
          auto& operands = biInst->operands();
          std::swap(operands[0], operands[1]);
          modified = true;
        }
      }
    }

    MatchContext<Value> matchCtx{inst};
    Value *v1, *v2, *v3, *v4;
    int64_t i1, i2;
    double f1, f2;

    // add(add(x, c1), c2) = add(x, c1+c2)
    if (add(add(any(v1), int_(i1)), int_(i2))(matchCtx)) {
      return builder.makeInst<BinaryInst>(ValueId::vADD, v1->type(), v1,
                                          Constant::gen_i32(i1 + i2));
    }

    return nullptr;
  };
  const auto ret = reduceBlock(builder, block, reducer);
  return modified | ret;
}
};  // namespace pass