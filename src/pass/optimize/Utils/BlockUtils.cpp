#include "pass/optimize/Utils/BlockUtils.hpp"

using namespace ir;

namespace pass {
bool reduceBlock(IRBuilder& builder,
                 BasicBlock& block,
                 const BlockReducer& reducer) {
  auto& insts = block.insts();
  bool modified = false;
  const auto oldSize = insts.size();
  for (auto iter = insts.begin(); iter != insts.end(); iter++) {
    auto inst = *iter;

    builder.set_pos(&block, iter);
    if (auto value = reducer(inst)) {
      assert(value != inst);
      // modified |= ins
      inst->replaceAllUseWith(value);
      modified = true;
      // std::cerr << "Replaced!" << std::endl;
    }
  }
  const auto newSize = insts.size();
  modified |= (newSize != oldSize);
  return modified;
}

}  // namespace pass