#include "pass/optimize/Utils/BlockUtils.hpp"

using namespace ir;

namespace pass {
bool reduceBlock(IRBuilder& builder, BasicBlock& block, const BlockReducer& reducer) {
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
/*
block:
  ...
  inst1
  after
  inst2
  ...

otherblock:

=>
preBlock:
  ...
  inst1
  after

postBlock:
  inst2
  ...
  
otherblock:

*/
BasicBlock* splitBlock(BasicBlockList& blocks,
                       BasicBlockList::iterator blockIter,
                       InstructionList::iterator after) {
  auto preBlock = *blockIter;
  auto postBlock = utils::make<BasicBlock>("", preBlock->function());

  const auto beg = std::next(after);
  const auto end = preBlock->insts().end();

  for(auto iter = beg; iter!= end; ) {
    auto next = std::next(iter);
    postBlock->emplace_back_inst(*iter);
    preBlock->insts().erase(iter);
    iter = next;
  }
  // insert postBlock after preBlock
  blocks.insert(std::next(blockIter), postBlock);
  return postBlock;
}

}  // namespace pass