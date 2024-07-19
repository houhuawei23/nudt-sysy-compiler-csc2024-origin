#include "ir/infrast.hpp"
#include "ir/function.hpp"
#include "ir/utils_ir.hpp"
#include "ir/instructions.hpp"
#include "support/arena.hpp"
namespace ir {

void Argument::print(std::ostream& os) const {
  os << *type() << " " << name();
}
bool BasicBlock::isTerminal() const {
  if (mInsts.empty()) return false;
  return mInsts.back()->isTerminator();
}

bool BasicBlock::verify(std::ostream& os) const {
  if (mInsts.empty()) return false;
  for (auto inst : mInsts) {
    if (inst->isTerminator() and inst != mInsts.back()) {
      os << "block have terminator inst not at the end" << std::endl;
      return false;
    }
  }
  // end with a terminator inst
  if (not mInsts.back()->isTerminator()) return false;

  return true;
}

void BasicBlock::print(std::ostream& os) const {
  // print all instructions

  os << name() << ":";
  /* comment begin */
  if (not mComment.empty()) {
    os << " ; " << mComment << std::endl;
  } else {
    os << std::endl;
  }
  if (not mPreBlocks.empty()) {
    os << "    ; " << "pres: ";
    for (auto it = pre_blocks().begin(); it != pre_blocks().end(); it++) {
      os << (*it)->name();
      if (std::next(it) != pre_blocks().end()) {
        os << ", ";
      }
    }
    os << std::endl;
  }
  if (not mNextBlocks.empty()) {
    os << "    ; " << "nexts: ";
    for (auto it = next_blocks().begin(); it != next_blocks().end(); it++) {
      os << (*it)->name();
      if (std::next(it) != next_blocks().end()) {
        os << ", ";
      }
    }
    os << std::endl;
  }
  /* comment end */

  for (auto& inst : mInsts) {
    os << "    " << *inst << std::endl;
  }
}

void BasicBlock::emplace_inst(inst_iterator pos, Instruction* i) {
  // Warning: didn't check _is_terminal
  if (i->isTerminator() and isTerminal()) {
    std::cerr << "[ERROR] insert a terminal inst to a terminal bb" << std::endl;
  }
  mInsts.emplace(pos, i);
  i->setBlock(this);
  if (auto phiInst = dyn_cast<PhiInst>(i)) {
    // assume that Phi insts are all at the front of a bb
    size_t index = std::distance(mInsts.begin(), pos);
    mPhiInsts.emplace(std::next(mPhiInsts.begin(), index), phiInst);
  }
}
void BasicBlock::emplace_first_inst(Instruction* inst) {
  // Warning: didn't check _is_terminal
  if (inst->isTerminator() and isTerminal()) {
    std::cerr << "[ERROR] insert a terminal inst to a terminal bb" << std::endl;
  }
  auto pos = mInsts.begin();
  mInsts.emplace(pos, inst);
  inst->setBlock(this);
  if (auto phiInst = dyn_cast<PhiInst>(inst)) mPhiInsts.emplace_front(phiInst);
}

void BasicBlock::emplace_back_inst(Instruction* i) {
  if (isTerminal()) {
    std::cerr << "[ERROR] emplace_back a non-terminal inst to a terminal bb"
              << std::endl;
    assert(false);
    return;
  }
  mInsts.emplace_back(i);
  i->setBlock(this);
  if (auto phiInst = dyn_cast<PhiInst>(i))
    // assert(false and "a phi can not be inserted at the back of a bb");
    mPhiInsts.emplace_back(phiInst);
}
void Instruction::setvarname() {
  auto cur_func = mBlock->function();
  mName = "%" + std::to_string(cur_func->varInc());
}

void BasicBlock::delete_inst(Instruction* inst) {
  // if inst1 use 2, 2->mUses have use user inst
  // in 2, del use of 1
  // if 3 use inst, 3.operands have use(3, 1)
  // first replace use(3, 1)
  // if you want to delete a inst, all use of it must be deleted in advance
  assert(inst->uses().size() == 0);
  for (auto op_use : inst->operands()) {
    auto op = op_use->value();
    op->uses().remove(op_use);
  }
  mInsts.remove(inst);
  if (auto phiInst = dyn_cast<PhiInst>(inst)) mPhiInsts.remove(phiInst);

  // delete inst;
}

void BasicBlock::force_delete_inst(Instruction* inst) {
  // assert(inst->uses().size()==0);
  for (auto op_use : inst->operands()) {
    auto op = op_use->value();
    op->uses().remove(op_use);
  }
  mInsts.remove(inst);
  if (auto phiInst = dyn_cast<PhiInst>(inst)) mPhiInsts.remove(phiInst);
}

void BasicBlock::replaceinst(Instruction* old_inst, Value* new_) {
  inst_iterator pos = find(mInsts.begin(), mInsts.end(), old_inst);
  if (pos != mInsts.end()) {
    if (auto inst = dyn_cast<Instruction>(new_)) {
      emplace_inst(pos, inst);
      old_inst->replaceAllUseWith(inst);
      delete_inst(old_inst);
    } else if (auto constant = dyn_cast<Constant>(new_)) {
      old_inst->replaceAllUseWith(constant);
      delete_inst(old_inst);
    }
  }
}

bool Instruction::isTerminator() {
  return mValueId == vRETURN || mValueId == vBR;
}
bool Instruction::isUnary() {
  return mValueId > vUNARY_BEGIN && mValueId < vUNARY_END;
};
bool Instruction::isBinary() {
  return mValueId > vBINARY_BEGIN && mValueId < vBINARY_END;
};
bool Instruction::isBitWise() {
  return false;
}
bool Instruction::isMemory() {
  return mValueId == vALLOCA || mValueId == vLOAD || mValueId == vSTORE ||
         mValueId == vGETELEMENTPTR;
};
bool Instruction::isNoName() {
  return isTerminator() or mValueId == vSTORE or mValueId == vMEMSET;
}
bool Instruction::isAggressiveAlive() {
  return mValueId == vSTORE or mValueId == vCALL or mValueId == vMEMSET or
         mValueId == vRETURN;
}

// TODO: need modify, use builder to create inst
Instruction* Instruction::copy_inst(std::function<Value*(Value*)> getValue) {
  if (auto allocainst = dyn_cast<AllocaInst>(this)) {
    if (allocainst->isScalar()) {
      return utils::make<AllocaInst>(allocainst->baseType());
    } else { 
      auto basetype = dyn_cast<ArrayType>(allocainst->baseType());
      return utils::make<AllocaInst>(basetype->baseType(), basetype->dims());
    }
  } else if (auto storeinst = dyn_cast<StoreInst>(this)) {
    auto value = getValue(storeinst->operand(0));
    auto addr = getValue(storeinst->operand(1));
    return utils::make<StoreInst>(value, addr);
  } else if (auto loadinst = dyn_cast<LoadInst>(this)) {
    auto ptr = getValue(loadinst->ptr());
    return utils::make<LoadInst>(ptr, loadinst->type(), nullptr);
  } else if (auto returninst = dyn_cast<ReturnInst>(this)) {
    return utils::make<ReturnInst>(getValue(returninst->returnValue()));
  } else if (auto unaryinst = dyn_cast<UnaryInst>(this)) {
    auto value = getValue(unaryinst->value());
    return utils::make<UnaryInst>(unaryinst->valueId(), unaryinst->type(),
                                  value);
  } else if (auto binaryinst = dyn_cast<BinaryInst>(this)) {
    auto lhs = getValue(binaryinst->lValue());
    auto rhs = getValue(binaryinst->rValue());
    return utils::make<BinaryInst>(binaryinst->valueId(), binaryinst->type(),
                                   lhs, rhs, nullptr);
  } else if (auto callinst = dyn_cast<CallInst>(this)) {
    auto callee = callinst->callee();
    std::vector<Value*> args;
    for (auto arg : callinst->rargs()) {
      args.push_back(getValue(arg->value()));
    }
    return utils::make<CallInst>(callee, args);
  } else if (auto branchinst = dyn_cast<BranchInst>(this)) {
    if (branchinst->is_cond()) {
      auto cond = getValue(branchinst->cond());
      auto true_bb = dyn_cast<BasicBlock>(getValue(branchinst->iftrue()));
      auto false_bb = dyn_cast<BasicBlock>(getValue(branchinst->iffalse()));
      return utils::make<BranchInst>(cond, true_bb, false_bb);
    } else {
      auto dest_bb = dyn_cast<BasicBlock>(getValue(branchinst->dest()));
      return utils::make<BranchInst>(dest_bb);
    }
  } else if (auto icmpinst = dyn_cast<ICmpInst>(this)) {
    auto lhs = getValue(icmpinst->lhs());
    auto rhs = getValue(icmpinst->rhs());
    return utils::make<ICmpInst>(icmpinst->valueId(), lhs, rhs, nullptr);
  } else if (auto fcmpinst = dyn_cast<FCmpInst>(this)) {
    auto lhs = getValue(fcmpinst->lhs());
    auto rhs = getValue(fcmpinst->rhs());
    return utils::make<FCmpInst>(fcmpinst->valueId(), lhs, rhs, nullptr);
  } else if (auto bitcastinst = dyn_cast<BitCastInst>(this)) {
    auto value = getValue(bitcastinst->value());
    return utils::make<BitCastInst>(bitcastinst->type(), value, nullptr);
  } else if (auto memsetinst = dyn_cast<MemsetInst>(this)) {
    auto value = getValue(memsetinst->value());
    return utils::make<MemsetInst>(memsetinst->type(), value, nullptr);
  } else if (auto getelemenptrinst = dyn_cast<GetElementPtrInst>(this)) {
    auto value = getValue(getelemenptrinst->value());
    auto idx = getValue(getelemenptrinst->index());
    if (getelemenptrinst->getid() == 0) {
      auto basetype = getelemenptrinst->baseType();
      return utils::make<GetElementPtrInst>(basetype, value, idx);
    } else if (getelemenptrinst->getid() == 1) {
      auto basetype = dyn_cast<ArrayType>(getelemenptrinst->baseType());
      std::vector<size_t> dims = basetype->dims();
      auto curdims = getelemenptrinst->cur_dims();
      return utils::make<GetElementPtrInst>(basetype->baseType(), value, idx,
                                            dims, curdims);
    } else {
      auto basetype = getelemenptrinst->baseType();
      auto curdims = getelemenptrinst->cur_dims();
      return utils::make<GetElementPtrInst>(basetype, value, idx, curdims);
    }
  } else if (auto phiinst = dyn_cast<PhiInst>(this)) {
    return utils::make<PhiInst>(nullptr, phiinst->type());
  } else {
    assert(false and "not support inst type");
    return nullptr;
  }
}
}  // namespace ir