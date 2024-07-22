#pragma once
#include <unordered_set>
#include "mir/mir.hpp"
#include "mir/LiveInterval.hpp"
#include "mir/instinfo.hpp"
#include "mir/lowering.hpp"
#include "support/arena.hpp"
#include <optional>

namespace mir {
class CodeGenContext;
class LoweringContext;
class ISelContext : public MIRBuilder {
  CodeGenContext& mCodeGenCtx;
  std::unordered_map<RegNum, MIRInst*> mInstMap, mConstantMap;

  // mReplaceList
  std::unordered_map<MIROperand*, MIROperand*> mReplaceMap;

  std::unordered_set<MIRInst*> mRemoveWorkList, mReplaceList;

  std::unordered_map<MIROperand*, uint32_t> mUseCnt;

public:
  ISelContext(CodeGenContext& ctx) : mCodeGenCtx(ctx) {}
  void run_isel(MIRFunction* func);
  bool has_one_use(MIROperand* op);
  MIRInst* lookup_def(MIROperand* op);

  void remove_inst(MIRInst* inst);
  void replace_operand(MIROperand* src, MIROperand* dst);

  MIROperand* get_inst_def(MIRInst* inst);

  void insert_inst(MIRInst* inst) {
    assert(inst != nullptr);
    mCurrBlock->insts().emplace(mInsertPoint, inst);
  }
  CodeGenContext& codegen_ctx() { return mCodeGenCtx; }
  MIRBlock* curr_block() { return mCurrBlock; }
};

class InstLegalizeContext final : public MIRBuilder {
public:
  MIRInst*& inst;
  MIRInstList& instructions;
  MIRInstList::iterator iter;
  CodeGenContext& codeGenCtx;
  std::optional<std::list<std::unique_ptr<MIRBlock>>::iterator> blockIter;
  MIRFunction& func;

public:
  InstLegalizeContext(
    MIRInst*& i,
    MIRInstList& insts,
    MIRInstList::iterator iter,
    CodeGenContext& ctx,
    std::optional<std::list<std::unique_ptr<MIRBlock>>::iterator> biter,
    MIRFunction& f)
    : inst(i),
      instructions(insts),
      iter(iter),
      codeGenCtx(ctx),
      blockIter(biter),
      func(f) {}
};

class TargetISelInfo {
public:
  virtual ~TargetISelInfo() = default;
  virtual bool isLegalInst(uint32_t opcode) const = 0;

  virtual bool match_select(MIRInst* inst, ISelContext& ctx) const = 0;

  /* */
  virtual void legalizeInstWithStackOperand(InstLegalizeContext& ctx,
                                            MIROperand* op,
                                            StackObject& obj) const = 0;

  virtual void postLegalizeInst(const InstLegalizeContext& ctx) const = 0;
  virtual MIROperand* materializeFPConstant(
    float fpVal,
    LoweringContext& loweringCtx) const = 0;
};

static bool isCompareOp(MIROperand* operand, CompareOp cmpOp) {
  auto op = static_cast<uint32_t>(operand->imm());
  return op == static_cast<uint32_t>(cmpOp);
}

static bool isICmpEqualityOp(MIROperand* operand) {
  const auto op = static_cast<CompareOp>(operand->imm());
  switch (op) {
    case CompareOp::ICmpEqual:
    case CompareOp::ICmpNotEqual:
      return true;
    default:
      return false;
  }
}

//! helper function to create a new MIRInst

uint32_t select_copy_opcode(MIROperand* dst, MIROperand* src);

inline MIROperand* getNeg(MIROperand* operand) {
  return MIROperand::asImm(-operand->imm(), operand->type());
}

inline MIROperand* getHighBits(MIROperand* operand) {
  assert(isOperandReloc(operand));
  return utils::make<MIROperand>(operand->storage(), OperandType::HighBits);
}
inline MIROperand* getLowBits(MIROperand* operand) {
  assert(isOperandReloc(operand));
  return utils::make<MIROperand>(operand->storage(), OperandType::LowBits);
}

}  // namespace mir
