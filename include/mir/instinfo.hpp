#pragma once
#include "mir/mir.hpp"

namespace mir {
/*
 * @brief: OperandFlag enum
 * @note:
 *      Operand Flag (操作数的相关状态)
 *          1. OperandFlagUse --> 被使用
 *          2. OperandFlagDef --> 被定义
 *          3. OperandFlagMetadata --> 立即数
 */
enum OperandFlag : uint32_t {
  OperandFlagUse = 1 << 0,
  OperandFlagDef = 1 << 1,
  OperandFlagMetadata = 1 << 2,
};

/*
 * @brief: InstFlag enum
 * @note:
 *      Instruction Flag (指令的相关状态 --> 指明属于什么指令)
 */
enum InstFlag : uint32_t {
  InstFlagNone = 0,

  InstFlagLoad = 1 << 0,
  InstFlagStore = 1 << 1,

  InstFlagTerminator = 1 << 2,

  InstFlagBranch = 1 << 3,
  InstFlagCall = 1 << 4,           // call instruction
  InstFlagNoFallThrough = 1 << 5,  // unconditional jump - j/jr/return
  InstFlagPush = 1 << 6,
  InstFlagLoadConstant = 1 << 7,
  InstFlagRegDef = 1 << 8,       // def ISA register
  InstFlagCommutative = 1 << 9,  // exchangable - add/sub/...
  InstFlagReturn = 1 << 10,
  InstFlagLegalizePreRA = 1 << 11,
  InstFlagWithDelaySlot = 1 << 12,
  InstFlagRegCopy = 1 << 13,
  InstFlagConditional = 1 << 14,
  InstFlagPCRel = 1 << 15,
  InstFlagMultiDef = 1 << 16,
  InstFlagInOrder = 1 << 17,
  InstFlagPadding = 1 << 18,
  InstFlagIndirectJump = 1 << 19,  // jump register - jr
  InstFlagSideEffect = InstFlagLoad | InstFlagStore | InstFlagTerminator |
                       InstFlagBranch | InstFlagCall | InstFlagPush |
                       InstFlagRegDef | InstFlagReturn | InstFlagWithDelaySlot |
                       InstFlagPadding | InstFlagIndirectJump,
};

/*
 * @brief: InstInfo Class (抽象基类)
 * @note:
 *      1. Instruction Information (存储各类不同指令的相关信息)
 *      2. 各类具体架构的指令集中的各个指令继承于此抽象基类
 */
class InstInfo {
public:
  InstInfo() = default;
  virtual ~InstInfo() = default;

public:  // get function
  virtual uint32_t operand_num() const = 0;
  virtual OperandFlag operand_flag(uint32_t idx) const = 0;
  virtual uint32_t inst_flag() const = 0;
  virtual std::string_view name() const = 0;

public:  // print
  virtual void print(std::ostream& out,
                     MIRInst& inst,
                     bool printComment) const = 0;
};

/*
 * @brief: TargetInstInfo Class
 * @note:
 *      Target Instruction Information (目标机器架构的指令集信息)
 */
class TargetInstInfo {
public:
  TargetInstInfo() = default;
  ~TargetInstInfo() = default;

public:  // get function
  virtual const InstInfo& getInstInfo(uint32_t opcode) const;
  const InstInfo& getInstInfo(MIRInst* inst) const {
    return getInstInfo(inst->opcode());
  }

public:  // match function
  virtual bool matchBranch(MIRInst* inst,
                           MIRBlock*& target,
                           double& prob) const;

  bool matchCopy(MIRInst* inst, MIROperand& dst, MIROperand& src) const;
  bool matchConditionalBranch(MIRInst* inst,
                              MIRBlock*& target,
                              double& prob) const;
  bool matchUnconditionalBranch(MIRInst* inst,
                                MIRBlock*& Target,
                                double& prob) const;
};

// utils function
constexpr bool isOperandVRegORISAReg(const MIROperand& operand) {
  return operand.isReg() &&
         (isVirtualReg(operand.reg()) || isISAReg(operand.reg()));
}
constexpr bool isOperandISAReg(const MIROperand& operand) {
  return operand.isReg() && isISAReg(operand.reg());
}
constexpr bool isOperandVReg(const MIROperand& operand) {
  return operand.isReg() && isVirtualReg(operand.reg());
}

constexpr bool requireFlag(InstFlag flag, InstFlag required) {
  return (static_cast<uint32_t>(flag) & static_cast<uint32_t>(required)) ==
         static_cast<uint32_t>(required);
}

constexpr bool requireFlag(uint32_t flag, InstFlag required) noexcept {
  return (static_cast<uint32_t>(flag) & static_cast<uint32_t>(required)) ==
         static_cast<uint32_t>(required);
}

constexpr bool requireOneFlag(uint32_t flag, uint32_t required) {
  return (static_cast<uint32_t>(flag) & static_cast<uint32_t>(required)) != 0;
}

constexpr bool isOperandIReg(const MIROperand& operand) {
  return operand.isReg() && operand.type() <= OperandType::Int64;
}

constexpr bool isOperandBoolReg(const MIROperand& operand) {
  return operand.isReg() && operand.type() == OperandType::Bool;
}

constexpr bool isOperandReloc(const MIROperand& operand) {
  return operand.isReloc() && operand.type() == OperandType::Special;
}
constexpr bool isOperandVRegOrISAReg(const MIROperand& operand) {
  return operand.isReg() &&
         (isVirtualReg(operand.reg()) || isISAReg(operand.reg()));
}

constexpr bool isOperandStackObject(const MIROperand& operand) {
  return operand.isReg() && isStackObject(operand.reg());
}

template <uint32_t N>
constexpr bool isSignedImm(intmax_t imm) {
  static_assert(N < 64);
  constexpr auto x = static_cast<intmax_t>(1) << (N - 1);
  return -x <= imm && imm < x;
}

void dumpVirtualReg(std::ostream& os, const MIROperand& operand);

}  // namespace mir

namespace mir::GENERIC {}  // namespace mir::GENERIC
