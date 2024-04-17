#pragma once
#include "mir/mir.hpp"

namespace mir {

enum OperandFlag : uint32_t {
    OperandFlagUse = 1 << 0,
    OperandFlagDef = 1 << 1,
    OperandFlagMetadata = 1 << 2,
};

enum InstFlag : uint32_t {
    InstFlagNone = 0,
    InstFlagLoad = 1 << 0,
    InstFlagStore = 1 << 1,
    InstFlagTerminator = 1 << 2,
    InstFlagBranch = 1 << 3,
    InstFlagCall = 1 << 4,
    InstFlagNoFallthrough = 1 << 5,
    InstFlagPush = 1 << 6,
    InstFlagLoadConstant = 1 << 7,
    InstFlagRegDef = 1 << 8,  // def ISA register
    InstFlagCommutative = 1 << 9,
    InstFlagReturn = 1 << 10,
    InstFlagLegalizePreRA = 1 << 11,
    InstFlagWithDelaySlot = 1 << 12,
    InstFlagRegCopy = 1 << 13,
    InstFlagConditional = 1 << 14,
    InstFlagPCRel = 1 << 15,
    InstFlagMultiDef = 1 << 16,
    InstFlagInOrder = 1 << 17,
    InstFlagPadding = 1 << 18,
    InstFlagIndirectJump = 1 << 19,
    InstFlagSideEffect =
        InstFlagLoad | InstFlagStore | InstFlagTerminator | InstFlagBranch |
        InstFlagCall | InstFlagPush | InstFlagRegDef | InstFlagReturn |
        InstFlagWithDelaySlot | InstFlagPadding | InstFlagIndirectJump,
};

class InstInfo {
   public:
    InstInfo() = default;
    ~InstInfo() = default;
    OperandFlag operand_flag(uint32_t idx);
    uint32_t inst_flag();
    void print(std::ostream& out, MIRInst& inst, bool printComment);
    uint32_t operand_num();
};

class TargetInstInfo {
   public:
    TargetInstInfo() = default;
    ~TargetInstInfo() = default;
    InstInfo& get_instinfo(uint32_t opcode);

    // virtual bool match_branch(const MIRInst* inst, MIRBlock* target)
};

//! helper functions
constexpr bool isOperandVReg(MIROperand* operand) {
    return operand->is_reg();
    // && isVirtualReg(operand->reg())
}

}  // namespace mir
