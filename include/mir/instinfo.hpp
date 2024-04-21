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
    virtual ~InstInfo() = default;

    virtual uint32_t operand_num() = 0;

    virtual OperandFlag operand_flag(uint32_t idx) = 0;

    virtual uint32_t inst_flag() = 0;

    virtual std::string_view name() = 0;

    virtual void print(std::ostream& out, MIRInst& inst, bool printComment) = 0;
};

class TargetInstInfo {
   public:
    TargetInstInfo() = default;
    ~TargetInstInfo() = default;
    virtual InstInfo& get_instinfo(uint32_t opcode);
    InstInfo& get_instinfo(MIRInst* inst) {
        return get_instinfo(inst->opcode());
    }

    // virtual bool match_branch(const MIRInst* inst, MIRBlock* target)
};

//! helper functions
constexpr bool isOperandVReg(MIROperand* operand) {
    return operand->is_reg();
    // && isVirtualReg(operand->reg())
}

void dumpVirtualReg(std::ostream& os, MIROperand* operand);

}  // namespace mir

namespace mir::GENERIC {
struct OperandDumper {
    MIROperand* operand;
};

static std::ostream& operator<<(std::ostream& os, OperandDumper opdp) {
    auto operand = opdp.operand;
    if (operand->is_reg()) {
        if (isVirtualReg(operand->reg())) {
            dumpVirtualReg(os, operand);
        } else if (isStackObject(operand->reg())) {
            os << "so" << (operand->reg() ^ stackObjectBegin);
        } else {
            os << "[reg]";
        }
        // os << "reg: " << operand->reg() ;

    } else if (operand->is_imm()) {
        os << "imm: " << operand->imm();
    } else if (operand->is_prob()) {
        // os << "prob: " << operand->prob();
        os << "prob ";
    } else if (operand->is_reloc()) {
        // operand->reloc()-
        // os << "reloc ";
        os << operand->reloc()->name();
    } else {
        os << "unknown ";
    }
    return os;
}

}  // namespace mir::GENERIC
