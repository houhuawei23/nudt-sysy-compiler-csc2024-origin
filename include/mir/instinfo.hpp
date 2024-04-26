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
    InstFlagCall = 1 << 4,
    InstFlagNoFallThrough = 1 << 5,
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

constexpr bool requireFlag(InstFlag flag, InstFlag required) {
    return (static_cast<uint32_t>(flag) & static_cast<uint32_t>(required)) == static_cast<uint32_t>(required);
}

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
        virtual uint32_t operand_num() = 0;
        virtual OperandFlag operand_flag(uint32_t idx) = 0;
        virtual uint32_t inst_flag() = 0;
        virtual std::string_view name() = 0;

    public:  // print
        virtual void print(std::ostream& out, MIRInst& inst, bool printComment) = 0;
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
        virtual InstInfo& get_instinfo(uint32_t opcode);
        InstInfo& get_instinfo(MIRInst* inst) { return get_instinfo(inst->opcode()); }

    public:  // match function
        virtual bool matchBranch(MIRInst* inst, MIRBlock*& target, double& prob) ;
        bool matchConditionalBranch(MIRInst* inst, MIRBlock*& target, double& prob) ;
        bool matchUnconditionalBranch(MIRInst* inst, MIRBlock*& Target, double& prob) ;

};

// utils function
constexpr bool isOperandVRegORISAReg(MIROperand* operand) {
    return operand->is_reg() && (isVirtualReg(operand->reg()) || isISAReg(operand->reg()));
}
constexpr bool requireFlag(uint32_t flag, InstFlag required) noexcept {
    return (static_cast<uint32_t>(flag) & static_cast<uint32_t>(required)) ==
           static_cast<uint32_t>(required);
}

constexpr bool isOperandVReg(MIROperand* operand) {
    return operand->is_reg() && isVirtualReg(operand->reg());
}
constexpr bool isOperandIReg(MIROperand* operand) {
    // TODO: not implemented yet

    return false;
}
constexpr bool isOperandReloc(MIROperand* operand) {
    return operand->is_reloc() && operand->type() == OperandType::Special;
}
constexpr bool isOperandVRegOrISAReg(MIROperand* operand) {
    return operand->is_reg() && (isVirtualReg(operand->reg()) || isISAReg(operand->reg()));
}


template <uint32_t N>
constexpr bool isSignedImm(intmax_t imm) {
    static_assert(N < 64);
    constexpr auto x = static_cast<intmax_t>(1) << (N - 1);
    return -x <= imm && imm < x;
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

    } else if (operand->is_imm()) {
        os << "imm: " << operand->imm();
    } else if (operand->is_prob()) {
        os << "prob ";
    } else if (operand->is_reloc()) {
        os << operand->reloc()->name();
    } else {
        os << "unknown ";
    }
    return os;
}

}  // namespace mir::GENERIC
