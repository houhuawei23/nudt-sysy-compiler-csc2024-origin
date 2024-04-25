#pragma once
#include "mir/mir.hpp"

namespace mir::RISCV {
/*
 * @brief: RISCVRegister enum
 * @note: 
 *      Risc-V架构 64位
 * @param: 
 *      1. GRBegin - General Register
 *      2. FRBegin - Float Register
 */
enum RISCVRegister : uint32_t {
    GRBegin,
    X0=GRBegin, X1, X2, X3, X4, X5, X6, X7,
    X8, X9, X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X19, X20, X21, X22, X23,
    X24, X25, X26, X27, X28, X29, X30, X31,
    GREnd,

    FRBegin,
    F0=FRBegin, F1, F2, F3, F4, F5, F6, F7,
    F8, F9, F10, F11, F12, F13, F14, F15,
    F16, F17, F18, F19, F20, F21, F22, F23,
    F24, F25, F26, F27, F28, F29, F30, F31,
    FREnd,
};

// return address
static auto ra = MIROperand::as_preg(RISCVRegister::X1, OperandType::Int64);

// stack pointer
static auto sp = MIROperand::as_preg(RISCVRegister::X2, OperandType::Int64);

// utils function
constexpr bool isOperandGR(MIROperand& operand) {
    if (not operand.is_reg() || not isIntType(operand.type())) return false;
    auto reg = operand.reg();
    return GRBegin <= reg && reg < GREnd;
}
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

constexpr bool isOperandImm12(MIROperand* operand) {
    if(operand->is_reloc() && operand->type() == OperandType::LowBits)
        return true;
    return operand->is_imm() && isSignedImm<12>(operand->imm());
}

constexpr bool isOperandNonZeroImm12(MIROperand* operand) {
    return isOperandImm12(operand) && operand->imm() != 0;
}

} // namespace RISCV
