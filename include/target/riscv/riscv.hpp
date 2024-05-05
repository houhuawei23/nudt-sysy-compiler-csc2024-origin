#pragma once
#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "autogen/riscv/InstInfoDecl.hpp"
namespace mir::RISCV {
/*
 * @brief: RISCVRegister enum
 * @note:
 *      Risc-V架构 64位
 * @param:
 *      1. GRBegin - General Register
 *      2. FRBegin - Float Register
 */
//! do not delete following line
// clang-format off
enum RISCVRegister : uint32_t {
    GPRBegin,
    X0=GPRBegin, X1, X2, X3, X4, X5, X6, X7,
    X8, X9, X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X19, X20, X21, X22, X23,
    X24, X25, X26, X27, X28, X29, X30, X31,
    GPREnd,
    FPRBegin,
    F0=FPRBegin, F1, F2, F3, F4, F5, F6, F7,
    F8, F9, F10, F11, F12, F13, F14, F15,
    F16, F17, F18, F19, F20, F21, F22, F23,
    F24, F25, F26, F27, F28, F29, F30, F31,
    FPREnd,
};

static void dumpISAReg(std::ostream& os, MIROperand* operand) {
    assert(operand != nullptr);
    auto reg = operand->reg();
    switch (reg) {
    case RISCV::X10:
        os << "a0";
        break;
    case RISCV::X11:
        os << "a1";
        break;
    case RISCV::X12:
        os << "a2";
        break;
    case RISCV::X13:
        os << "a3";
        break;
    case RISCV::X14:
        os << "a4";
        break;
    case RISCV::X15:
        os << "a5";
        break;
    case RISCV::X16:
        os << "a6";
        break;
    case RISCV::X17:
        os << "a7";
        break;
    case RISCV::X5:
        os << "t0";
        break;
    case RISCV::X6:
        os << "t1";
        break;
    case RISCV::X7:
        os << "t2";
        break;
    case RISCV::X28:
        os << "t3";
        break;
    case RISCV::X29:
        os << "t4";
        break;
    case RISCV::X30:
        os << "t5";
        break;
    case RISCV::X31:
        os << "t6";
        break;
    case RISCV::X8:
        os << "s0";
        break;
    case RISCV::X9:
        os << "s1";
        break;
    case RISCV::X18:
        os << "s2";
        break;
    case RISCV::X19:
        os << "s3";
        break;
    case RISCV::X20:
        os << "s4";
        break;
    case RISCV::X21:
        os << "s5";
        break;
    case RISCV::X22:
        os << "s6";
        break;
    case RISCV::X23:
        os << "s7";
        break;
    case RISCV::X24:
        os << "s8";
        break;
    case RISCV::X25:
        os << "s9";
        break;
    case RISCV::X26:
        os << "s10";
        break;
    case RISCV::X27:
        os << "s11";
        break;
    case RISCV::X3:
        os << "gp";
        break;

    default:
        break;
    }
}

// return address
static auto ra = MIROperand::as_isareg(RISCVRegister::X1, OperandType::Int64);

// stack pointer
static auto sp = MIROperand::as_isareg(RISCVRegister::X2, OperandType::Int64);

// utils function
constexpr bool isOperandGR(MIROperand& operand) {
    if (not operand.is_reg() || not isIntType(operand.type()))
        return false;
    auto reg = operand.reg();
    return GPRBegin <= reg && reg < GPREnd;
}

static std::string_view getRISCVGPRTextualName(uint32_t idx) noexcept {
    // NOLINTNEXTLINE
    constexpr std::string_view name[] = {
        "x0", "ra", "sp",  "gp",  "tp", "t0", "t1", "t2",  // zero
        "s0", "s1", "a0",  "a1",  "a2", "a3", "a4", "a5",  //
        "a6", "a7", "s2",  "s3",  "s4", "s5", "s6", "s7",  //
        "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",  //
    };
    return name[idx];
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
        } else if (isISAReg(operand->reg())) {
            dumpISAReg(os, operand);
        } else if (isOperandGR(*operand)) {
            os << getRISCVGPRTextualName(operand->reg());
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
        if (operand->type() == OperandType::HighBits) {
            os << "%pcrel_hi(";
        } else if (operand->type() == OperandType::LowBits) {
            os << "%pcrel_lo(";
        }
        os << operand->reloc()->name();
        if (operand->type() != OperandType::Special) {
            os << ")";
        }
    } else {
        os << "unknown ";
    }
    return os;
}

constexpr bool isOperandImm12(MIROperand* operand) {
    if (operand->is_reloc() && operand->type() == OperandType::LowBits)
        return true;
    return operand->is_imm() && isSignedImm<12>(operand->imm());
}

constexpr bool isOperandNonZeroImm12(MIROperand* operand) {
    return isOperandImm12(operand) && operand->imm() != 0;
}

}  // namespace mir::RISCV
