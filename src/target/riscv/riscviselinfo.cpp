#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/iselinfo.hpp"

#include "target/riscv/riscv.hpp"
#include "autogen/riscv/InstInfoDecl.hpp"
#include "autogen/riscv/ISelInfoDecl.hpp"

namespace mir::RISCV {

static MIROperand* getVRegAs(ISelContext& ctx, MIROperand* ref) {
    return MIROperand::as_vreg(ctx.codegen_ctx().next_id(), ref->type());
}

constexpr RISCVInst getIntegerBinaryRegOpcode(uint32_t opcode) {
    switch (opcode) {
        case InstAnd:
            return AND;
        case InstOr:
            return OR;
        case InstXor:
            return XOR;
        default:
            assert(false && "Unsupported binary register instruction");
    }
}
constexpr RISCVInst getIntegerBinaryImmOpcode(uint32_t opcode) {
    switch (opcode) {
        case InstAnd:
            return ANDI;
        case InstOr:
            return ORI;
        case InstXor:
            return XORI;
        default:
            assert(false && "Unsupported binary immediate instruction");
    }
}

static RISCVInst getLoadOpcode(MIROperand* dst) {
    switch (dst->type()) {
        case OperandType::Bool:
        case OperandType::Int8:
            return LB;
        case OperandType::Int16:
            return LH;
        case OperandType::Int32:
            return LW;
        case OperandType::Int64:
            return LD;
        // case OperandType::Float32:
        //     return FLW;
        default:
            assert(false && "Unsupported operand type for load instruction");
            // reportUnreachable(CMMC_LOCATION());
    }
}
static RISCVInst getStoreOpcode(MIROperand* src) {
    switch (src->type()) {
        case OperandType::Bool:
        case OperandType::Int8:
            return SB;
        case OperandType::Int16:
            return SH;
        case OperandType::Int32:
            return SW;
        case OperandType::Int64:
            return SD;
        // case OperandType::Float32:
        //     return FSW;
        default:
            assert(false && "Unsupported operand type for store instruction");
            // reportUnreachable(CMMC_LOCATION());
    }
}

static bool selectAddrOffset(MIROperand* addr,
                             ISelContext& ctx,
                             MIROperand*& base,
                             MIROperand*& offset) {
    // TODO: Select address offset for load/store instructions
    std::cerr << "selectAddrOffset not implemented for RISCV" << std::endl;
    const auto addrInst = ctx.lookup_def(addr);
    if (addrInst) {
        std::cout << "addrInst->opcode() " << addrInst->opcode() << std::endl;
        if (addrInst->opcode() == InstLoadStackObjectAddr) {
            base = addrInst->operand(1);
            offset = MIROperand::as_imm(0, OperandType::Int64);
            return true;
        }
        if (addrInst->opcode() == InstLoadGlobalAddress) {

        }
    }
    if(isOperandIReg(addr)) {
        base = addr;
        offset = MIROperand::as_imm(0, OperandType::Int64);
        return true;
    }
    return false;
}

static bool isOperandI64(MIROperand* op) {
    return op->type() == OperandType::Int64;
}
static bool isOperandI32(MIROperand* op) {
    return op->type() == OperandType::Int32;
}

static bool isZero(MIROperand* operand) {
    if (operand->is_reg() && operand->reg() == RISCV::X0)
        return true;
    return operand->is_imm() && operand->imm() == 0;
}

static MIROperand* getZero(MIROperand* operand) {
    return MIROperand::as_isareg(RISCV::X0, operand->type());
}

static MIROperand* getOne(MIROperand* operand) {
    return MIROperand::as_imm(1, operand->type());
}



}  // namespace mir::RISCV

//! Dont Change !!
#include "autogen/riscv/ISelInfoImpl.cpp"

namespace mir::RISCV {

bool RISCVISelInfo::is_legal_geninst(uint32_t opcode) const {
    return true;
}
static bool legalizeInst(MIRInst* inst, ISelContext& ctx) {
    bool modified = false;

    auto imm2reg = [&](MIROperand*& op) {
        if (op->is_imm()) {
            auto reg = getVRegAs(ctx, op);
            auto tmp = isOperandVRegOrISAReg(reg);
            std::cout << "imm2reg" << tmp << std::endl;
            auto inst = ctx.new_inst(InstLoadImm);
            inst->set_operand(0, reg);
            inst->set_operand(1, op);
            op = reg;
            modified = true;
        }
    };

    switch (inst->opcode()) {
        case InstStore: {
            std::cout << "Store !!!!!" << std::endl;
            auto val = inst->operand(1);
            // imm2reg(val);
            auto reg = getVRegAs(ctx, val);
            auto tmp = isOperandVRegOrISAReg(reg);
            std::cout << "imm2reg" << tmp << std::endl;
            auto new_inst = ctx.new_inst(InstLoadImm);
            new_inst->set_operand(0, reg);
            new_inst->set_operand(1, val);
            inst->set_operand(1, reg);
            modified = true;
            break;
        }
    }
    return modified;
}

bool RISCVISelInfo::match_select(MIRInst* inst, ISelContext& ctx) const {
    bool ret = legalizeInst(inst, ctx);
    return matchAndSelectImpl(inst, ctx);
}
}  // namespace mir::RISCV