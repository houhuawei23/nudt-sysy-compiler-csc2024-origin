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
    bool debug = false;
    auto dumpInst = [&](MIRInst* inst) {
        auto& instInfo = ctx.codegen_ctx().instInfo.get_instinfo(inst);
        instInfo.print(std::cerr << "selectAddrOffset: ", *inst, false);
        std::cerr << std::endl;
    };
    // TODO: Select address offset for load/store instructions

    const auto addrInst = ctx.lookup_def(addr);
    if (addrInst) {
        if (debug)
            dumpInst(addrInst);
        if (addrInst->opcode() == InstLoadStackObjectAddr) {
            base = addrInst->operand(1);
            offset = MIROperand::as_imm(0, OperandType::Int64);
            return true;
        }
        if (addrInst->opcode() == InstLoadGlobalAddress) {
        }
    }
    if (isOperandIReg(addr)) {
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
            auto inst = ctx.new_inst(InstLoadImm);
            inst->set_operand(0, reg);
            inst->set_operand(1, op);
            op = reg;
            modified = true;
        }
    };
    auto imm2regBeta = [&](MIRInst* oldInst, uint32_t opIdx) {
        auto op = oldInst->operand(opIdx);
        if (op->is_imm()) {
            auto reg = getVRegAs(ctx, op);
            auto newInst = ctx.new_inst(InstLoadImm);
            newInst->set_operand(0, reg);
            newInst->set_operand(1, op);
            oldInst->set_operand(opIdx, reg);
            modified = true;
        }
    };
    switch (inst->opcode()) {
        case InstAdd:
        case InstAnd:
        case InstOr:
        case InstXor: {
            auto* src1 = inst->operand(1);
            auto* src2 = inst->operand(2);
            imm2regBeta(inst, 1);
            imm2regBeta(inst, 2);
            break;
        }
        case InstSub: { /* InstSub dst, src1, src2 */
            auto* src1 = inst->operand(1);
            auto* src2 = inst->operand(2);
            imm2regBeta(inst, 1);

            if (src2->is_imm()) { /* sub to add*/
                auto neg = getNeg(src2);
                if (isOperandImm12(neg)) {
                    inst->set_opcode(InstAdd);
                    inst->set_operand(2, neg);
                    modified = true;
                    break;
                }
            }
            imm2regBeta(inst, 2);
            break;
        }
        case InstNeg: {
            auto* val = inst->operand(1);
            /* sub dst, x0, src2 */
            inst->set_opcode(InstSub);
            inst->set_operand(1, getZero(val));
            inst->set_operand(2, val);
            break;
        }
        case InstAbs: {
            imm2regBeta(inst, 1);
            break;
        }
        case InstMul:
        case InstSDiv:
        case InstSRem:
        case InstUDiv:
        case InstURem: {
            imm2regBeta(inst, 1);
            imm2regBeta(inst, 2);
            break;
        }
        case InstICmp: {
            /* InstICmp dst, src1, src2, op */
            auto op = inst->operand(3);
            if (isICmpEqualityOp(op)) {
                /**
                 * ICmp dst, src1, src2, EQ/NE
                 * legalize ->
                 * xor newdst, src1, src2
                 * ICmp dst, newdst, 0, EQ/NE
                 * instSelect ->
                 * xor newdst, src1, src2
                 * sltiu dst, newdst, 1
                 */
                auto newDst = getVRegAs(ctx, inst->operand(0));
                auto newInst = ctx.new_inst(InstXor);
                newInst->set_operand(0, newDst);
                newInst->set_operand(1, inst->operand(1));
                newInst->set_operand(2, inst->operand(2));

                inst->set_operand(1, newDst); /* icmp */
                inst->set_operand(2, getZero(inst->operand(2)));
                // ctx.remove_inst(inst);
                modified = true;
            } else {
                imm2regBeta(inst, 1);
                imm2regBeta(inst, 2);
            }
            break;
        }
        case InstStore: { /* InstStore addr, src, align*/
            imm2regBeta(inst, 1);
            break;
        }
    }
    return modified;
}

bool RISCVISelInfo::match_select(MIRInst* inst, ISelContext& ctx) const {
    bool debugMatchSelect = false;
    bool ret = legalizeInst(inst, ctx);
    return matchAndSelectImpl(inst, ctx, debugMatchSelect);
}

static MIROperand* getAlign(int64_t immVal) {
    return MIROperand::as_imm(immVal, OperandType::Special);
}
/**
 * sw rs2, offset(rs1)
 * M[x[rs1] + sext(offset)] = x[rs2][31: 0]
 * lw rd, offset(rs1) or lw rd, rs1, offset
 * x[rd] = sext(M[x[rs1] + sext(offset)][31:0])
 */
void RISCVISelInfo::legalizeInstWithStackOperand(InstLegalizeContext& ctx,
                                                 MIROperand* op,
                                                 StackObject& obj) const {
    bool debugLISO = false;

    auto& inst = ctx.inst;
    auto& instInfo = ctx.codeGenCtx.instInfo.get_instinfo(inst);
    auto dumpInst = [&](MIRInst* inst) {
        instInfo.print(std::cerr, *inst, true);
        std::cerr << std::endl;
    };
    if (debugLISO) {
        instInfo.print(std::cerr, *inst, true);
    }
    int64_t immVal = obj.offset;
    switch (inst->opcode()) {
        case SD:
        case SW:
        case SH:
        case SB:
        case FSW: {
            immVal += inst->operand(1)->imm();
            break;
        }
        case LD:
        case LW:
        // case LWU:
        case LH:
        case LHU:
        case LB:
        case LBU:
        case FLW: {
            immVal += inst->operand(1)->imm();
            break;
        }
        default:
            break;
    }

    MIROperand* base = sp;
    auto offset = MIROperand::as_imm(immVal, OperandType::Int64);

    switch (inst->opcode()) {
        case InstLoadStackObjectAddr: {
            /**
             * addi rd, rs1, imm
             * x[rd] = x[rs1] + sext(imm)
             */
            if (debugLISO)
                std::cout << "addi rd, rs1, imm" << std::endl;

            inst->set_opcode(ADDI);
            inst->set_operand(1, base);
            inst->set_operand(2, offset);
            break;
        }
        case InstStoreRegToStack: {
            /**
             * StoreRegToStack obj[0 Metadata], src[1 Use]
             * sw rs2[0 Use], offset[1 Metadata](rs1[2 Use])
             * M[x[rs1] + sext(offset)] = x[rs2][31: 0]
             */
            if (debugLISO)
                std::cout << "sw rs2, offset(rs1)" << std::endl;
            inst->set_opcode(isOperandGR(*inst->operand(1)) ? SD : FSW);
            auto oldSrc = inst->operand(1);
            inst->set_operand(0, oldSrc); /* src2 := src */
            inst->set_operand(1, offset); /* offset */
            inst->set_operand(2, base);   /* base */
            inst->set_operand(
                3,
                getAlign(isOperandGR(*inst->operand(0)) ? 8 : 4)); /* align */

            break;
        }
        case InstLoadRegFromStack: {
            /**
             * LoadRegFromStack dst[0 Def], obj[1 Metadata]
             * lw rd, offset(rs1)
             */
            if (debugLISO)
                std::cout << "lw rd, offset(rs1)" << std::endl;

            inst->set_opcode(isOperandGR(*inst->operand(0)) ? LD : FLW);

            inst->set_operand(1, offset);
            inst->set_operand(2, base);
            inst->set_operand(3,
                              getAlign(isOperandGR(*inst->operand(0)) ? 8 : 4));
            break;
        }
        case SD:
        case SW:
        case SH:
        case SB:
        case FSW: {
            if (debugLISO)
                std::cout << "sw rs2, offset(rs1)" << std::endl;

            inst->set_operand(1, offset);
            inst->set_operand(2, base);
            break;
        }
        case LD:
        case LW:
        // case LWU:
        case LH:
        case LHU:
        case LB:
        case LBU:
        case FLW: {
            if (debugLISO)
                std::cout << "lw rd, offset(rs1)" << std::endl;
            //! careful with the order of operands,
            //! sw and lw have different order
            inst->set_operand(1, offset);
            inst->set_operand(2, base);
            break;
        }
        default:
            std::cerr
                << "Unsupported instruction for legalizeInstWithStackOperand"
                << std::endl;
    }
}

void RISCVISelInfo::postLegalizeInst(const InstLegalizeContext& ctx) const {
    auto& inst = ctx.inst;
    switch (inst->opcode()) {
        case InstCopy:
        case InstCopyFromReg:
        case InstCopyToReg: {
            const auto& dst = inst->operand(0);
            const auto& src = inst->operand(1);
            if (isOperandIReg(dst) && isOperandIReg(src)) {
                inst->set_opcode(MV);
            }
            // else if
            else {
                std::cerr << "Unsupported instruction for postLegalizeInst"
                          << std::endl;
            }
            break;
        }
        default:
            std::cerr << "Unsupported instruction for postLegalizeInst"
                      << std::endl;
    }
}
}  // namespace mir::RISCV