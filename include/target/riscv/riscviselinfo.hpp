#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/iselinfo.hpp"

namespace mir::RISCV {
static MIROperand* getVRegAs(ISelContext& ctx, MIROperand* ref) {
    return MIROperand::as_vreg(ctx.codegen_ctx().next_id(), ref->type());
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
        if (addrInst->opcode() == InstLoadStackObjectAddr) {
            base = addrInst->operand(1);
            offset = MIROperand::as_imm(0, OperandType::Int64);
            return true;
        }

    }

    return false;
}
}  // namespace mir::RISCV
