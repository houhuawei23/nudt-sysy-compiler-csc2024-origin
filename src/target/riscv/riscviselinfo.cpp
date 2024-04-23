#include "mir/mir.hpp"
#include "mir/iselinfo.hpp"

#include "target/riscv/InstInfoDecl.hpp"
#include "target/riscv/ISelInfoDecl.hpp"
#include "target/riscv/ISelInfoImpl.hpp"

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
