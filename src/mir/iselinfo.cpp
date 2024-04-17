#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "mir/iselinfo.hpp"

namespace mir {
void ISelContext::run_isel(MIRFunction* func) {
    // TODO: implement
}

uint32_t select_copy_opcode(MIROperand* dst, MIROperand* src) {
    if (dst->is_reg() && isISAReg(dst->reg())) {
        // dst is a isa reg
        if (src->is_imm()) {
            return InstLoadImmToReg;
        }
        return InstCopyToReg;
    }
    if (src->is_imm()) {
        return InstLoadImmToReg;
    }
    if (src->is_reg() && isISAReg(src->reg())) {
        return InstCopyFromReg;
    }
    assert(isOperandVReg(src) and isOperandVReg(dst));
    return InstCopy;
}

}  // namespace mir
