#include "mir/utils.hpp"

namespace mir {

void allocateStackObjects(MIRFunction& func, CodeGenContext& ctx) {
    /* callee saved */
    std::unordered_set<MIROperand*> overwrited;

    forEachDefOperand(func, ctx, [&](MIROperand* op) {
        if (op->is_reg() && isISAReg(op->reg()) &&
            ctx.frameInfo.is_callee_saved(*op)) {
            overwrited.insert(MIROperand::as_isareg(
                op->reg(),
                ctx.registerInfo->getCanonicalizedRegisterType(op->type())));
        }
    });

    /* callee arguments */

    /* local variables */

    /* emit prologue and epilogue */
}

}  // namespace mir
