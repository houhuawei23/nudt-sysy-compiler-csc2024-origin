#include "mir/utils.hpp"

namespace mir {
struct StackObjectInterval final {
    uint32_t begin, end;
};
using Intervals = std::unordered_map<MIROperand*, StackObjectInterval*, MIROperandHasher>;

struct Slot final {
    uint32_t color;
    uint32_t end;

    bool operator<(const Slot& rhs) const noexcept { return end > rhs.end; }
};

static void removeUnusedSpillStackObjects(MIRFunction* mfunc) {
    std::unordered_set<MIROperand*, MIROperandHasher> stackObjects;
    for (auto& [ref, stack] : mfunc->stack_objs()) {
        // assert(isStackObject(ref.reg()));
    }
}

void allocateStackObjects(MIRFunction* func, CodeGenContext& ctx) {
    /* callee saved */
    std::unordered_set<MIROperand*> callee_saved;
    for (auto& block : func->blocks()) {
        forEachDefOperand(*block, ctx, [&](MIROperand* op) {
            if (op->is_unused()) return;
            if (op->is_reg() && isISAReg(op->reg()) && ctx.frameInfo.is_callee_saved(*op)) {
                callee_saved.insert(MIROperand::as_isareg(op->reg(), ctx.registerInfo->getCanonicalizedRegisterType(op->type())));
            }
        });
    }
    const auto preAllocatedBase = ctx.frameInfo.insert_prologue_epilogue(func, callee_saved, ctx, ctx.registerInfo->get_return_address_register());
    removeUnusedSpillStackObjects(func);

    /* callee arguments */

    /* local variables */

    /* emit prologue and epilogue */
}

}  // namespace mir
