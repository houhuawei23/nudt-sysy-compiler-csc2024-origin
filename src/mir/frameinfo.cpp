#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "mir/frameinfo.hpp"
#include "mir/target.hpp"
#include <unordered_set>
#include <vector>

namespace mir {
int32_t TargetFrameInfo::insert_prologue_epilogue(
    MIRFunction* func,
    std::unordered_set<MIROperand*>& callee_save_regs,
    CodeGenContext& ctx,
    MIROperand* return_addr_reg) {
    // TODO: implement prologue and epilogue insertion
    std::vector<std::pair<MIROperand*, MIROperand*>> overwrited;

    /* allocate stack space for callee-saved registers */
    for (auto reg : callee_save_regs) {
        auto size = getOperandSize(
            ctx.registerInfo->getCanonicalizedRegisterType(reg->type()));
        auto align = size;
        auto storage = func->add_stack_obj(ctx.next_id(), size, align, 0,
                                           StackObjectUsage::CalleeSaved);
        overwrited.emplace_back(reg, storage);
    }

    /* insert prologue and epilogue code for callee-saved registers */
    for (auto& block : func->blocks()) {
        auto& insts = block->insts();
        /* backup all callee-saved registers */
        if(block == func->blocks().front()) {
            for (auto[reg, storage] : overwrited) {
                auto inst = new MIRInst{InstStoreRegToStack};
                inst->set_operand(0, reg);
                inst->set_operand(1, storage);
                insts.emplace_front(inst);
            }
        }
        /* restore all callee-saved registers (after return) */
        auto& last_inst = insts.back();
        auto& instinfo = ctx.instInfo.get_instinfo(last_inst);
        if(requireFlag(instinfo.inst_flag(), InstFlag::InstFlagReturn)){
            /* reverse order */
            //! Right or Not?
            for (auto it = overwrited.rbegin(); it!= overwrited.rend(); ++it) {
                auto[reg, storage] = *it;
                auto inst = new MIRInst{InstLoadRegFromStack};
                inst->set_operand(0, reg);
                inst->set_operand(1, storage);
                insts.emplace_back(inst);
            }
        }
    }

    return 0;
}
}  // namespace mir
