#pragma once
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/LiveInterval.hpp"

namespace mir {
static void linearAllocator(MIRFunction& mfunc, CodeGenContext& ctx) {
    MultiClassRegisterSelector selector { *ctx.registerInfo };  // 寄存器选择

    for (auto& block : mfunc.blocks()) {
        auto& instructions = block->insts();
        for (auto it = instructions.begin(); it != instructions.end(); it++) {
            auto inst = *it;
            auto& instInfo = ctx.instInfo.getInstInfo(inst);
        
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                auto op = inst->operand(idx);

                if (!isOperandVReg(op)) continue;
                auto new_op = selector.getFreeRegister(op.type());
                if (new_op.isUnused()) {
                    /* no free register available */
                    continue;
                } else {
                    selector.markAsUsed(new_op);
                    op = new_op;
                }
            }
        }
    }
}
}