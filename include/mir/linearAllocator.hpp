#pragma once
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/LiveInterval.hpp"

namespace mir {
static void linearAllocator(MIRFunction& mfunc, CodeGenContext& ctx) {
    auto liveInterval = calcLiveIntervals(mfunc, ctx);  // 计算变量活跃区间
    MultiClassRegisterSelector selector { *ctx.registerInfo };  // 寄存器选择
    auto inst2Num = liveInterval.inst2Num;  // instruction id
    auto reg2Interval = liveInterval.reg2Interval;  // register live interval

    for (auto& block : mfunc.blocks()) {
        for (auto& inst : block->insts()) {
            auto& instInfo = ctx.instInfo.get_instinfo(inst);

            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                auto op = inst->operand(idx);
                if (!isOperandVReg(op)) continue;
                auto virtualReg = op->reg();
                auto new_op = selector.getFreeRegister(op->type());
                *op = new_op;
                selector.markAsUsed(new_op);
            }
        }
    }
}
}