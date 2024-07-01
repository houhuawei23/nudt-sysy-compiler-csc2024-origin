#include "mir/RegisterAllocator.hpp"
#include "mir/target.hpp"

namespace mir {
void IPRAUsageCache::add(const CodeGenContext& ctx, MIRRelocable* symbol, MIRFunction& mfunc) {
    IPRAInfo info;
    for (auto& block : mfunc.blocks()) {
        for (auto inst : block->insts()) {
            auto& instInfo = ctx.instInfo.get_instinfo(inst);
            
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                auto op = inst->operand(idx);
                if (!isOperandISAReg(op)) continue;  // 该操作数必须是物理寄存器
                if (ctx.frameInfo.is_caller_saved(*op)) info.emplace(op);
            }

            if (requireFlag(instInfo.inst_flag(), InstFlagCall)) {
                auto callee = inst->operand(0)->reloc();
                if (callee != symbol) {
                    auto calleeInfo = query(callee);
                    if (calleeInfo) {
                        for (auto reg : *calleeInfo) info.emplace(reg);
                    } else {
                        return;
                    }
                }
            }
        }
    }
    _cache.emplace(symbol, std::move(info));
}
void IPRAUsageCache::add(MIRRelocable* symbol, IPRAInfo info) {
    _cache.emplace(symbol, std::move(info));
}
const IPRAInfo* IPRAUsageCache::query(MIRRelocable* calleeFunc) const {
    if (auto iter = _cache.find(calleeFunc); iter != _cache.cend()) return &iter->second;
    return nullptr;
}
};