#pragma once
#include "mir/allocator.hpp"

namespace mir {
static void fastAllocator(MIRFunction& mfunc, CodeGenContext& ctx) {
    struct VirtualRegisterInfo final {
        std::unordered_set<mir::MIRBlock*> _uses;
        std::unordered_set<mir::MIRBlock*> _defs;
    };

    for (auto& block : mfunc.blocks()) {
        for (auto& inst : block->insts()) {
            
        }
    }
}
}  // namespace mir