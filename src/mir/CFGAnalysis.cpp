#include "mir/CFGAnalysis.hpp"

namespace mir {
CFGAnalysis calcCFG(MIRFunction& mfunc, CodeGenContext& ctx) {
    assert(ctx.flags.endsWithTerminator);  // 确保该函数以终止指令结束
    CFGAnalysis res;
    auto& CFGInfo = res.block2CFGInfo();
    auto& blocks = mfunc.blocks();

    auto connect = [&](MIRBlock* src, MIRBlock* dst, double prob) {
        CFGInfo[src].successors.push_back({dst, prob});
        CFGInfo[dst].predecessors.push_back({src, prob});
    };

    for (auto it = blocks.begin(); it != blocks.end(); it++) {
        auto& block = *it;
        auto next = std::next(it);
        auto terminator = block->insts().back();
        
        MIRBlock* targetBlock; double prob;
        if (ctx.instInfo.matchBranch(terminator, targetBlock, prob)) {  // Match Jump Branch
            if (requireFlag(ctx.instInfo.get_instinfo(terminator).inst_flag(), InstFlagNoFallThrough)) {  // unconditional branch
                connect(block.get(), targetBlock, 1.0);
            } else {  // conditional
                if (next != blocks.end()) {  // 非exit块 
                    if (next->get() == targetBlock) {
                        connect(block.get(),  targetBlock, 1.0);
                    } else {
                        connect(block.get(), targetBlock, prob);
                        connect(block.get(), next->get(), 1.0 - prob);
                    }
                } else {  // exit块
                    connect(block.get(), targetBlock, prob);
                }
            }
        } else if (requireFlag(ctx.instInfo.get_instinfo(terminator).inst_flag(), InstFlagIndirectJump)) {  // jump register
            // TODO: jump the register
        }
    }

    return res;
}
}