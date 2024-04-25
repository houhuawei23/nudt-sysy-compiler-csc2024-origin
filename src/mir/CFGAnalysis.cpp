#include "mir/CFGAnalysis.hpp"

namespace mir {
CFGAnalysis calcCFG(MIRFunction& mfunc, CodeGenContext& ctx) {
    assert(ctx.flags.endsWithTerminator);  // 确保每一个块以终止指令结束

    CFGAnalysis res;
    auto& mInfo = res.mInfo();
    auto& blocks = mfunc.blocks();

    auto connect = [&](MIRBlock* src, MIRBlock* dst, double prob) {
        mInfo[src].successors.push_back({dst, prob});
        mInfo[dst].predecessors.push_back({src, prob});
    };

    for (auto it = blocks.begin(); it != blocks.end(); it++) {
        auto& block = *it;
        auto next = std::next(it);
        auto terminator = block->insts().back();
        
        MIRBlock* targetBlock; double prob;
        ctx.instInfo.get_instinfo(terminator);
        if (true) {  // Match Jump Branch
            if (true) {  // unconditional branch
                connect(block.get(), targetBlock, 1.0);
            } else {  // conditional
                if (next != blocks.end()) {
                    if (next->get() == targetBlock) {
                        connect(block.get(),  targetBlock, 1.0);
                    } else {
                        connect(block.get(), targetBlock, prob);
                        connect(block.get(), next->get(), 1.0 - prob);
                    }
                } else {
                    connect(block.get(), targetBlock, prob);
                }
            }
        } else if (true) {

        }
    }

    return res;
}
}