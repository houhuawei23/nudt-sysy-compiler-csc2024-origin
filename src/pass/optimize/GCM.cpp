#include "pass/optimize/GCM.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>

namespace pass {
// 找到两个bb的最近公共祖先
ir::BasicBlock* GCM::LCA(ir::BasicBlock* lhs, ir::BasicBlock* rhs) {
    if (!lhs) return rhs;
    if (!rhs) return lhs;
    // while (lhs->domLevel > rhs->domLevel)
    //     lhs = lhs->idom;
    // while (rhs->domLevel > lhs->domLevel)
    //     rhs = rhs->idom;
    // while (lhs != rhs)
    // {
    //     lhs = lhs->idom;
    //     rhs = rhs->idom;
    // }
    while (domctx->domlevel(lhs) > domctx->domlevel(rhs))
        lhs = domctx->idom(lhs);
    while (domctx->domlevel(rhs) < domctx->domlevel(lhs))
        rhs = domctx->idom(rhs);
    while (lhs != rhs) {
        lhs = domctx->idom(lhs);
        rhs = domctx->idom(rhs);
    }
    return lhs;
}

// 通过指令的类型判断该指令是否固定在bb上
bool GCM::ispinned(ir::Instruction* instruction) {
    if (dyn_cast<ir::BinaryInst>(instruction)) {
        // 二元运算指令不固定(除法固定，因为除法指令可能产生除零错误)
        if (instruction->valueId() == ir::vSDIV || instruction->valueId() == ir::vSREM) return true;
        return false;
    }

    else if (dyn_cast<ir::UnaryInst>(instruction))  // 一元运算指令不固定
        return false;
    // else if (dyn_cast<ir::CallInst>(instruction)) // call指令不固定
    //     return false;
    else  // 其他指令固定在自己的BB上
        return true;
}
// 提前调度:思想是如果把一个指令尽量往前提，那么应该在提之前将该指令参数来自的指令前提
void GCM::scheduleEarly(ir::Instruction* instruction, ir::BasicBlock* entry) {
    if (insts_visited.count(instruction))  // 如果已经访问过，则不进行提前调度
        return;

    insts_visited.insert(instruction);
    auto destBB = entry;  // 初始化放置块为entry块,整棵树的root
    ir::BasicBlock* opBB = nullptr;
    for (auto opiter = instruction->operands().begin(); opiter != instruction->operands().end();) {
        auto op = *opiter;
        opiter++;
        if (auto opInst = dyn_cast<ir::Instruction>(op->value())) {
            scheduleEarly(opInst, entry);
            opBB = opInst->block();
            if (domctx->domlevel(opBB) > domctx->domlevel(destBB)) {
                destBB = opBB;
            }
        }
    }
    if (!ispinned(instruction)) {
        auto instbb = instruction->block();
        if (destBB == instbb) return;

        auto bestBB = instbb;
        auto curBB = instbb;
        while (domctx->domlevel(curBB) > domctx->domlevel(destBB)) {
            if (lpctx->looplevel(curBB) < lpctx->looplevel(bestBB)) bestBB = curBB;
            curBB = domctx->idom(curBB);
        }

        if (bestBB == instbb) return;
        instbb->move_inst(instruction);                // 将指令从bb中移除
        bestBB->emplace_lastbutone_inst(instruction);  // 将指令移入destBB
    }
}

void GCM::run(ir::Function* F, TopAnalysisInfoManager* tp) {
    domctx = tp->getDomTree(F);
    domctx->refresh();
    lpctx = tp->getLoopInfo(F);
    lpctx->refresh();
    std::vector<ir::Instruction*> pininsts;
    insts_visited.clear();

    for (auto bb : F->blocks()) {
        for (auto institer = bb->insts().begin(); institer != bb->insts().end();) {
            auto inst = *institer;
            institer++;
            if (ispinned(inst)) pininsts.push_back(inst);
        }
    }

    for (auto inst : pininsts) {
        insts_visited.clear();
        for (auto opiter = inst->operands().begin(); opiter != inst->operands().end();) {
            auto op = *opiter;
            opiter++;
            if (ir::Instruction* opinst = dyn_cast<ir::Instruction>(op->value())) scheduleEarly(opinst, F->entry());
        }
    }
}
}  // namespace pass