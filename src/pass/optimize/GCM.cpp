#include "pass/optimize/GCM.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>

namespace pass
{
    // 找到两个bb的最近公共祖先
    ir::BasicBlock *GCM::LCA(ir::BasicBlock *lhs, ir::BasicBlock *rhs)
    {
        if (!lhs)
            return rhs;
        if (!rhs)
            return lhs;
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
            lhs=domctx->idom(lhs);
        while (domctx->domlevel(rhs) < domctx->domlevel(lhs))
            rhs=domctx->idom(rhs);
        while (lhs != rhs)
        {
            lhs = domctx->idom(lhs);
            rhs = domctx->idom(rhs);
        }
        return lhs;
    }

    // 通过指令的类型判断该指令是否固定在bb上
    bool GCM::ispinned(ir::Instruction *instruction)
    {
        if (dyn_cast<ir::BinaryInst>(instruction)) // 二元运算指令不固定
            return false;
        else if (dyn_cast<ir::UnaryInst>(instruction)) // 一元运算指令不固定
            return false;
        else if (dyn_cast<ir::PhiInst>(instruction)) // phi指令不固定
            return false;
        else if (dyn_cast<ir::CallInst>(instruction)) // call指令不固定
            return false;
        else // 其他指令固定在自己的BB上
            return true;
    }
    // 提前调度:思想是如果把一个指令尽量往前提，那么应该在提之前将该指令参数来自的指令前提
    void GCM::scheduleEarly(ir::Instruction *instruction)
    {
        if (insts_visited.find(instruction) == insts_visited.end()) // 如果已经访问过，则不进行提前调度
            return;
        insts_visited.insert(instruction);

        ir::BasicBlock *destBB = instruction->block()->function()->entry(); // 初始化放置块为entry块,整棵树的root
        ir::BasicBlock *opBB;

        for (auto op : instruction->operands()) // 遍历使用这条指令的所有指令
        {
            if (dyn_cast<ir::Instruction>(op->value()))
            {
                auto opInst = dyn_cast<ir::Instruction>(op->value());
                scheduleEarly(opInst);
                opBB = opInst->block();
                // if (opBB->domLevel > destBB->domLevel)
                // {
                //     destBB = opBB;
                // }
                if (domctx->domlevel(opBB) > domctx->domlevel(destBB))
                {
                    destBB = opBB;
                }
            }
        }
        if (!ispinned(instruction))
        {
            instruction->block()->delete_inst(instruction); // 将指令从bb中移除
            destBB->emplace_back_inst(instruction);          // 将指令移入destBB
            Earlymap[instruction] = destBB;
        }
    }

    // 延迟调度
    void GCM::scheduleLate(ir::Instruction *instruction)
    {
        if (insts_visited.find(instruction) == insts_visited.end()) // 如果已经访问过，则不进行延迟调度
            return;
        insts_visited.insert(instruction);

        ir::BasicBlock *destBB = nullptr;
        for (auto use : instruction->uses()) // 遍历这条指令使用的所有指令
        {
            if (dyn_cast<ir::Instruction>(use->user()))
            {
                ir::Instruction *useInst = dyn_cast<ir::Instruction>(use->user());
                scheduleLate(useInst);
                if (ir::PhiInst *phiInst = dyn_cast<ir::PhiInst>(useInst))
                {
                    size_t phisize = phiInst->getsize();
                    for (size_t i = 0; i < phisize; i++)
                    {
                        if (ir::Instruction *v = dyn_cast<ir::Instruction>(phiInst->getValue(i)))
                        {
                            if (v == instruction)
                            {
                                destBB = LCA(destBB, dyn_cast<ir::BasicBlock>(phiInst->getBlock(i)));
                                break;
                            }
                        }
                    }
                }
                else
                {
                    destBB = LCA(destBB, useInst->block());
                }
            }
        }

        if (!destBB)
            return;
        if (!ispinned(instruction))
        {

            // TODO 计算循环深度，在earlyBB和lateBB之间找到循环深度最浅的块插入指令
            ir::BasicBlock *EarlyBB = Earlymap[instruction];
            ir::BasicBlock *LateBB = destBB;
            ir::BasicBlock *bestBB = LateBB;
            ir::BasicBlock *curBB = LateBB;
            // while (curBB->domLevel >= EarlyBB->domLevel)
            // {
            //     if (curBB->looplevel < bestBB->looplevel)
            //     {
            //         bestBB = curBB;
            //     }
            //     curBB = curBB->idom;
            // }
            while (domctx->domlevel(curBB) >= domctx->domlevel(EarlyBB))
            {
                if (lpctx->looplevel(curBB) < lpctx->looplevel(bestBB))
                {
                    bestBB = curBB;
                }
                curBB=domctx->idom(curBB);
            }
            instruction->block()->delete_inst(instruction); // 将指令从bb中移除
            bestBB->emplace_back_inst(instruction);          // 将指令移入bestBB
        }
    }

    void GCM::run(ir::Function *F,topAnalysisInfoManager* tp)
    {
        if(not F->entry())return ;
        domctx=tp->getDomTree(F);
        domctx->refresh();
        lpctx=tp->getLoopInfo(F);
        lpctx->refresh();
        std::vector<ir::Instruction *> pinned;
        for (ir::BasicBlock *bb : F->blocks())
        {
            for (ir::Instruction *inst : bb->insts())
            {
                if (ispinned(inst))
                {
                    pinned.push_back(inst);
                }
            }
        }

        insts_visited.clear();
        for (auto instruction : pinned)
        {
            scheduleEarly(instruction);
        }
        insts_visited.clear();
        for (auto instruction : pinned)
        {
            scheduleLate(instruction);
        }
    }
} // namespace