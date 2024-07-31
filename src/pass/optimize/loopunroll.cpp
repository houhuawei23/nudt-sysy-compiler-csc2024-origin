#include "pass/optimize/optimize.hpp"
#include "pass/optimize/loopunroll.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
namespace pass {
int loopUnroll::calunrolltime(ir::Loop* loop, int times) {
    int codecnt = 0;
    for (auto bb : loop->blocks()) {
        codecnt += bb->insts().size();
    }
    int unrolltimes = 1;
    for (int i = 2; i <= (int)sqrt(times); i++) {
        if (i * codecnt > 1000) break;
        if (times % i == 0) unrolltimes = i;
    }
    return unrolltimes;
}

void loopUnroll::dynamicunroll(ir::Loop* loop, ir::indVar* iv) {
    return;
}
void loopUnroll::constunroll(ir::Loop* loop, ir::indVar* iv) {
    if (loop->exits().size() != 1)  // 只对单exit的loop做unroll
        return;

    int ivbegin = iv->getBeginI32();
    int ivend = iv->endValue()->dynCast<ir::Constant>()->i32();
    int ivstep = iv->getStepI32();
    ir::BinaryInst* ivbinary = iv->iterInst();
    ir::Instruction* ivcmp = iv->cmpInst();

    if (!ivbinary->isInt32() || (ivstep == 0))  // 只考虑int循环,step为0退出
        return;

    int times = 0;
    if (ivbinary->valueId() == ir::vADD) {
        if (ivcmp->valueId() == ir::vIEQ) {
            times = (ivbegin == ivend) ? 1 : 0;
        } else if (ivcmp->valueId() == ir::vINE) {
            times = ((ivend - ivbegin) % ivstep) ? ((ivend - ivbegin) / ivstep) : -1;
        } else if (ivcmp->valueId() == ir::vISGE) {
            times = -1;
        } else if (ivcmp->valueId() == ir::vISLE) {
            times = (ivend - ivbegin) / ivstep + 1;
        } else if (ivcmp->valueId() == ir::vISGT) {
            times = -1;
        } else if (ivcmp->valueId() == ir::vISLT) {
            times = ((ivend - ivbegin) % ivstep == 0) ? ((ivend - ivbegin) / ivstep) : ((ivend - ivbegin) / ivstep + 1);
        } else {
            times = -1;
        }
    } else if (ivbinary->valueId() == ir::vSUB) {
        if (ivcmp->valueId() == ir::vIEQ) {
            times = (ivbegin == ivend) ? 1 : 0;
        } else if (ivcmp->valueId() == ir::vINE) {
            times = ((ivbegin - ivend) % ivstep) ? ((ivbegin - ivend) / ivstep) : -1;
        } else if (ivcmp->valueId() == ir::vISGE) {
            times = (ivbegin - ivend) / ivstep + 1;
        } else if (ivcmp->valueId() == ir::vISLE) {
            times = -1;
        } else if (ivcmp->valueId() == ir::vISGT) {
            times = ((ivbegin - ivend) % ivstep == 0) ? ((ivbegin - ivend) / ivstep) : ((ivbegin - ivend) / ivstep + 1);
        } else if (ivcmp->valueId() == ir::vISLT) {
            times = -1;
        } else {
            times = -1;
        }
    }
    if (times < 0) {
        return;
    }

    doconstunroll(loop, iv, times);
}

void loopUnroll::doconstunroll(ir::Loop* loop, ir::indVar* iv, int times) {
    ir::Function* func = loop->header()->function();
    int unrolltimes = calunrolltime(loop, times);
    ir::BasicBlock* head = loop->header();
    ir::BasicBlock* latch = loop->getLoopLatch();
    ir::BasicBlock* preheader = loop->getLoopPreheader();
    ir::BasicBlock* exit;
    ir::BasicBlock* headnext;
    for (auto bb : loop->exits())
        exit = bb;
    for (auto bb : head->next_blocks()) {
        if (loop->contains(bb)) headnext = bb;
    }

    std::unordered_map<ir::Value*, ir::Value*> reachDefBeforeHead;
    std::unordered_map<ir::Value*, ir::Value*> reachDefAfterLatch;
    std::unordered_map<ir::Value*, ir::Value*> beginToEnd;
    std::unordered_map<ir::Value*, ir::Value*> endToBegin;
    for (auto inst : head->insts()) {
        if (auto phiinst = inst->dynCast<ir::PhiInst>()) {
            reachDefBeforeHead[phiinst] = phiinst->getvalfromBB(preheader);
        } else
            break;
    }

    ir::BasicBlock* latchnext = func->newBlock();
    loop->blocks().insert(latchnext);
    for (auto inst : head->insts()) {
        if (auto phiinst = inst->dynCast<ir::PhiInst>()) {
            ir::PhiInst* newphiinst = utils::make<ir::PhiInst>(nullptr, phiinst->type());
            latchnext->emplace_first_inst(newphiinst);
            auto val = phiinst->getvalfromBB(latch);
            newphiinst->addIncoming(val, latch);
            reachDefAfterLatch[phiinst] = newphiinst;
            beginToEnd[val] = newphiinst;
            endToBegin[newphiinst] = val;
        } else
            break;
    }

    ir::BranchInst* jmplatchnext2head = utils::make<ir::BranchInst>(head, latchnext);
    latchnext->emplace_back_inst(jmplatchnext2head);
    ir::BasicBlock::delete_block_link(latch, head);
    ir::BasicBlock::block_link(latch, latchnext);
    ir::BasicBlock::block_link(latchnext, head);
    ir::BranchInst* latchbr = latch->insts().back()->dynCast<ir::BranchInst>();
    latchbr->replaceDest(head, latchnext);// head的phi未更新

    std::vector<ir::BasicBlock*> bbexcepthead;
    for (auto bb : loop->blocks()) {
        if (bb != head) bbexcepthead.push_back(bb);
    }

    ir::BasicBlock* oldbegin = headnext;
    ir::BasicBlock* oldlatchnext = latchnext;
    for (int i = 0; i < unrolltimes; i++) {
        copyloop(bbexcepthead, oldbegin, loop, func);
        ir::BasicBlock::delete_block_link(oldlatchnext,head);//考虑到headnext不可能有phi，不必考虑修改前驱导致的对phi的影响
        ir::BasicBlock::block_link(oldlatchnext, copymap[headnext]->dynCast<ir::BasicBlock>());
        ir::BasicBlock::block_link(copymap[oldlatchnext]->dynCast<ir::BasicBlock>(), head);
        auto newbegin = copymap[oldbegin]->dynCast<ir::BasicBlock>();
        auto newlatchnext = copymap[oldlatchnext]->dynCast<ir::BasicBlock>();
    }
}

void loopUnroll::copyloop(std::vector<ir::BasicBlock*> bbs, ir::BasicBlock* begin, ir::Loop* L, ir::Function* func) {//复制
    auto Module = func->module();
    auto getValue = [&](ir::Value* val) -> ir::Value* {
        if (auto c = dyn_cast<ir::Constant>(val)) return c;
        if (copymap.count(val)) return copymap[val];
        return val;
    };
    for (auto gvlaue : Module->globalVars()) {
        copymap[gvlaue] = gvlaue;
    }
    for (auto arg : func->args()) {
        copymap[arg] = arg;
    }
    for (auto bb : bbs) {
        auto copybb = func->newBlock();
        copymap[bb] = copybb;
    }
    for (auto bb : bbs) {
        auto copybb = copymap[bb]->dynCast<ir::BasicBlock>();
        for (auto pred : bb->pre_blocks()) {
            copybb->pre_blocks().emplace_back(copymap[pred]->dynCast<ir::BasicBlock>());
        }
        for (auto succ : bb->next_blocks()) {
            copybb->next_blocks().emplace_back(copymap[succ]->dynCast<ir::BasicBlock>());
        }
    }

    std::set<ir::BasicBlock*> vis;
    std::vector<ir::PhiInst*> phis;
    ir::BasicBlock::BasicBlockDfs(begin, [&](ir::BasicBlock* bb) -> bool {
        if (vis.count(bb)) return true;
        vis.insert(bb);
        auto copybb = copymap[bb]->dynCast<ir::BasicBlock>();
        for (auto inst : bb->insts()) {
            auto copyinst = inst->copy(getValue);
            copymap[inst] = copyinst;
            copybb->emplace_back_inst(copyinst);
            if (auto phiinst = inst->dynCast<ir::PhiInst>()) {
                phis.push_back(phiinst);
            }
        }
        return true;
    });
    for (auto phi : phis) {
        auto copyphi = copymap[phi]->dynCast<ir::PhiInst>();
        for (size_t i = 0; i < phi->getsize(); i++){
            auto phival = getValue(phi->getValue(i));
            auto phibb = getValue(phi->getBlock(i))->dynCast<ir::BasicBlock>();
            copyphi->addIncoming(phival, phibb);
        }
    }
}

bool loopUnroll::definuseout(ir::Instruction* inst, ir::Loop* L) {
    for (auto use : inst->uses()) {
        if (auto useinst = use->user()->dynCast<ir::Instruction>()) {
            auto useinstbb = useinst->block();
            if (auto phiinst = useinst->dynCast<ir::PhiInst>()) {
                for (size_t i = 0; i < phiinst->getsize(); i++) {
                    auto phival = phiinst->getValue(i);
                    auto phibb = phiinst->getBlock(i);
                    if (phival == inst) {
                        if (!L->contains(phibb)) return true;
                    }
                }
            } else {
                if (!L->contains(useinstbb)) return true;
            }
        }
    }
    return false;
}

bool loopUnroll::defoutusein(ir::Use* op, ir::Loop* L) {
    if (auto opinst = op->value()->dynCast<ir::Instruction>()) {
        auto opinstbb = opinst->block();
        if (!L->contains(opinstbb))  // 考虑head?
            return true;
    }
    return false;
}

bool loopUnroll::isconstant(ir::indVar* iv) {  // 判断迭代的end是否为常数
    if (auto constiv = iv->endValue()->dynCast<ir::Constant>()) return true;
    return false;
}
void loopUnroll::run(ir::Function* func, TopAnalysisInfoManager* tp) {
    lpctx = tp->getLoopInfo(func);
    ivctx = tp->getIndVarInfo(func);
    for (auto& loop : lpctx->loops()) {
        ir::indVar* iv = ivctx->getIndvar(loop);
        if (iv && isconstant(iv))
            constunroll(loop, iv);
        else
            dynamicunroll(loop, iv);
    }
}
}  // namespace pass