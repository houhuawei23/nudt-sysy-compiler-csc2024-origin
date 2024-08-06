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
std::unordered_map<ir::Value*, ir::Value*> loopUnroll::copymap;

int loopUnroll::calunrolltime(ir::Loop* loop, int times) {
    int codecnt = 0;
    for (auto bb : loop->blocks()) {
        codecnt += bb->insts().size();
    }
    int unrolltimes = 2;
    for (int i = 2; i <= (int)sqrt(times); i++) {
        if (i * codecnt > 1000) break;
        if (times % i == 0) unrolltimes = i;
    }
    return unrolltimes;
}

void loopUnroll::dynamicunroll(ir::Loop* loop, ir::indVar* iv) {
    // return ;
    if (loop->exits().size() != 1)  // 只对单exit的loop做unroll
        return;

    int ivbegin = iv->getBeginI32();
    ir::BinaryInst* ivbinary = iv->iterInst();
    ir::Instruction* ivcmp = iv->cmpInst();
    if (!ivbinary->isInt32())  // 只考虑int循环,step为0退出
        return;

    if (ivbinary->valueId() == ir::vADD) {
        if (ivcmp->valueId() == ir::vIEQ) {
            return;
        } else if (ivcmp->valueId() == ir::vINE) {
            return;
        } else if (ivcmp->valueId() == ir::vISGE) {
            return;
        } else if (ivcmp->valueId() == ir::vISLE) {
            ;
        } else if (ivcmp->valueId() == ir::vISGT) {
            return;
        } else if (ivcmp->valueId() == ir::vISLT) {
            ;
        } else {
            return;
        }
    } else if (ivbinary->valueId() == ir::vSUB) {
        if (ivcmp->valueId() == ir::vIEQ) {
            return;
        } else if (ivcmp->valueId() == ir::vINE) {
            return;
        } else if (ivcmp->valueId() == ir::vISGE) {
            ;
        } else if (ivcmp->valueId() == ir::vISLE) {
            return;
        } else if (ivcmp->valueId() == ir::vISGT) {
            ;
        } else if (ivcmp->valueId() == ir::vISLT) {
            return;
        } else {
            return;
        }
    } else {
        return;  // 不考虑其他运算
    }

    dodynamicunroll(loop, iv);
}

void loopUnroll::dodynamicunroll(ir::Loop* loop, ir::indVar* iv) {
    headuseouts.clear();
    ir::Function* func = loop->header()->function();
    int unrolltimes = 4;  // 可以修改的超参数
    // std::cerr << "dynamic unrolltimes: " << unrolltimes << std::endl;

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
    getdefinuseout(loop);
    insertremainderloop(loop, func);  // 插入尾循环
    // 修改迭代上限
    auto tailhead = getValue(head)->dynCast<ir::BasicBlock>();
    int ivbegin = iv->getBeginI32();
    ir::Value* ivend = iv->endValue();
    int ivstep = iv->getStepI32();
    ir::BinaryInst* ivbinary = iv->iterInst();
    ir::Instruction* ivcmp = iv->cmpInst();
    // TODO 计算end - begin

    ir::BinaryInst* distance;
    if (ivbinary->valueId() == ir::vADD) {
        if (auto ivendload = ivend->dynCast<ir::LoadInst>()) {
            auto loadglovalend = utils::make<ir::LoadInst>(ivendload->ptr(), ivend->type());
            if (ivcmp->valueId() == ir::vISLE)
                distance = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), loadglovalend, ir::ConstantInteger::gen_i32(iv->getBeginI32() - 1));
            else if (ivcmp->valueId() == ir::vISLT)
                distance = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), loadglovalend, iv->getBegin());

            auto isrem = utils::make<ir::BinaryInst>(ir::vSREM, ir::Type::TypeInt32(), distance, ir::ConstantInteger::gen_i32(unrolltimes * ivstep));
            auto isub = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), loadglovalend, isrem);

            auto icmp = utils::make<ir::ICmpInst>(ir::vISGE, distance, ir::ConstantInteger::gen_i32(unrolltimes * ivstep));
            auto newcondbr = utils::make<ir::BranchInst>(icmp, head, tailhead);
            auto oldbr = preheader->insts().back();
            preheader->delete_inst(oldbr);

            preheader->emplace_back_inst(loadglovalend);
            preheader->emplace_back_inst(distance);
            preheader->emplace_back_inst(isrem);
            preheader->emplace_back_inst(isub);
            preheader->emplace_back_inst(icmp);
            preheader->emplace_back_inst(newcondbr);
            ir::BasicBlock::block_link(preheader, tailhead);

            for (auto op : ivcmp->operands()) {
                if (op->value() == ivend) {
                    ivcmp->setOperand(op->index(), isub);
                    break;
                }
            }

            for (auto copyinst : tailhead->insts()) {
                if (auto copyphi = copyinst->dynCast<ir::PhiInst>()) {
                    auto originphi = copyphi->getvalfromBB(head)->dynCast<ir::PhiInst>();
                    auto originval = originphi->getvalfromBB(preheader);
                    copyphi->addIncoming(originval, preheader);
                } else {
                    break;
                }
            }
        } else {
            if (ivcmp->valueId() == ir::vISLE)
                distance = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), ivend, ir::ConstantInteger::gen_i32(iv->getBeginI32() - 1));
            else if (ivcmp->valueId() == ir::vISLT)
                distance = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), ivend, iv->getBegin());
            auto isrem = utils::make<ir::BinaryInst>(ir::vSREM, ir::Type::TypeInt32(), distance, ir::ConstantInteger::gen_i32(unrolltimes * ivstep));
            auto isub = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), ivend, isrem);

            auto icmp = utils::make<ir::ICmpInst>(ir::vISGE, distance, ir::ConstantInteger::gen_i32(unrolltimes * ivstep));
            auto newcondbr = utils::make<ir::BranchInst>(icmp, head, tailhead);
            auto oldbr = preheader->insts().back();
            preheader->delete_inst(oldbr);

            preheader->emplace_back_inst(distance);
            preheader->emplace_back_inst(isrem);
            preheader->emplace_back_inst(isub);
            preheader->emplace_back_inst(icmp);
            preheader->emplace_back_inst(newcondbr);
            ir::BasicBlock::block_link(preheader, tailhead);

            for (auto op : ivcmp->operands()) {
                if (op->value() == ivend) {
                    ivcmp->setOperand(op->index(), isub);
                    break;
                }
            }

            for (auto copyinst : tailhead->insts()) {
                if (auto copyphi = copyinst->dynCast<ir::PhiInst>()) {
                    auto originphi = copyphi->getvalfromBB(head)->dynCast<ir::PhiInst>();
                    auto originval = originphi->getvalfromBB(preheader);
                    copyphi->addIncoming(originval, preheader);
                } else {
                    break;
                }
            }
        }

    } else if (ivbinary->valueId() == ir::vSUB) {
        if (auto ivendload = ivend->dynCast<ir::LoadInst>()) {
            auto loadglovalend = utils::make<ir::LoadInst>(ivendload->ptr(), ivend->type());
            if (ivcmp->valueId() == ir::vISLE)
                distance = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), ir::ConstantInteger::gen_i32(iv->getBeginI32() + 1), loadglovalend);
            else if (ivcmp->valueId() == ir::vISLT)
                distance = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), iv->getBegin(), loadglovalend);

            auto isrem = utils::make<ir::BinaryInst>(ir::vSREM, ir::Type::TypeInt32(), distance, ir::ConstantInteger::gen_i32(unrolltimes * ivstep));
            auto iadd = utils::make<ir::BinaryInst>(ir::vADD, ir::Type::TypeInt32(), ivend, isrem);

            auto icmp = utils::make<ir::ICmpInst>(ir::vISGE, distance, ir::ConstantInteger::gen_i32(unrolltimes * ivstep));
            auto newcondbr = utils::make<ir::BranchInst>(icmp, head, tailhead);
            auto oldbr = preheader->insts().back();
            preheader->delete_inst(oldbr);

            preheader->emplace_back_inst(loadglovalend);
            preheader->emplace_back_inst(distance);
            preheader->emplace_back_inst(isrem);
            preheader->emplace_back_inst(iadd);
            preheader->emplace_back_inst(icmp);
            preheader->emplace_back_inst(newcondbr);
            ir::BasicBlock::block_link(preheader, tailhead);

            for (auto op : ivcmp->operands()) {
                if (op->value() == ivend) {
                    ivcmp->setOperand(op->index(), iadd);
                    break;
                }
            }

            for (auto copyinst : tailhead->insts()) {
                if (auto copyphi = copyinst->dynCast<ir::PhiInst>()) {
                    auto originphi = copyphi->getvalfromBB(head)->dynCast<ir::PhiInst>();
                    auto originval = originphi->getvalfromBB(preheader);
                    copyphi->addIncoming(originval, preheader);
                } else {
                    break;
                }
            }
        } else {
            if (ivcmp->valueId() == ir::vISLE)
                distance = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), ir::ConstantInteger::gen_i32(iv->getBeginI32() + 1), ivend);
            else if (ivcmp->valueId() == ir::vISLT)
                distance = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), iv->getBegin(), ivend);

            auto isrem = utils::make<ir::BinaryInst>(ir::vSREM, ir::Type::TypeInt32(), distance, ir::ConstantInteger::gen_i32(unrolltimes * ivstep));
            auto iadd = utils::make<ir::BinaryInst>(ir::vADD, ir::Type::TypeInt32(), ivend, isrem);

            auto icmp = utils::make<ir::ICmpInst>(ir::vISGE, distance, ir::ConstantInteger::gen_i32(unrolltimes * ivstep));
            auto newcondbr = utils::make<ir::BranchInst>(icmp, head, tailhead);
            auto oldbr = preheader->insts().back();
            preheader->delete_inst(oldbr);

            preheader->emplace_back_inst(distance);
            preheader->emplace_back_inst(isrem);
            preheader->emplace_back_inst(iadd);
            preheader->emplace_back_inst(icmp);
            preheader->emplace_back_inst(newcondbr);

            for (auto op : ivcmp->operands()) {
                if (op->value() == ivend) {
                    ivcmp->setOperand(op->index(), iadd);
                    break;
                }
            }

            for (auto copyinst : tailhead->insts()) {
                if (auto copyphi = copyinst->dynCast<ir::PhiInst>()) {
                    auto originphi = copyphi->getvalfromBB(head)->dynCast<ir::PhiInst>();
                    auto originval = originphi->getvalfromBB(preheader);
                    copyphi->addIncoming(originval, preheader);
                } else {
                    break;
                }
            }
        }
    }

    std::vector<std::vector<ir::Value*>> phireplacevec;
    ir::BasicBlock* latchnext = func->newBlock();
    nowlatchnext = latchnext;
    loop->blocks().insert(latchnext);

    int cnt = 0;

    for (auto inst : head->insts()) {
        if (auto phiinst = inst->dynCast<ir::PhiInst>()) {
            ir::PhiInst* newphiinst = utils::make<ir::PhiInst>(nullptr, phiinst->type());
            latchnext->emplace_back_inst(newphiinst);  // 保证映射正确
            auto val = phiinst->getvalfromBB(latch);
            newphiinst->addIncoming(val, latch);
            phireplacevec.push_back({phiinst, newphiinst});
        } else
            break;
    }

    ir::BranchInst* jmplatchnext2head = utils::make<ir::BranchInst>(head, latchnext);
    latchnext->emplace_back_inst(jmplatchnext2head);
    ir::BasicBlock::delete_block_link(latch, head);
    ir::BasicBlock::block_link(latch, latchnext);
    ir::BasicBlock::block_link(latchnext, head);
    ir::BranchInst* latchbr = latch->insts().back()->dynCast<ir::BranchInst>();
    latchbr->replaceDest(head, latchnext);  // head的phi未更新

    std::vector<ir::BasicBlock*> bbexcepthead;
    for (auto bb : loop->blocks()) {
        if (bb != head) bbexcepthead.push_back(bb);
    }

    ir::BasicBlock* oldbegin = headnext;
    ir::BasicBlock* oldlatchnext = latchnext;

    for (int i = 0; i < unrolltimes - 1; i++) {  // 复制循环体
        copymap.clear();
        copyloop(bbexcepthead, oldbegin, loop, func);

        auto newbegin = copymap[oldbegin]->dynCast<ir::BasicBlock>();
        auto newlatchnext = copymap[oldlatchnext]->dynCast<ir::BasicBlock>();
        nowlatchnext = newlatchnext;

        ir::BranchInst* oldlatchnextbr = oldlatchnext->insts().back()->dynCast<ir::BranchInst>();
        oldlatchnextbr->replaceDest(head, newbegin);
        ir::BasicBlock::delete_block_link(oldlatchnext, head);  // 考虑到headnext不可能有phi，不必考虑修改前驱导致的对phi的影响
        ir::BasicBlock::block_link(oldlatchnext, newbegin);
        ir::BasicBlock::block_link(newlatchnext, head);

        oldbegin = newbegin;
        oldlatchnext = newlatchnext;

        // //更新bbexcepthead
        for (auto bb : bbexcepthead) {
            auto copybb = copymap[bb]->dynCast<ir::BasicBlock>();
            for (auto inst : copybb->insts()) {
                for (auto op : inst->operands()) {
                    for (auto vec : phireplacevec) {
                        if (std::find(vec.begin(), vec.end(), op->value()) != vec.end()) {
                            auto newval = vec.back();
                            inst->setOperand(op->index(), newval);
                        }
                    }
                }
            }
        }

        cnt = 0;
        for (auto inst : oldlatchnext->insts()) {
            if (auto oldphiinst = inst->dynCast<ir::PhiInst>()) {
                phireplacevec[cnt].push_back(oldphiinst);
                cnt++;
            } else
                break;
        }

        std::vector<ir::BasicBlock*> newbbexcepthead;
        for (auto bb : bbexcepthead) {
            newbbexcepthead.push_back(copymap[bb]->dynCast<ir::BasicBlock>());
        }
        bbexcepthead = newbbexcepthead;
    }
    // 修改head的phi
    cnt = 0;
    for (auto inst : head->insts()) {
        if (auto phiinst = inst->dynCast<ir::PhiInst>()) {
            auto newval = phireplacevec[cnt].back();
            cnt++;
            for (size_t i = 0; i < phiinst->getsize(); i++) {
                auto phibb = phiinst->getBlock(i);
                if (phibb == latch) {
                    phiinst->delBlock(phibb);
                    phiinst->addIncoming(newval, oldlatchnext);
                }
            }
        } else
            break;
    }

    // 修改迭代变量为 iv = iv + unrolltimes

    auto ivphi = iv->phiinst();
    if (iv->iterInst()->valueId() == ir::vADD) {
        auto iadd = utils::make<ir::BinaryInst>(ir::vADD, ir::Type::TypeInt32(), ivphi, ir::ConstantInteger::gen_i32(unrolltimes * (iv->getStepI32())));
        nowlatchnext->emplace_lastbutone_inst(iadd);
        for (size_t i = 0; i < ivphi->getsize(); i++) {
            auto phibb = ivphi->getBlock(i);
            if (phibb == nowlatchnext) {
                ivphi->setOperand(2 * i, iadd);
            }
        }
    } else if (iv->iterInst()->valueId() == ir::vSUB) {
        auto isub = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), ivphi, ir::ConstantInteger::gen_i32(unrolltimes * (iv->getStepI32())));
        nowlatchnext->emplace_lastbutone_inst(isub);
        for (size_t i = 0; i < ivphi->getsize(); i++) {
            auto phibb = ivphi->getBlock(i);
            if (phibb == nowlatchnext) {
                ivphi->setOperand(2 * i, isub);
            }
        }
    }
}
void loopUnroll::constunroll(ir::Loop* loop, ir::indVar* iv) {
    if (loop->exits().size() != 1)  // 只对单exit的loop做unroll
        return;

    int ivbegin = iv->getBeginI32();
    int ivend = iv->endValue()->dynCast<ir::ConstantValue>()->i32();
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
    } else {
        times = -1;  // 不考虑其他运算
    }
    if (times <= 0) {
        return;
    }

    doconstunroll(loop, iv, times);
}

void loopUnroll::insertremainderloop(ir::Loop* loop, ir::Function* func) {
    copymap.clear();
    ir::BasicBlock* head = loop->header();
    ir::BasicBlock* preheader = loop->getLoopPreheader();
    ir::BasicBlock* exit;
    for (auto bb : loop->exits())
        exit = bb;

    std::vector<ir::BasicBlock*> bbs;
    for (auto bb : loop->blocks()) {
        bbs.push_back(bb);
    }
    copyloopremainder(bbs, head, loop, func);
    auto copyhead = getValue(head)->dynCast<ir::BasicBlock>();
    copyhead->pre_blocks().remove(preheader);
    copyhead->next_blocks().remove(exit);
    ir::BranchInst* headbr = head->insts().back()->dynCast<ir::BranchInst>();
    headbr->replaceDest(exit, copyhead);
    ir::BasicBlock::delete_block_link(head, exit);
    ir::BasicBlock::block_link(head, copyhead);
    ir::BasicBlock::block_link(copyhead, exit);
    // 替换remainderloop的headphi
    for (auto inst : head->insts()) {
        if (auto phiinst = inst->dynCast<ir::PhiInst>()) {
            auto copyphiinst = getValue(phiinst)->dynCast<ir::PhiInst>();
            copyphiinst->delBlock(preheader);
            copyphiinst->addIncoming(phiinst, head);
        } else {
            break;
        }
    }

    for (auto inst : headuseouts) {
        if (getValue(inst) != inst) {
            repalceuseout(inst, getValue(inst)->dynCast<ir::Instruction>(), loop);
            // std::cerr<<"replace useout: "<<std::endl;
        }
    }
}

void loopUnroll::doconstunroll(ir::Loop* loop, ir::indVar* iv, int times) {
    headuseouts.clear();
    ir::Function* func = loop->header()->function();
    int unrolltimes = calunrolltime(loop, times);
    // unrolltimes = 2;//debug
    int remainder = times % unrolltimes;
    // std::cerr << "times: " << times << std::endl;
    // std::cerr << "unrolltimes: " << unrolltimes << std::endl;
    // std::cerr << "remainder: " << remainder << std::endl;

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
    getdefinuseout(loop);

    if (remainder != 0) {
        insertremainderloop(loop, func);  // 插入尾循环
        // 修改迭代上限
        int ivbegin = iv->getBeginI32();
        ir::Value* ivend = iv->endValue();  // 常数
        int ivstep = iv->getStepI32();
        ir::BinaryInst* ivbinary = iv->iterInst();
        ir::Instruction* ivcmp = iv->cmpInst();
        if (ivbinary->valueId() == ir::vADD) {
            auto isub = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), ivend, ir::ConstantInteger::gen_i32(remainder * ivstep));
            preheader->emplace_lastbutone_inst(isub);
            for (auto op : ivcmp->operands()) {
                if (op->value() == ivend) {
                    ivcmp->setOperand(op->index(), isub);
                    break;
                }
            }
        } else if (ivbinary->valueId() == ir::vSUB) {
            auto iadd = utils::make<ir::BinaryInst>(ir::vADD, ir::Type::TypeInt32(), ivend, ir::ConstantInteger::gen_i32(remainder * ivstep));
            preheader->emplace_lastbutone_inst(iadd);
            for (auto op : ivcmp->operands()) {
                if (op->value() == ivend) {
                    ivcmp->setOperand(op->index(), iadd);
                    break;
                }
            }
        }
    }

    std::vector<std::vector<ir::Value*>> phireplacevec;
    ir::BasicBlock* latchnext = func->newBlock();
    nowlatchnext = latchnext;
    loop->blocks().insert(latchnext);

    int cnt = 0;

    for (auto inst : head->insts()) {
        if (auto phiinst = inst->dynCast<ir::PhiInst>()) {
            ir::PhiInst* newphiinst = utils::make<ir::PhiInst>(nullptr, phiinst->type());
            latchnext->emplace_back_inst(newphiinst);  // 保证映射正确
            auto val = phiinst->getvalfromBB(latch);
            newphiinst->addIncoming(val, latch);
            phireplacevec.push_back({phiinst, newphiinst});
        } else
            break;
    }

    ir::BranchInst* jmplatchnext2head = utils::make<ir::BranchInst>(head, latchnext);
    latchnext->emplace_back_inst(jmplatchnext2head);
    ir::BasicBlock::delete_block_link(latch, head);
    ir::BasicBlock::block_link(latch, latchnext);
    ir::BasicBlock::block_link(latchnext, head);
    ir::BranchInst* latchbr = latch->insts().back()->dynCast<ir::BranchInst>();
    latchbr->replaceDest(head, latchnext);  // head的phi未更新

    std::vector<ir::BasicBlock*> bbexcepthead;
    for (auto bb : loop->blocks()) {
        if (bb != head) bbexcepthead.push_back(bb);
    }

    ir::BasicBlock* oldbegin = headnext;
    ir::BasicBlock* oldlatchnext = latchnext;

    for (int i = 0; i < unrolltimes - 1; i++) {  // 复制循环体
        copymap.clear();
        copyloop(bbexcepthead, oldbegin, loop, func);

        auto newbegin = copymap[oldbegin]->dynCast<ir::BasicBlock>();
        auto newlatchnext = copymap[oldlatchnext]->dynCast<ir::BasicBlock>();
        nowlatchnext = newlatchnext;

        ir::BranchInst* oldlatchnextbr = oldlatchnext->insts().back()->dynCast<ir::BranchInst>();
        oldlatchnextbr->replaceDest(head, newbegin);
        ir::BasicBlock::delete_block_link(oldlatchnext, head);  // 考虑到headnext不可能有phi，不必考虑修改前驱导致的对phi的影响
        ir::BasicBlock::block_link(oldlatchnext, newbegin);
        ir::BasicBlock::block_link(newlatchnext, head);

        oldbegin = newbegin;
        oldlatchnext = newlatchnext;

        // //更新bbexcepthead
        for (auto bb : bbexcepthead) {
            auto copybb = copymap[bb]->dynCast<ir::BasicBlock>();
            for (auto inst : copybb->insts()) {
                for (auto op : inst->operands()) {
                    for (auto vec : phireplacevec) {
                        if (std::find(vec.begin(), vec.end(), op->value()) != vec.end()) {
                            auto newval = vec.back();
                            inst->setOperand(op->index(), newval);
                        }
                    }
                }
            }
        }

        cnt = 0;
        for (auto inst : oldlatchnext->insts()) {
            if (auto oldphiinst = inst->dynCast<ir::PhiInst>()) {
                phireplacevec[cnt].push_back(oldphiinst);
                cnt++;
            } else
                break;
        }

        std::vector<ir::BasicBlock*> newbbexcepthead;
        for (auto bb : bbexcepthead) {
            newbbexcepthead.push_back(copymap[bb]->dynCast<ir::BasicBlock>());
        }
        bbexcepthead = newbbexcepthead;
    }
    // 修改head的phi
    cnt = 0;
    for (auto inst : head->insts()) {
        if (auto phiinst = inst->dynCast<ir::PhiInst>()) {
            auto newval = phireplacevec[cnt].back();
            cnt++;
            for (size_t i = 0; i < phiinst->getsize(); i++) {
                auto phibb = phiinst->getBlock(i);
                if (phibb == latch) {
                    phiinst->delBlock(phibb);
                    phiinst->addIncoming(newval, oldlatchnext);
                }
            }
        } else
            break;
    }
    // 修改迭代变量为 iv = iv + unrolltimes

    auto ivphi = iv->phiinst();
    if (iv->iterInst()->valueId() == ir::vADD) {
        auto iadd = utils::make<ir::BinaryInst>(ir::vADD, ir::Type::TypeInt32(), ivphi, ir::ConstantInteger::gen_i32(unrolltimes * (iv->getStepI32())));
        nowlatchnext->emplace_lastbutone_inst(iadd);
        for (size_t i = 0; i < ivphi->getsize(); i++) {
            auto phibb = ivphi->getBlock(i);
            if (phibb == nowlatchnext) {
                ivphi->setOperand(2 * i, iadd);
            }
        }
    } else if (iv->iterInst()->valueId() == ir::vSUB) {
        auto isub = utils::make<ir::BinaryInst>(ir::vSUB, ir::Type::TypeInt32(), ivphi, ir::ConstantInteger::gen_i32(unrolltimes * (iv->getStepI32())));
        nowlatchnext->emplace_lastbutone_inst(isub);
        for (size_t i = 0; i < ivphi->getsize(); i++) {
            auto phibb = ivphi->getBlock(i);
            if (phibb == nowlatchnext) {
                ivphi->setOperand(2 * i, isub);
            }
        }
    }
}

void loopUnroll::copyloop(std::vector<ir::BasicBlock*> bbs, ir::BasicBlock* begin, ir::Loop* L, ir::Function* func) {  // 复制循环体
    std::vector<ir::BasicBlock*> copybbs;
    auto Module = func->module();
    // auto getValue = [&](ir::Value* val) -> ir::Value* {
    //     if (auto c = dyn_cast<ir::ConstantValue>(val)) return c;
    //     if (copymap.count(val)) return copymap[val];
    //     return val;
    // };
    for (auto gvlaue : Module->globalVars()) {
        copymap[gvlaue] = gvlaue;
    }
    for (auto arg : func->args()) {
        copymap[arg] = arg;
    }
    for (auto bb : bbs) {
        auto copybb = func->newBlock();
        copymap[bb] = copybb;
        copybbs.push_back(copybb);
    }
    for (auto bb : bbs) {
        auto copybb = copymap[bb]->dynCast<ir::BasicBlock>();
        for (auto pred : bb->pre_blocks()) {
            if (pred != getValue(pred)) copybb->pre_blocks().emplace_back(getValue(pred)->dynCast<ir::BasicBlock>());
        }
        for (auto succ : bb->next_blocks()) {
            if (succ != getValue(succ)) copybb->next_blocks().emplace_back(getValue(succ)->dynCast<ir::BasicBlock>());
        }
    }

    std::set<ir::BasicBlock*> vis;
    std::vector<ir::PhiInst*> phis;
    ir::BasicBlock::BasicBlockDfs(begin, [&](ir::BasicBlock* bb) -> bool {
        if (vis.count(bb) || (std::count(bbs.begin(), bbs.end(), bb) == 0)) return true;
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
        return false;
    });
    for (auto phi : phis) {
        auto copyphi = copymap[phi]->dynCast<ir::PhiInst>();
        for (size_t i = 0; i < phi->getsize(); i++) {
            auto phival = getValue(phi->getValue(i));
            auto phibb = getValue(phi->getBlock(i))->dynCast<ir::BasicBlock>();
            copyphi->addIncoming(phival, phibb);
        }
    }
}

void loopUnroll::copyloopremainder(std::vector<ir::BasicBlock*> bbs, ir::BasicBlock* begin, ir::Loop* L, ir::Function* func) {  // 复制余数循环
    std::vector<ir::BasicBlock*> copybbs;
    auto Module = func->module();
    // auto getValue = [&](ir::Value* val) -> ir::Value* {
    //     if (auto c = dyn_cast<ir::ConstantValue>(val)) return c;
    //     if (copymap.count(val)) return copymap[val];
    //     return val;
    // };
    for (auto gvlaue : Module->globalVars()) {
        copymap[gvlaue] = gvlaue;
    }
    for (auto arg : func->args()) {
        copymap[arg] = arg;
    }
    for (auto bb : bbs) {
        auto copybb = func->newBlock();
        copymap[bb] = copybb;
        copybbs.push_back(copybb);
    }
    for (auto bb : bbs) {
        auto copybb = copymap[bb]->dynCast<ir::BasicBlock>();
        for (auto pred : bb->pre_blocks()) {
            copybb->pre_blocks().emplace_back(getValue(pred)->dynCast<ir::BasicBlock>());
        }
        for (auto succ : bb->next_blocks()) {
            copybb->next_blocks().emplace_back(getValue(succ)->dynCast<ir::BasicBlock>());
        }
    }

    std::set<ir::BasicBlock*> vis;
    std::vector<ir::PhiInst*> phis;
    ir::BasicBlock::BasicBlockDfs(begin, [&](ir::BasicBlock* bb) -> bool {
        if (vis.count(bb) || (std::count(bbs.begin(), bbs.end(), bb) == 0)) return true;
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
        return false;
    });
    for (auto phi : phis) {
        auto copyphi = copymap[phi]->dynCast<ir::PhiInst>();
        for (size_t i = 0; i < phi->getsize(); i++) {
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

void loopUnroll::getdefinuseout(ir::Loop* L) {
    auto head = L->header();
    for (auto inst : head->insts()) {
        if (definuseout(inst, L)) {
            headuseouts.push_back(inst);
        }
    }
}

void loopUnroll::repalceuseout(ir::Instruction* inst, ir::Instruction* copyinst, ir::Loop* L) {
    std::vector<ir::Use*> usetoreplace;
    for (auto use : inst->uses()) {
        if (auto useinst = use->user()->dynCast<ir::Instruction>()) {
            auto useinstbb = useinst->block();
            if (auto phiinst = useinst->dynCast<ir::PhiInst>()) {
                for (size_t i = 0; i < phiinst->getsize(); i++) {
                    auto phival = phiinst->getValue(i);
                    auto phibb = phiinst->getBlock(i);
                    if (phival == inst) {
                        if (!L->contains(phibb)) {
                            usetoreplace.push_back(use);
                        }
                    }
                }
            } else {
                if (!L->contains(useinstbb)) usetoreplace.push_back(use);
            }
        }
    }
    for (auto use : usetoreplace) {
        auto useinst = use->user()->dynCast<ir::Instruction>();
        useinst->setOperand(use->index(), copyinst);
    }
}

bool loopUnroll::isconstant(ir::indVar* iv) {  // 判断迭代的end是否为常数
    if (auto constiv = iv->endValue()->dynCast<ir::ConstantValue>()) return true;
    return false;
}
void loopUnroll::run(ir::Function* func, TopAnalysisInfoManager* tp) {
    lpctx = tp->getLoopInfo(func);
    ivctx = tp->getIndVarInfo(func);
    for (auto& loop : lpctx->loops()) {
        ir::indVar* iv = ivctx->getIndvar(loop);
        if (loop->subLoops().empty() && iv) {
            if (isconstant(iv))
                constunroll(loop, iv);
            else
                dynamicunroll(loop, iv);
        }
    }
}
}  // namespace pass