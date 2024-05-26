#include "pass/optimize/optimize.hpp"
#include "pass/optimize/inline.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>
namespace pass {
void Inline::callinline(ir::CallInst* call) {
    ir::Function* callee = call->callee();
    ir::BasicBlock* nowBB = call->parent();
    ir::Function* caller = nowBB->parent();
    ir::BasicBlock* retBB = caller->new_block();
    ir::BasicBlock* calleeAllocaBB = callee->entry();
    ir::BasicBlock* callerAllocaBB = caller->entry();
    ir::Function* copyfunc = callee->copy_func(); 
    if (nowBB == caller->exit()) {
        caller->setExit(retBB);
    }

    // 将call之后的指令移动到retBB中
    bool findfalg = false;
    for (auto inst : nowBB->insts()) {
        if (findfalg) {
            inst->set_parent(retBB);
            retBB->emplace_back_inst(inst);
            nowBB->insts().remove(inst);
        } else {
            if (auto ci = dyn_cast<ir::CallInst>(inst)) {
                if (call == ci) {
                    findfalg = true;
                }
            }
        }
    }
    // 将nowBB的后继变为retBB的后继
    for (ir::BasicBlock* succBB : nowBB->next_blocks()) {
        ir::BasicBlock::delete_block_link(nowBB, succBB);
        ir::BasicBlock::block_link(retBB, succBB);
        for (auto phi : succBB->phi_insts()) {  // 修改phi指令中的nowBB为retBB
            ir::PhiInst* phiinst = dyn_cast<ir::PhiInst>(phi);
            for (size_t i = 0; i < phiinst->getsize(); i++) {
                ir::BasicBlock* phiBB = phiinst->getbb(i);
                if (phiBB == nowBB) {
                    phiinst->replaceBB(retBB, i);
                }
            }
        }
    }
    std::vector<std::pair<ir::ReturnInst*, ir::BasicBlock*>> retmap;
    // 被调用函数的返回块无条件跳转到retBB
    for (ir::BasicBlock* bb : copyfunc->blocks()) {
        bb->set_parent(caller);
        caller->blocks().emplace_back(bb);
        copyfunc->blocks().remove(bb);
        if (bb->next_blocks().empty()) {
            auto lastinst = bb->insts().back();
            if (auto ret = dyn_cast<ir::ReturnInst>(lastinst)) {
                retmap.push_back(std::make_pair(ret, bb));
            }
        }
    }
    // 如果函数的返回值不是void，则需要把call的使用全部替换为返回值的使用
    if ((!copyfunc->ret_type()->is_void()) && (!retmap.empty())) {
        if (retmap.size() == 1) {  // 如果只有一个返回值
            call->replace_all_use_with(retmap[0].first->return_value());
        } else {
            ir::PhiInst* newphi = new ir::PhiInst(retBB, call->type());
            retBB->emplace_first_inst(newphi);
            for (auto [ret, bb] : retmap) {
                newphi->addIncoming(ret->return_value(), bb);
            }
            call->replace_all_use_with(newphi);
        }
    }
    // 在copyfunc的retbb中插入无条件跳转指令到caller的retBB
    for (auto [ret, bb] : retmap) {
        ir::BasicBlock::block_link(bb, retBB);
        auto jmprettobb = new ir::BranchInst(retBB, bb);
        bb->delete_inst(ret);
        bb->emplace_back_inst(jmprettobb);
    }
    // 处理被调用函数的参数
    for (size_t i = 0; i < copyfunc->args().size(); i++) {
        auto realArg = call->operand(i);
        auto formalArg = copyfunc->arg_i(i);
        if (!formalArg->type()
                 ->is_pointer()) {  // 如果传递参数不是数组等指针，直接替换
            formalArg->replace_all_use_with(realArg);
        } else {  // 如果传递参数是数组等指针，
            std::vector<ir::StoreInst*> storetomove;
            for (auto use : formalArg->uses()) {
                if (ir::StoreInst* storeinst =
                        dyn_cast<ir::StoreInst>(use->user())) {
                    storetomove.push_back(storeinst);
                    auto allocainst = storeinst->ptr();
                    std::vector<ir::LoadInst*> loadtoremove;
                    for (auto allocause : allocainst->uses()) {
                        if (ir::LoadInst* loadinst =
                                dyn_cast<ir::LoadInst>(allocause->user())) {
                            loadtoremove.push_back(loadinst);
                        }
                    }
                    for (auto rmloadinst : loadtoremove) {
                        rmloadinst->replace_all_use_with(realArg);
                        auto loadBB = rmloadinst->parent();
                        loadBB->delete_inst(rmloadinst);
                    }
                }
            }
            for (auto rmstoreinst : storetomove) {
                auto storeBB = rmstoreinst->parent();
                storeBB->delete_inst(rmstoreinst);
            }
        }
    }
    // 删除caller中调用copyfunc的call指令
    nowBB->delete_inst(call);
    // 连接nowBB和calle的entry,在nowBB末尾插入无条件跳转指令到copyfunc的entry
    ir::BasicBlock::block_link(nowBB, calleeAllocaBB);
    auto jmpnowtoentry = new ir::BranchInst(calleeAllocaBB, nowBB);
    nowBB->emplace_back_inst(jmpnowtoentry);
    // 将callee的alloca提到caller
    for (auto inst : calleeAllocaBB->insts()) {
        if (auto allocainst = dyn_cast<ir::AllocaInst>(inst)) {
            allocainst->set_parent(callerAllocaBB);
            callerAllocaBB->emplace_first_inst(allocainst);
            calleeAllocaBB->insts().remove(allocainst);
        }
    }
}

std::vector<ir::CallInst*> Inline::getcall(ir::Module* module,
                                           ir::Function* function) {
    std::vector<ir::CallInst*> calllist;
    for (auto func : module->funcs()) {
        for (auto bb : func->blocks()) {
            for (auto inst : bb->insts()) {
                if (auto callinst = dyn_cast<ir::CallInst>(inst)) {
                    if (function == callinst->callee()) {
                        calllist.push_back(callinst);
                    }
                }
            }
        }
    }

    return calllist;
}

std::vector<ir::Function*> Inline::getinlineFunc(ir::Module* module) {
    std::vector<ir::Function*> functiontoremove;
    for (auto func : module->funcs()) {
        if (func->name() != "main" && !func->blocks().empty() &&
            !func->get_is_inline()) {  // TODO 分析哪些函数可以被内联优化展开
            functiontoremove.push_back(func);
        }
    }
    return functiontoremove;
}
void Inline::run(ir::Module* module) {
    std::vector<ir::Function*> functiontoremove = getinlineFunc(module);
    while (!functiontoremove.empty()) {  // 找到所有调用了可以被内联优化展开的函数的call指令
        auto func = functiontoremove.back();
        std::vector<ir::CallInst*> callList = getcall(module, func);
        for (auto call : callList) {
            callinline(call);
        }
        module->delete_func(func);  // TODO
        functiontoremove.pop_back();
        if (functiontoremove.empty()) {
            functiontoremove = getinlineFunc(module);
        }
    }
}

}  // namespace pass