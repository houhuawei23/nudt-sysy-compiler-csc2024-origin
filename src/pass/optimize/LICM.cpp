#include "pass/optimize/optimize.hpp"
#include "pass/optimize/LICM.hpp"
#include "ir/value.hpp"
#include <set>
#include <cassert>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>

namespace pass {
void LICM::clear() {
    Allallocas.clear();
    allocaDefs.clear();
    allocaUses.clear();
}

void LICM::getallallocas(ir::Function* func) {  // 得到函数内所有数组的alloca
    for (auto bb : func->blocks()) {
        for (auto inst : bb->insts()) {
            if (auto alloca = inst->dynCast<ir::AllocaInst>()) {
                Allallocas.push_back(alloca);
            }
        }
    }
}

void LICM::storeLiftfunc(ir::Function* func) {
    getallallocas(func);
    bool change = true;
    for (auto allocainst : Allallocas) {  // 对于每个alloca，通过dfs所有user得到所有的定义与使用
        for (auto puse = allocainst->uses().begin(); puse != allocainst->uses().end();) {
            auto use = *puse;
            puse++;
            dfs(allocainst, use->user()->dynCast<ir::Instruction>());
        }
    }

    for (auto alloca : Allallocas) {  // 对于每个alloca，进行store提升，不动点算法
        while(change)
            change = storemove(alloca);
    }
}

void LICM::storeLift(ir::Module* module) {
    bool change;
    for (auto func : module->funcs()) {  // 遍历所有函数，对于每个函数做storelift直到没有变化
        if (func->isOnlyDeclare())
            continue;
        clear();
        storeLiftfunc(func);
    }
}

void LICM::dfs(ir::AllocaInst* alloca, ir::Instruction* inst) {
    // std::cerr<<"dfs"<<std::endl;
    if (inst->dynCast<ir::GetElementPtrInst>()) {  // gep也视作alloca
        for (auto puse = inst->uses().begin(); puse != inst->uses().end();) {
            auto use = *puse;
            puse++;
            dfs(alloca, use->user()->dynCast<ir::Instruction>());
        }
    } else if (inst->dynCast<ir::StoreInst>()) {  // store视作定义
        allocaDefs[alloca].insert(inst);
    } else if (inst->dynCast<ir::LoadInst>()) {  // load视作使用
        allocaUses[alloca].insert(inst);
    } else if (ir::CallInst* call = inst->dynCast<ir::CallInst>()) {
        if (call->isgetarrayorfarray()) {  // 如果调用getfarray与getarray，则视作定义
            allocaDefs[alloca].insert(inst);
        } else if (call->isputarrayorfarray()) {  // 如果调用putarray，则视作使用
            allocaUses[alloca].insert(inst);
        } else {  // 其他视作定义与使用
            allocaDefs[alloca].insert(inst);
            allocaUses[alloca].insert(inst);
        }
    } else if (inst->dynCast<ir::MemsetInst>()) {  // memset视作定义
        allocaDefs[alloca].insert(inst);
    } else if (inst->dynCast<ir::BitCastInst>()) {  // bitcast视作alloca
        for (auto puse = inst->uses().begin(); puse != inst->uses().end();) {
            auto use = *puse;
            puse++;
            dfs(alloca, use->user()->dynCast<ir::Instruction>());
        }
    } else {
        return;
    }
}

void LICM::globaldfs(ir::Value* val, ir::Instruction* inst) {
    if (inst->dynCast<ir::GetElementPtrInst>()) {
        for (auto puse = inst->uses().begin(); puse != inst->uses().end();) {
            auto use = *puse;
            puse++;
            globaldfs(val, use->user()->dynCast<ir::Instruction>());
        }
    } else if (inst->dynCast<ir::StoreInst>()) {
        gDefs[val].insert(inst);
    } else if (inst->dynCast<ir::LoadInst>()) {
        gUses[val].insert(inst);
    } else if (ir::CallInst* call = inst->dynCast<ir::CallInst>()) {
        if (call->isgetarrayorfarray()) {  // 如果调用getfarray与getarray，则视作定义
            gDefs[val].insert(inst);
        } else if (call->isputarrayorfarray()) {  // 如果调用putarray，则视作使用
            gUses[val].insert(inst);
        } else {  // 其他视作定义与使用
            gDefs[val].insert(inst);
            gUses[val].insert(inst);
        }
    } else if (inst->dynCast<ir::MemsetInst>()) {
        gDefs[val].insert(inst);
    } else if (inst->dynCast<ir::BitCastInst>()) {
        for (auto puse = inst->uses().begin(); puse != inst->uses().end();) {
            auto use = *puse;
            puse++;
            globaldfs(val, use->user()->dynCast<ir::Instruction>());
        }
    } else {
        return;
    }
}

bool LICM::storemove(ir::AllocaInst* alloca) {
    std::set<ir::Instruction*> DefsforAlloca = allocaDefs[alloca];
    std::set<ir::Instruction*> UsesforAlloca = allocaUses[alloca];
    // assert(!DefsforAlloca.empty() && "DefsforAlloca is empty");
    if (DefsforAlloca.empty())
        return false;
    ir::BasicBlock* bb = nullptr;
    // for (auto def : DefsforAlloca) {
    //     if (!bb) {
    //         bb = def->block();
    //     } else {
    //         if (bb != def->block())  // 只处理所有定义在同一个块的情况
    //             return false;
    //     }
    // }

    // for (auto def : DefsforAlloca){
    //     ;
    // }
    assert(bb && "bb is null");
    loopctx = flmap[bb->function()];

    if (loopctx->looplevel(bb) == 0)  // 不在循环内，直接返回
        return false;

    bool finduse = false;
    for (auto inst : bb->insts()) {  // 检查是否存在使用后定义的情况
        if (UsesforAlloca.count(inst)) {
            finduse = true;
        }

        if (finduse && DefsforAlloca.count(inst)) {
            return false;
        }
    }

    for (auto def : DefsforAlloca) {
        if (def->dynCast<ir::MemsetInst>()) {  // memset均为0
            ;

        } else if (def->dynCast<ir::CallInst>()) {  // 调用指令不移动
            return false;

        } else if (auto storeinst = def->dynCast<ir::StoreInst>()) {  // 要求存的值与位置均为常数
            ir::Value* ptr = storeinst->ptr();
            if (!storeinst->value()->dynCast<ir::Constant>())  // 值不为常数则返回
                return false;

            if (auto gepinst = ptr->dynCast<ir::GetElementPtrInst>()) {  // index不为常数则返回 TODO 若不为常数，根据支配关系也可提升
                if (!gepinst->index()->dynCast<ir::Constant>()) return false;
            } else {
                return false;
            }
        } else {
            assert(false && "storemove: unexpected instruction");
            return false;
        }
    }

    ir::Loop* innerloop = loopctx->getinnermostLoop(bb);
    ir::BasicBlock* perheader = innerloop->getLoopPreheader();
    std::vector<ir::Instruction*> insttolift;
    for (auto inst : bb->insts()) {//移动所有定义
        if (DefsforAlloca.count(inst)) insttolift.push_back(inst);
    }
    for (auto inst : insttolift) {
        bb->move_inst(inst);
        perheader->emplace_lastbutone_inst(inst);
        std::cerr<<" storelift" << std::endl;
    }
    return true;
}

void LICM::loadLift(ir::Module* module) {
    for (auto func : module->funcs()) {//每个func的allocas
        if (func->isOnlyDeclare()) continue;
        Allallocas.clear();
        getallallocas(func);
        for (auto allocainst : Allallocas) {
            for (auto puse = allocainst->uses().begin(); puse != allocainst->uses().end();) {
                auto use = *puse;
                puse++;
                globaldfs(allocainst, use->user()->dynCast<ir::Instruction>());
            }
        }
    }

    std::vector<ir::GlobalVariable*> globals = module->globalVars();//所有全局变量
    for (auto globalval : globals) {  // 计算每个全局变量的use与def
        for (auto puse = globalval->uses().begin(); puse != globalval->uses().end();) {
            auto use = *puse;
            puse++;
            globaldfs(globalval, use->user()->dynCast<ir::Instruction>());
        }
    }

    for (auto globalval : globals) {  // 计算每个函数显式使用的全局变量
        for (auto useinst : gUses[globalval]) {
            funcUseGlobals[useinst->block()->function()].insert(globalval);
        }

        for (auto definst : gDefs[globalval]) {  // 计算每个函数显式定义的全局变量
            funcUseGlobals[definst->block()->function()].insert(globalval);
        }
    }

    bool change = true;
    while (change) {  // 用不动点算法计算每个函数使用和定义的全局变量
        change = false;
        for (auto func : module->funcs()) {  // 遍历每个函数调用的函数，将callee对全局变量的使用和定义加入caller的集合中
            for (auto funccalled : cgctx->callees(func)) {
                for (auto calleduse : funcUseGlobals[funccalled]) {
                    if (funcUseGlobals[func].count(calleduse) == 0) {
                        funcUseGlobals[func].insert(calleduse);
                        change = true;
                    }
                }

                for (auto calleddef : funcDefGlobals[funccalled]) {
                    if (funcDefGlobals[func].count(calleddef) == 0) {
                        funcDefGlobals[func].insert(calleddef);
                        change = true;
                    }
                }
            }
        }
    }

    for (auto func : module->funcs()) {  // 对于func，将call视作对 func使用的全局变量 的使用
        for (auto Useval : funcUseGlobals[func]) {
            for (auto puse = func->uses().begin(); puse != func->uses().end();) {
                auto use = *puse;
                puse++;
                gUses[Useval].insert(use->user()->dynCast<ir::Instruction>());
            }
        }
    }

    for (auto func : module->funcs()) {  // 对于func，将call视作对 func定义的全局变量 的定义
        for (auto Defval : funcDefGlobals[func]) {
            for (auto puse = func->uses().begin(); puse != func->uses().end();) {
                auto use = *puse;
                puse++;
                gDefs[Defval].insert(use->user()->dynCast<ir::Instruction>());
            }
        }
    }

    for (auto useval : keyset(gUses)) {  // 计算每个alloca和gloval的每个use所在的最内层循环
        for (auto inst : gUses[useval]) {
            loopctx = flmap[inst->block()->function()];
            useLoops[useval][inst] = loopctx->getinnermostLoop(inst->block());
        }
    }
    for (auto defval : keyset(gDefs)) {  // 计算每个alloca和gloval的每个def所在的最内层循环
        for (auto inst : gDefs[defval]) {
            loopctx = flmap[inst->block()->function()];
            defLoops[defval].insert(loopctx->getinnermostLoop(inst->block()));//?use不在循环内？
        }
    }

    for (auto array : keyset(gUses)) {
        change = true;
        while (change)
            change = loadmove(array);
    }
}

bool LICM::loadmove(ir::Value* array) {
    bool change = false;
    for (auto user : gUses[array]) {
        if (!user->dynCast<ir::LoadInst>())  // 只移动load指令
            continue;
        auto userbb = user->block();
        loopctx = flmap[userbb->function()];
        
        if (loopctx->looplevel(userbb) == 0)  // 循环深度为0不需要移动
            continue;
        ir::Loop* innerLoop = useLoops[array][user];
        auto preheader = innerLoop->getLoopPreheader();
        bool flag = false;
        for (auto defloop : defLoops[array]) {
            if (!defloop){
                continue;
            }
            if (check(innerLoop, defloop)){
                flag = true; 
                break;
            }
        }
        if (flag) continue;

        if (defLoops[array].count(innerLoop) == 0) {//如果定义已经被lift
            if (auto gep = (user->dynCast<ir::LoadInst>()->ptr())->dynCast<ir::GetElementPtrInst>()) {
                if (!gep->index()->dynCast<ir::Constant>()) continue;
                if (loopctx->getinnermostLoop(gep->block()) == innerLoop) continue;//如果gep还在也不能提
            }

            auto userinst = user->dynCast<ir::Instruction>();
            userinst->block()->move_inst(userinst);
            preheader->emplace_lastbutone_inst(userinst);
            change = true;
            std::cerr<<" loadlift" << std::endl;
        }
    }
    return change;
}

bool LICM::check(ir::Loop* innerLoop, ir::Loop* defloop) {
    auto iheader = innerLoop->header();
    auto dheader = defloop->header();

    auto ifunc = iheader->function();
    auto dfunc = dheader->function();
    if (ifunc != dfunc) return false;

    auto ideep = loopctx->looplevel(iheader);
    auto ddeep = loopctx->looplevel(dheader);
    if (ideep == ddeep) return innerLoop == defloop;
    if (ideep > ddeep)
        return false;
    else {
        for (int i = 0; i < (ddeep - ideep); i++) {
            defloop = defloop->parent();
        }
    }
    return innerLoop == defloop;
}

std::vector<ir::Value*> LICM::keyset(std::unordered_map<ir::Value*, std::set<ir::Instruction*>> map) {
    std::vector<ir::Value*> keys;
    for (auto it = map.begin(); it != map.end(); ++it) {
        keys.push_back(it->first);
    }
    return keys;
}

void LICM::run(ir::Module* module, TopAnalysisInfoManager* tp) {
    cgctx = tp->getCallGraph();
    cgctx->refresh();
    for (auto func : module->funcs()) {
        if (func->isOnlyDeclare()) continue;
        loopctx = tp->getLoopInfo(func);
        loopctx->refresh();
        flmap[func] = loopctx;
    }
    storeLift(module);
    // loadLift(module);
}
}  // namespace pass