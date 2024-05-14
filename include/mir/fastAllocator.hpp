#pragma once
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/LiveInterval.hpp"

#include <queue>
#include <unordered_set>
#include <iostream>

namespace mir {
static void fastAllocator(MIRFunction& mfunc, CodeGenContext& ctx) {
    auto liveInterval = calcLiveIntervals(mfunc, ctx);  // 计算变量活跃区间

    // 虚拟寄存器相关使用信息
    struct VirtualRegisterInfo final {
        std::unordered_set<mir::MIRBlock*> _uses;  // use
        std::unordered_set<mir::MIRBlock*> _defs;  // def
    };
    std::unordered_map<MIROperand*, VirtualRegisterInfo, MIROperandHasher> vreg_info;
    std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> isaReg_hint;

    for (auto& block : mfunc.blocks()) {
        for (auto& inst : block->insts()) {
            auto& instInfo = ctx.instInfo.get_instinfo(inst);
            /* ??? */
            if (inst->opcode() == InstCopyFromReg) {
                isaReg_hint[inst->operand(0)] = inst->operand(1);
            }
            if (inst->opcode() == InstCopyToReg) {
                isaReg_hint[inst->operand(1)] = inst->operand(0);
            }
            /* 统计虚拟寄存器相关定义和使用情况 */
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                auto op = inst->operand(idx);
                if (!isOperandVReg(op)) continue;
                
                if (instInfo.operand_flag(idx) & OperandFlagUse) {  // used
                    vreg_info[op]._uses.insert(block.get());
                } 
                if (instInfo.operand_flag(idx) & OperandFlagDef) {  // defed
                    vreg_info[op]._defs.insert(block.get());
                }
            }
        }
    }

    // find all cross-block vregs and allocate stack slot for them
    std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> stack_map;        // 栈映射 (全局分析) - 需要spill到虚拟寄存器 -> 栈
    std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> isaReg_stack_map;
    {
        for (auto [vreg, info] : vreg_info) {
            /* 1. reg未在块中被定义或定义后未被使用 -> invalid */
            if (info._uses.empty() || info._defs.empty()) continue;
            /* 2. reg定义和使用在同一块中 */
            if (info._uses.size() == 1 && info._defs.size() == 1 && *(info._uses.begin()) == *(info._defs.begin())) continue;
            /* 3. reg的定义和使用跨多个块 -> 防止占用寄存器过久, spill到内存 */
            auto size = getOperandSize(ctx.registerInfo->getCanonicalizedRegisterType(vreg->type()));
            auto storage = mfunc.add_stack_obj(ctx.next_id(), size, size, 0, StackObjectUsage::RegSpill); 
            stack_map[vreg] = storage;
        }
    }

    for (auto& block : mfunc.blocks()) {
        std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> local_stack_map;  // 栈映射 (局部分析)
        std::unordered_map<MIROperand*, std::vector<MIROperand*>, MIROperandHasher> cur_map;  // 当前块中, 每个虚拟寄存器的映射 -> 栈 or 物理寄存器/
        std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> phy_map;
        std::unordered_map<uint32_t, std::queue<MIROperand*>> allocation_queue;  // 分为两类: int and float
        std::unordered_set<MIROperand*, MIROperandHasher> protected_lockedISAReg;
        std::unordered_set<MIROperand*, MIROperandHasher> underRenamed_ISAReg;

        MultiClassRegisterSelector selector { *ctx.registerInfo };

        /* 给定操作数, 得到与其相关的栈映射 */
        const auto get_stack_storage = [&](MIROperand* op) {
            if (auto it = local_stack_map.find(op); it != local_stack_map.end()) return it->second;
            if (auto it = stack_map.find(op); it != stack_map.end()) return local_stack_map[op] = it->second;
            auto size = getOperandSize(ctx.registerInfo->getCanonicalizedRegisterType(op->type()));
            auto storage = mfunc.add_stack_obj(ctx.next_id(), size, size, 0, StackObjectUsage::RegSpill);
            return local_stack_map[op] = storage;
        };
        const auto get_datamap = [&](MIROperand* op) -> std::vector<MIROperand*>& {
            auto& map = cur_map[op];
            if (map.empty()) map.push_back(get_stack_storage(op));
            return map;
        };

        auto& instructions = block->insts();

        std::unordered_set<MIROperand*, MIROperandHasher> dirty_regs;
        auto& liveIntervalInfo = liveInterval.block2Info[block.get()];

        auto is_allocatableType = [](OperandType type) { return type <= OperandType::Float32; };
        /* collect underRenamed ISARegisters */
        auto collect_underRenamedISARegs = [&](MIRInstList::iterator it) {
            while (it != instructions.end()) {
                auto inst = *it;
                auto& instInfo = ctx.instInfo.get_instinfo(inst);
                bool hasReg = false;
                for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                    auto op = inst->operand(idx);
                    if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op->reg()) &&
                        is_allocatableType(op->type()) && (instInfo.operand_flag(idx) & OperandFlagUse)) {
                        underRenamed_ISAReg.insert(op);
                        hasReg = true;
                    }
                }
                if (hasReg) ++it;
                else break;
            }
        };
        collect_underRenamedISARegs(instructions.begin());
        
        for (auto it = instructions.begin(); it != instructions.end();) {
            auto next = std::next(it);
            std::unordered_set<MIROperand*, MIROperandHasher> protect;
            std::unordered_set<MIROperand*, MIROperandHasher> release_vreg;

            /* spill register to stack */
            const auto evict_reg = [&](MIROperand* operand) {
                // assert(isOperandVReg(operand));
                auto& map = get_datamap(operand);
                MIROperand* isaReg = nullptr;
                bool already_in_stack = false;
                for (auto reg : map) {
                    if (isStackObject(reg->reg())) {
                        already_in_stack = true;
                    } else if (isISAReg(reg->reg())) {
                        isaReg = reg;
                    }
                }
                if (!isaReg) return;  // 当前虚拟寄存器已经被spill到栈中, 无需进行spill操作
                phy_map.erase(isaReg);
                auto stack_storage = get_stack_storage(operand);
                if (!already_in_stack) {  // spill to stack
                    auto inst = new MIRInst(InstStoreRegToStack);
                    inst->set_operand(0, stack_storage); inst->set_operand(1, isaReg);
                    instructions.insert(it, inst);
                }
                map = { stack_storage };
            };
            /* ??? */
            const auto is_protected = [&](MIROperand* isaReg) {
                assert(isOperandISAReg(isaReg));
                return protect.count(isaReg) || protected_lockedISAReg.count(isaReg) || underRenamed_ISAReg.count(isaReg);
            };
            /* ??? */
            const auto get_free_reg = [&](MIROperand* op) -> MIROperand* {
                auto reg_class = ctx.registerInfo->get_alloca_class(op->type());
                auto& q = allocation_queue[reg_class];
                MIROperand* isaReg;

                const auto get_free_register = [&] {
                    std::vector<MIROperand*> tmp;
                    do {
                        auto reg = selector.getFreeRegister(op->type());
                        if (reg->is_unused()) {  // 寄存器数量不够
                            for (auto operand : tmp) {  // 释放相关寄存器
                                selector.markAsDiscarded(*operand);
                            }
                            return new MIROperand;
                        }
                        if (is_protected(reg)) {
                            tmp.push_back(reg);
                            selector.markAsUsed(*reg);
                        } else {
                            for (auto operand : tmp) {
                                selector.markAsDiscarded(*operand);
                            }
                            return reg;
                        }
                    } while (true);
                };
            
                if (auto hintIter = isaReg_hint.find(op); 
                    hintIter != isaReg_hint.end() && selector.isFree(*(hintIter->second)) && !is_protected(hintIter->second)) {
                    isaReg = hintIter->second;
                } else if (auto reg = get_free_register(); !reg->is_unused()) {
                    isaReg = reg;
                } else {  // evict
                    assert(!q.empty());
                    isaReg = q.front();
                    while (is_protected(isaReg)) {
                        assert(q.size() != 1);
                        q.pop(); q.push(isaReg);
                        isaReg = q.front();
                    }
                    q.pop();
                    selector.markAsDiscarded(*isaReg);
                }
                if (auto it = phy_map.find(isaReg); it != phy_map.end()) evict_reg(it->second);
                assert(!is_protected(isaReg));

                q.push(isaReg);
                phy_map[isaReg] = op;
                selector.markAsUsed(*isaReg);
                return isaReg;
            };
            const auto use = [&](MIROperand* op) {
                if (!isOperandVReg(op)) {  // 栈 or 物理寄存器
                    if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op->reg()) && is_allocatableType(op->type())) {
                        underRenamed_ISAReg.erase(op);
                    }
                    return;
                }

                // 虚拟寄存器
                if (op->reg_flag() & RegisterFlagDead) release_vreg.insert(op);

                auto& map = get_datamap(op);
                MIROperand* stack_storage;
                for (auto& reg : map) {
                    if (!isStackObject(reg->reg())) {
                        // loaded
                        *op = *reg; protect.insert(op);
                        return;
                    }
                    stack_storage = reg;
                }

                // load from stack
                assert(!stack_storage->is_unused());
                auto reg = get_free_reg(op);
                auto inst = new MIRInst(InstLoadRegFromStack);
                inst->set_operand(1, stack_storage); inst->set_operand(0, reg);
                instructions.insert(it, inst);
                *op = *reg; map.push_back(op);
                protect.insert(op);
            };
            const auto def = [&](MIROperand* op) {
                if (!isOperandVReg(op)) {
                    if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op->reg()) && is_allocatableType(op->type())) {
                        protected_lockedISAReg.erase(op);
                        if (auto it = phy_map.find(op); it != phy_map.end()) {
                            evict_reg(it->second);
                        }
                    }
                    return;
                }

                if (stack_map.count(op)) dirty_regs.insert(op);

                auto& map = get_datamap(op);
                MIROperand* stack_storage;
                for (auto& reg : map) {
                    if (!isStackObject(reg->reg())) {
                        *op = *reg; map = { op }; protect.insert(op);
                        return;
                    }
                    stack_storage = reg;
                }
                auto reg = get_free_reg(op); *op = *reg;
                map = { op }; protect.insert(op);
            };
            const auto before_branch = [&]() {  // write back all out dirty vregs into stack slots before branch
                for(auto dirty : dirty_regs) {
                    if(liveIntervalInfo.outs.count(dirty->reg())) evict_reg(dirty);
                }
            };

            auto& inst = *it;
            auto& instInfo = ctx.instInfo.get_instinfo(inst);
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                auto flag = instInfo.operand_flag(idx);
                if ((flag & OperandFlagUse) || (flag & OperandFlagDef)) {  // 寄存器, 排除立即数
                    auto op = inst->operand(idx);
                    if (!isOperandVReg(op)) {
                        if (isOperandISAReg(op)) protect.insert(op);
                    }
                }
            }
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                auto flag = instInfo.operand_flag(idx);
                if (flag & OperandFlagUse) {
                    auto op = inst->operand(idx);
                    use(op);
                }
            }
            if (requireFlag(instInfo.inst_flag(), InstFlagCall)) {
                std::vector<MIROperand*> saved_regs;  // 调用者保存寄存器
                for (auto [p, v] : phy_map) {
                    if (ctx.frameInfo.is_caller_saved(*p)) {
                        saved_regs.push_back(v);
                    }
                }
                for (auto v : saved_regs) evict_reg(v);
                protected_lockedISAReg.clear();
                collect_underRenamedISARegs(next);
            
            }
        
            protect.clear();
            for (auto operand : release_vreg) {  // release dead vregs
                auto& map = get_datamap(operand);
                for (auto& reg : map) {
                    if (isISAReg(reg->reg())) {
                        phy_map.erase(reg);
                        selector.markAsDiscarded(*reg);
                    }
                }
                map.clear();
            }
        
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                if (instInfo.operand_flag(idx) & OperandFlagDef) {
                    auto op = inst->operand(idx);
                    def(op);
                }
            }

            if (requireFlag(instInfo.inst_flag(), InstFlagBranch)) {
                before_branch();
            }

            it = next;
        }
    }
}
}  // namespace mir
