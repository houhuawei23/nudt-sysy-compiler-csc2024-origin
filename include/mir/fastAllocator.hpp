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
    std::unordered_map<MIROperand*, VirtualRegisterInfo, MIROperandHasher> useDefInfo;
    std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> isaRegHint;

    MultiClassRegisterSelector selector { *ctx.registerInfo };

    for (auto& block : mfunc.blocks()) {
        for (auto& inst : block->insts()) {
            auto& instInfo = ctx.instInfo.get_instinfo(inst);

            if (inst->opcode() == InstCopyFromReg) {
                isaRegHint[inst->operand(0)] = inst->operand(1);
            }
            if (inst->opcode() == InstCopyToReg) {
                isaRegHint[inst->operand(1)] = inst->operand(0);
            }

            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                auto op = inst->operand(idx);
                if (!isOperandVReg(op)) continue;
                
                if (instInfo.operand_flag(idx) & OperandFlagUse) {
                    useDefInfo[op]._uses.insert(block.get());
                } 
                if (instInfo.operand_flag(idx) & OperandFlagDef) {
                    useDefInfo[op]._defs.insert(block.get());
                }
            }
        }
    }

    // find all cross-block vregs and allocate stack slot for them
    std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> stackMap;        // 全局栈映射
    std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> isaRegStackMap;

    {
        for (auto& [reg, info] : useDefInfo) {
            /* 1. reg未在块中被定义或被使用 -> invalid */
            if (info._uses.empty() || info._defs.empty()) continue;
            /* 2. reg定义和使用在同一块中 -> local */
            if (info._uses.size() == 1 && info._defs.size() == 1 && *(info._uses.begin()) == *(info._defs.begin())) continue;
            /* 3. reg的定义和使用跨多个块 -> spill到内存 */
            auto size = getOperandSize(ctx.registerInfo->getCanonicalizedRegisterType(reg->type()));
            auto storage = mfunc.add_stack_obj(ctx.next_id(), size, size, 0, StackObjectUsage::RegSpill); 
            stackMap[reg] = storage;
        }
    }

    for (auto& block : mfunc.blocks()) {
        std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> localStackMap;       // 局部栈映射
        std::unordered_map<MIROperand*, std::vector<MIROperand*>, MIROperandHasher> curMap;

        std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> phyMap;  // physical map: Operand* --> Operand*
        std::unordered_map<uint32_t, std::queue<MIROperand*>> allocationQueue;  // 分为两类: int and float
        
        std::unordered_set<MIROperand*, MIROperandHasher> protectedLockedISAReg;  // retvals/callee arguments
        std::unordered_set<MIROperand*, MIROperandHasher> underRenamedISAReg;     // callee retvals/arguments

        MultiClassRegisterSelector selector { *ctx.registerInfo };

        const auto get_stack_storage = [&](MIROperand* op) {
            if (auto it = localStackMap.find(op); it != localStackMap.end()) return it->second;    
            auto ref = localStackMap[op];
            if (auto it = stackMap.find(op); it != stackMap.end()) return localStackMap[op] = it->second;
            auto size = getOperandSize(ctx.registerInfo->getCanonicalizedRegisterType(op->type()));
            auto storage = mfunc.add_stack_obj(ctx.next_id(), size, size, 0, StackObjectUsage::RegSpill);
            return localStackMap[op] = storage;
        };
        const auto get_datamap = [&](MIROperand* op) -> std::vector<MIROperand*>& {
            auto& map = curMap[op];
            if (map.empty()) map.push_back(get_stack_storage(op));
            return map;
        };

        auto& instructions = block->insts();

        std::unordered_set<MIROperand*, MIROperandHasher> dirtyRegs;
        auto& liveIntervalInfo = liveInterval.block2Info[block.get()];

        auto is_allocatableType = [](OperandType type) { return type <= OperandType::Float32; };
        auto collect_under_renamedISARegs = [&](MIRInstList::iterator it) {
            while (it != instructions.end()) {
                auto inst = *it;
                auto& instInfo = ctx.instInfo.get_instinfo(inst);
                bool hasReg = false;
                for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                    auto op = inst->operand(idx);
                    if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op->reg()) &&
                        is_allocatableType(op->type()) && (instInfo.operand_flag(idx) & OperandFlagUse)) {
                        underRenamedISAReg.insert(op);
                        hasReg = true;
                    }
                }
                if (hasReg) ++it;
                else break;
            }
        };
        collect_under_renamedISARegs(instructions.begin());
        
        for (auto it = instructions.begin(); it != instructions.end();) {
            auto next = std::next(it);
            std::unordered_set<MIROperand*, MIROperandHasher> protect;
            std::unordered_set<MIROperand*, MIROperandHasher> release_vreg;

            /* utils function */
            /* spill register to stack */
            const auto evict_reg = [&](MIROperand* operand) {
                assert(isOperandVReg(operand));
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
                if (!isaReg) return;
                phyMap.erase(isaReg);
                auto stack_storage = get_stack_storage(operand);
                if (!already_in_stack) {  // spill to stack
                    auto inst = new MIRInst(InstStoreRegToStack);
                    inst->set_operand(0, stack_storage); inst->set_operand(1, isaReg);
                    instructions.insert(it, inst);
                }
                map = { stack_storage };
            };
            const auto is_protected = [&](MIROperand* isaReg) {
                assert(isOperandISAReg(isaReg));
                return protect.count(isaReg) || protectedLockedISAReg.count(isaReg) || underRenamedISAReg.count(isaReg);
            };
            const auto get_free_reg = [&](MIROperand* op) -> MIROperand* {
                auto reg_class = ctx.registerInfo->get_alloca_class(op->type());
                auto& q = allocationQueue[reg_class];
                MIROperand* isaReg;

                const auto get_free_register = [&] {
                    std::vector<MIROperand*> tmp;
                    do {
                        auto reg = selector.getFreeRegister(op->type());
                        if (reg->is_unused()) {  // in case of invalid register
                            for (auto operand : tmp) {
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
            
                if (auto hintIter = isaRegHint.find(op); 
                    hintIter != isaRegHint.end() && selector.isFree(*(hintIter->second)) && !is_protected(hintIter->second)) {
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
                if (auto it = phyMap.find(isaReg); it != phyMap.end()) evict_reg(it->second);
                assert(!is_protected(isaReg));

                q.push(isaReg);
                phyMap[isaReg] = op;
                selector.markAsUsed(*isaReg);
                return isaReg;
            };
            const auto use = [&](MIROperand* op, MIRInst* instruction, int idx) {
                if (!isOperandVReg(op)) {
                    if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op->reg()) && is_allocatableType(op->type())) {
                        underRenamedISAReg.erase(op);
                    }
                    return;
                }
                if (op->reg_flag() & RegisterFlagDead) release_vreg.insert(op);

                auto& map = get_datamap(op);
                MIROperand* stack_storage;
                for (auto& reg : map) {
                    if (!isStackObject(reg->reg())) {
                        // loaded
                        op = reg; instruction->set_operand(idx, reg);
                        protect.insert(reg);
                        return;
                    }
                }

                // load from stack
                assert(!stack_storage->is_unused());
                auto reg = get_free_reg(op);
                auto inst = new MIRInst(InstLoadRegFromStack);
                inst->set_operand(1, stack_storage); inst->set_operand(0, reg);
                instructions.insert(it, inst);
                map.push_back(reg); op = reg;
                protect.insert(reg);
                instruction->set_operand(idx, reg);
            };
            const auto def = [&](MIROperand* op, MIRInst* instruction, int idx) {
                if (!isOperandVReg(op)) {
                    if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op->reg()) && is_allocatableType(op->type())) {
                        protectedLockedISAReg.erase(op);
                        if (auto it = phyMap.find(op); it != phyMap.end()) {
                            evict_reg(it->second);
                        }
                    }
                    return;
                }

                if (stackMap.count(op)) dirtyRegs.insert(op);

                auto& map = get_datamap(op);
                MIROperand* stack_storage;
                for (auto& reg : map) {
                    if (!isStackObject(reg->reg())) {
                        instruction->set_operand(idx, reg);
                        op = reg; map = { reg }; protect.insert(reg);
                        return;
                    }
                    stack_storage = reg;
                }
                auto reg = get_free_reg(op);
                map = { reg }; protect.insert(reg);
                instruction->set_operand(idx, reg);
            };
            const auto before_branch = [&]() {  // write back all out dirty vregs into stack slots before branch
                for(auto dirty : dirtyRegs) {
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
                    use(op, inst, idx);

                }
            }
            if (requireFlag(instInfo.inst_flag(), InstFlagCall)) {
                std::vector<MIROperand*> saved_regs;  // 调用者保存寄存器
                std::unordered_set<MIROperand*, MIROperandHasher>* callee_usage = nullptr;

                // if (auto symbol = inst->operand(0))
            }
        
            protect.clear();
            for (auto operand : release_vreg) {  // release dead vregs
                auto& map = get_datamap(operand);
                for (auto& reg : map) {
                    if (isISAReg(reg->reg())) {
                        phyMap.erase(reg);
                        selector.markAsDiscarded(*reg);
                    }
                }
                map.clear();
            }
        
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                if (instInfo.operand_flag(idx) & OperandFlagDef) {
                    auto op = inst->operand(idx);
                    def(op, inst, idx);
                }
            }

            if (requireFlag(instInfo.inst_flag(), InstFlagBranch)) {
                before_branch();
            }
        }
    }
}
}  // namespace mir
