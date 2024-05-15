#pragma once
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/LiveInterval.hpp"

#include <queue>
#include <unordered_set>
#include <iostream>

namespace mir {
static void fastAllocator(MIRFunction& mfunc, CodeGenContext& ctx) {
    std::cout << "fast allocator begin" << std::endl;
    auto liveInterval = calcLiveIntervals(mfunc, ctx);  // 计算变量活跃区间

    struct VirtualRegisterInfo final {  // 虚拟寄存器相关使用信息
        std::unordered_set<mir::MIRBlock*> _uses;
        std::unordered_set<mir::MIRBlock*> _defs;
    };
    std::unordered_map<MIROperand*, VirtualRegisterInfo, MIROperandHasher> useDefIfno;
    std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> isaRegHint;

    MultiClassRegisterSelector selector { *ctx.registerInfo };

    for (auto& block : mfunc.blocks()) {
        for (auto& inst : block->insts()) {
            auto& instInfo = ctx.instInfo.get_instinfo(inst);

            if (inst->opcode() == InstCopyFromReg) {
                isaRegHint[inst->operand(0)] = inst->operand(1);
            } else if (inst->opcode() == InstCopyToReg) {
                isaRegHint[inst->operand(1)] = inst->operand(0);
            }

            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                auto op = inst->operand(idx);
                if (!isOperandVReg(op)) continue;
                
                auto new_op = selector.getFreeRegister(op->type());
                *op = new_op;
                selector.markAsUsed(new_op);
                // if (instInfo.operand_flag(idx) & OperandFlagUse) {
                //     useDefIfno[op]._uses.insert(block.get());
                // } 
                // if (instInfo.operand_flag(idx) & OperandFlagDef) {
                //     useDefIfno[op]._defs.insert(block.get());
                }
            }
        }
    }

    // find all cross-block vregs and allocate stack slot for them
    // std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> stackMap;  // 全局栈存储
    // std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> isaRegStackMap;

    // {
    //     for (auto& [reg, info] : useDefIfno) {
    //         if (info._uses.empty() || info._defs.empty()) continue;  // invalid
    //         if (info._uses.size() == 1 && info._defs.size() == 1 && *(info._uses.begin()) == *(info._defs.begin())) continue;  // local
        
    //         auto size = getOperandSize(ctx.registerInfo->getCanonicalizedRegisterType(reg->type()));
    //         auto storage = mfunc.add_stack_obj(ctx.next_id(), size, size, 0, StackObjectUsage::RegSpill); 
    //         stackMap[reg] = storage;
    //     }
    // }

    // for (auto& block : mfunc.blocks()) {
        // std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> localStackMap;  // 局部栈存储

        // std::unordered_map<MIROperand*, std::vector<MIROperand*>, MIROperandHasher> curMap;
        // std::unordered_map<MIROperand*, MIROperand*, MIROperandHasher> phyMap;
        // std::unordered_map<uint32_t, std::queue<MIROperand*>> allocationQueue;
        
        // std::unordered_set<MIROperand*, MIROperandHasher> protectedLockedISAReg;  // retvals/callee arguments
        // std::unordered_set<MIROperand*, MIROperandHasher> underRenamedISAReg;     // callee retvals/arguments

        // MultiClassRegisterSelector selector { *ctx.registerInfo };

        // auto get_stack_storage = [&](MIROperand* op) {  // 获取给定MIROperand的栈存储
        //     if (auto it = localStackMap.find(op); it != localStackMap.end()) return it->second;    
        //     auto& ref = localStackMap[op];
        //     if (auto it = stackMap.find(op); it != stackMap.end()) return ref = it->second;
        //     auto size = getOperandSize(ctx.registerInfo->getCanonicalizedRegisterType(op->type()));
        //     auto storage = mfunc.add_stack_obj(ctx.next_id(), size, size, 0, StackObjectUsage::RegSpill);
        //     return ref = storage;
        // };
        // auto get_data_map = [&](MIROperand* op) -> std::vector<MIROperand*>& {
        //     auto& map = curMap[op];
        //     if (map.empty()) map.push_back(get_stack_storage(op));
        //     return map;
        // };

        // auto& instructions = block->insts();

        // std::unordered_set<MIROperand*, MIROperandHasher> dirtyRegs;
        // auto& liveIntervalInfo = liveInterval.block2Info[block.get()];

        // auto isAllocatableType = [](OperandType type) { return type <= OperandType::Float32; };
        // auto collectUnderRenamedISARegs = [&](MIRInstList::iterator it) {
        //     while (it != instructions.end()) {
        //         auto inst = *it;
        //         auto& instInfo = ctx.instInfo.get_instinfo(inst);
        //         bool hasReg = false;
        //         for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
        //             auto op = inst->operand(idx);
        //             if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op->reg()) && isAllocatableType(op->type()) && (instInfo.operand_flag(idx) & OperandFlagUse)) {
        //                 underRenamedISAReg.insert(op);
        //                 hasReg = true;
        //             }
        //         }
        //         if (hasReg) ++it;
        //         else break;
        //     }
        // };

        // collectUnderRenamedISARegs(instructions.begin());
        
        // for (auto it = instructions.begin(); it != instructions.end();) {

            // auto next = std::next(it);

            // auto evictVReg = [&](MIROperand* operand) {
            //     assert(isOperandVReg(operand));

            //     auto& map = get_data_map(operand);
            //     MIROperand* isaReg;
            //     bool alreadyInStack = false;
            //     for (auto reg : map) {
            //         if (isStackObject(reg->reg())) alreadyInStack = true;
            //         if (isISAReg(reg->reg())) isaReg = reg;
            //     }
            //     if (isaReg->is_unused()) return;

            //     phyMap.erase(isaReg);
            //     auto stackStorage = get_stack_storage(operand);
            //     if (!alreadyInStack) {
            //         // spill to stack
            //         instructions.insert(it, MIRInst{ InstStoreRegToStack }.set_operand(0, isaReg)->set_operand(1, stackStorage));
            //     }
            //     map = { stackStorage };
            // };
        // }

        // std::unordered_set<MIROperand*, MIROperandHasher> protect;
        // auto isProtected = [&](MIROperand* isaReg) {
        //     assert(isOperandISAReg(isaReg));
        //     return protect.count(isaReg) || protectedLockedISAReg.count(isaReg) || underRenamedISAReg.count(isaReg);
        // };
        // auto getFreeReg = [&](MIROperand* operand) ->MIROperand* {
        //     auto regClass = ctx.registerInfo->get_alloca_class(operand->type());
        //     auto& q = allocationQueue[regClass];
        //     MIROperand* isaReg;

        //     auto getFreeRegister = [&] {
        //         std::vector<MIROperand*> temp;
        //         do {
        //             auto reg = selector.getFreeRegister(operand->type());
        //             if (reg.is_unused()) {
        //                 for (auto op : temp) selector.markAsDiscarded(*op);
        //                 return MIROperand{};
        //             }
        //             if (isProtected(&reg)) {
        //                 temp.push_back(&reg);
        //                 selector.markAsUsed(reg);
        //             } else {
        //                 for (auto op : temp) selector.markAsDiscarded(*op);
        //                 return reg;
        //             }
        //         } while (true);
        //     };

        //     if (auto hintIter = isaRegHint.find(operand); 
        //         hintIter != isaRegHint.end() && selector.isFree(*(hintIter->second)) && !isProtected(hintIter->second)) {
        //         isaReg = hintIter->second;
        //     } else if (auto reg = getFreeRegister(); !reg.is_unused()) {
        //         isaReg = &reg;
        //     }
        // };
        // auto inst = 
    // }
// }
}  // namespace mir
