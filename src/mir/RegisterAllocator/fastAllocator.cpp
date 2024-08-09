#include "mir/MIR.hpp"
#include "mir/target.hpp"
#include "mir/LiveInterval.hpp"
#include "mir/RegisterAllocator.hpp"
#include "mir/registerinfo.hpp"
#include <queue>
#include <unordered_set>
#include <iostream>

namespace mir {
void fastAllocator(MIRFunction& mfunc, CodeGenContext& ctx, IPRAUsageCache& infoIPRA) {
    // 计算变量的活跃区间
    auto liveInterval = calcLiveIntervals(mfunc, ctx);

    // 虚拟寄存器相关使用信息
    struct VirtualRegisterInfo final {
        std::unordered_set<mir::MIRBlock*> _uses;  // use
        std::unordered_set<mir::MIRBlock*> _defs;  // def
    };
    std::unordered_map<RegNum, VirtualRegisterInfo> useDefInfo;
    std::unordered_map<RegNum, MIROperand> isaRegHint;
    for (auto& block : mfunc.blocks()) {
        for (auto& inst : block->insts()) {
            auto& instInfo = ctx.instInfo.getInstInfo(inst);

            if (inst->opcode() == InstCopyFromReg) {
                isaRegHint[inst->operand(0).reg()] = inst->operand(1);
            }
            if (inst->opcode() == InstCopyToReg) {
                isaRegHint[inst->operand(1).reg()] = inst->operand(0);
            }

            /* 统计虚拟寄存器相关定义和使用情况 */
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                auto op = inst->operand(idx);
                if (!isOperandVReg(op)) continue;
                
                if (instInfo.operand_flag(idx) & OperandFlagUse) {  // used
                    useDefInfo[op.reg()]._uses.insert(block.get());
                } 
                if (instInfo.operand_flag(idx) & OperandFlagDef) {  // defed
                    useDefInfo[op.reg()]._defs.insert(block.get());
                }
            }
        }
    }

    // find all cross-block vregs and allocate stack slot for them
    std::unordered_map<RegNum, MIROperand> stack_map;  // 全局分析 - 将跨多个块的虚拟寄存器spill到内存
    {
        for (auto [vreg, info] : useDefInfo) {
            /* 1. reg未在块中被定义或定义后未被使用 -> invalid */
            if (info._uses.empty() || info._defs.empty()) continue;
            
            /* 2. reg定义和使用在同一块中 */
            if (info._uses.size() == 1 && info._defs.size() == 1 && *(info._uses.begin()) == *(info._defs.begin())) continue;
            
            /* 3. reg的定义和使用跨多个块 -> 防止占用寄存器过久, spill到内存 */
            auto size = getOperandSize(ctx.registerInfo->getCanonicalizedRegisterType(ctx.registerInfo->getCanonicalizedRegisterType(vreg)));
            auto storage = mfunc.newStackObject(ctx.nextId(), size, size, 0, StackObjectUsage::RegSpill); 
            stack_map[vreg] = storage;
        }
    }

    /* 局部寄存器分配 -- 考虑在每个块内对其进行单独的分析 */
    for (auto& block : mfunc.blocks()) {
        std::unordered_map<RegNum, MIROperand> local_stack_map;  // 存储当前块内需要spill到内存的虚拟寄存器
        std::unordered_map<RegNum, std::vector<MIROperand>> cur_map;  // 当前块中, 每个虚拟寄存器的映射 -> 栈 or 物理寄存器
        std::unordered_map<RegNum, MIROperand> phy_map;  // 当前块中, [物理寄存器, MIROperand] (注意: 同一个物理寄存器可以分配给多个MIROperand)
        std::unordered_map<uint32_t, std::queue<MIROperand>> allocation_queue;  // 分为两类: int registers and float registers
        
        std::unordered_set<RegNum> protected_lockedISAReg;  // retvals/callee arguments
        std::unordered_set<RegNum> underRenamed_ISAReg;  // callee retvals/arguments

        MultiClassRegisterSelector selector { *ctx.registerInfo };

        /* 局部块内需要spill到内存的虚拟寄存器 */
        const auto get_stack_storage = [&](MIROperand op) {
            if (auto it = local_stack_map.find(op.reg()); it != local_stack_map.end()) return it->second;
            if (auto it = stack_map.find(op.reg()); it != stack_map.end()) return local_stack_map[op.reg()] = it->second;

            auto size = getOperandSize(ctx.registerInfo->getCanonicalizedRegisterType(op.type()));
            auto storage = mfunc.newStackObject(ctx.nextId(), size, size, 0, StackObjectUsage::RegSpill);
            return local_stack_map[op.reg()] = storage;
        };
        /* 操作数的相关映射 */
        const auto get_datamap = [&](MIROperand op) -> std::vector<MIROperand>& {
            auto& map = cur_map[op.reg()];
            if (map.empty()) map.push_back(get_stack_storage(op));
            return map;
        };

        auto& instructions = block->insts();
        std::unordered_set<RegNum> dirty_regs;
        auto& liveIntervalInfo = liveInterval.block2Info[block.get()];

        auto is_allocatableType = [](OperandType type) { return type <= OperandType::Float32; };
        /* collect underRenamed ISARegisters (寄存器重命名) */
        auto collect_underRenamedISARegs = [&](MIRInstList::iterator it) {
            while (it != instructions.end()) {
                auto inst = *it;
                auto& instInfo = ctx.instInfo.getInstInfo(inst);
                bool hasReg = false;
                for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                    auto op = inst->operand(idx);
                    if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op.reg()) && is_allocatableType(op.type()) && 
                        (instInfo.operand_flag(idx) & OperandFlagUse)) {
                        underRenamed_ISAReg.insert(op.reg());
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
            std::unordered_set<RegNum> protect;
            std::unordered_set<MIROperand, MIROperandHasher> releaseVRegs;
            
            /* 相关功能函数 */
            const auto evict_reg = [&](MIROperand operand) {
                assert(isOperandVReg(operand));
                auto& map = get_datamap(operand);
                MIROperand isaReg;
                bool already_in_stack = false;
                for (auto reg : map) {
                    if (isStackObject(reg.reg())) already_in_stack = true;
                    if (isISAReg(reg.reg())) isaReg = reg;
                }
                if (isaReg.isUnused()) return;  // 当前虚拟寄存器已经被spill到栈中, 无需进行spill操作
                
                phy_map.erase(isaReg.reg());
                auto stack_storage = get_stack_storage(operand);
                if (!already_in_stack) {  // spill to stack
                    auto inst = new MIRInst(InstStoreRegToStack);
                    inst->set_operand(0, MIROperand(stack_storage));
                    inst->set_operand(1, MIROperand::asISAReg(isaReg.reg(), ctx.registerInfo->getCanonicalizedRegisterType(isaReg.reg())));
                    instructions.insert(it, inst);
                }
                map = { stack_storage };
            };
            const auto is_protected = [&](MIROperand& isaReg) {
                assert(isOperandISAReg(isaReg));
                return protect.count(isaReg.reg()) || protected_lockedISAReg.count(isaReg.reg()) || underRenamed_ISAReg.count(isaReg.reg());
            };
            const auto get_free_reg = [&](MIROperand op) -> MIROperand {
                auto reg_class = ctx.registerInfo->getAllocationClass(op.type());
                auto& q = allocation_queue[reg_class];
                MIROperand isaReg;

                const auto get_free_register = [&] {
                    std::vector<MIROperand> temp;
                    do {
                        auto reg = selector.getFreeRegister(op.type());
                        if (reg.isUnused()) {  // 寄存器数量不够
                            for (auto operand : temp) {  // 释放相关寄存器
                                selector.markAsDiscarded(operand);
                            }
                            return MIROperand{};
                        }
                        if (is_protected(reg)) {
                            temp.push_back(reg);
                            selector.markAsUsed(reg);
                        } else {
                            for (auto operand : temp) {
                                selector.markAsDiscarded(operand);
                            }
                            return reg;
                        }
                    } while (true);
                };
            
                if (auto hintIter = isaRegHint.find(op.reg()); 
                    hintIter != isaRegHint.end() && selector.isFree(hintIter->second) && !is_protected((hintIter->second))) {
                    isaReg = hintIter->second;
                } else if (auto reg = get_free_register(); !reg.isUnused()) {
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
                    selector.markAsDiscarded(isaReg);
                }
                if (auto it = phy_map.find(isaReg.reg()); it != phy_map.end()) evict_reg(it->second);
                assert(!is_protected(isaReg));

                q.push(isaReg);
                phy_map[isaReg.reg()] = op;
                selector.markAsUsed(isaReg);
                return isaReg;
            };
            const auto use = [&](MIROperand op) {
                // 1. 栈 or 物理寄存器
                if (!isOperandVReg(op)) {
                    if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op.reg()) && is_allocatableType(op.type())) {
                        underRenamed_ISAReg.erase(op.reg());
                    }
                    return;
                }

                // 2. 虚拟寄存器
                if (op.reg_flag() & RegisterFlagDead) releaseVRegs.insert(op);

                auto& map = get_datamap(op);
                MIROperand stack_storage;
                for (auto& reg : map) {
                    // 分配物理寄存器
                    if (!isStackObject(reg.reg())) {
                        op = reg;
                        protect.insert(reg.reg());
                        return;
                    }
                    stack_storage = reg;
                }

                // 分配栈空间
                assert(!stack_storage.isUnused());
                auto reg = get_free_reg(op);
                auto inst = new MIRInst(InstLoadRegFromStack);
                inst->set_operand(1, MIROperand(stack_storage)); inst->set_operand(0, MIROperand(reg));
                instructions.insert(it, inst);
                op = reg; map.push_back(op);
                protect.insert(op.reg());
            };
            const auto def = [&](MIROperand op) {
                if (!isOperandVReg(op)) {
                    if (isOperandISAReg(op) && !ctx.registerInfo->is_zero_reg(op.reg()) && is_allocatableType(op.type())) {
                        protected_lockedISAReg.erase(op.reg());
                        if (auto it = phy_map.find(op.reg()); it != phy_map.end()) {
                            evict_reg(it->second);
                        }
                    }
                    return;
                }

                if (stack_map.count(op.reg())) dirty_regs.insert(op.reg());

                auto& map = get_datamap(op);
                MIROperand stack_storage;
                for (auto& reg : map) {
                    if (!isStackObject(reg.reg())) {
                        op = reg; map = { op }; protect.insert(op.reg());
                        return;
                    }
                    stack_storage = reg;
                }
                auto reg = get_free_reg(op); op = reg;
                map = { op }; protect.insert(op.reg());
            };
            const auto before_branch = [&]() {  // write back all out dirty vregs into stack slots before branch
                for(auto dirty : dirty_regs) {
                    if(liveIntervalInfo.outs.count(dirty)) evict_reg(MIROperand::asISAReg(dirty, ctx.registerInfo->getCanonicalizedRegisterType(dirty)));
                }
            };

            auto& inst = *it;
            auto& instInfo = ctx.instInfo.getInstInfo(inst);

            /*  */
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                auto flag = instInfo.operand_flag(idx);
                if ((flag & OperandFlagUse) || (flag & OperandFlagDef)) {  // 操作数一定为寄存器, 排除立即数的情况
                    auto op = inst->operand(idx);
                    if (!isOperandVReg(op) && isOperandISAReg(op)) {
                        protect.insert(op.reg());
                    }
                }
            }

            /* OperandFlagUse -- use(op) */
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                auto flag = instInfo.operand_flag(idx);
                if (flag & OperandFlagUse) {
                    auto op = inst->operand(idx);
                    use(op);
                }
            }
            
            /* InstFlagCall */
            if (requireFlag(instInfo.inst_flag(), InstFlagCall)) {
                std::vector<MIROperand> savedVRegs;
                const IPRAInfo* calleeUsage = nullptr;
                if (auto symbol = inst->operand(0).reloc()) {
                    calleeUsage = infoIPRA.query(symbol->name());
                }
                for (auto [p, v] : phy_map) {
                    auto phyRegister = MIROperand::asISAReg(p, OperandType::Special);
                    if (ctx.frameInfo.isCallerSaved(phyRegister)) {
                        /* 虽然是Caller Saved Registers, 但是在Callee Function中未用到该寄存器 */
                        if (calleeUsage && !calleeUsage->count(p)) continue;
                        savedVRegs.push_back(v);
                    }
                    // delete phyRegister;  // 内存空间管理
                }
                for (auto v : savedVRegs) evict_reg(v);

                protected_lockedISAReg.clear();
                collect_underRenamedISARegs(next);
            }
            protect.clear();
        
            /* release dead vregs */
            for (auto operand : releaseVRegs) {
                auto& map = get_datamap(operand);
                for (auto& reg : map) {
                    if (isISAReg(reg.reg())) {
                        phy_map.erase(reg.reg());
                        selector.markAsDiscarded(reg);
                    }
                }
                map.clear();
            }
        
            /* OperandFlagDef -- def(op) */
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                if (instInfo.operand_flag(idx) & OperandFlagDef) {
                    auto op = inst->operand(idx);
                    def(op);
                }
            }

            /* InstFlagBranch -- before_branch() */
            if (requireFlag(instInfo.inst_flag(), InstFlagBranch)) {
                before_branch();
            }

            it = next;
        }
        assert(block->verify(std::cerr, ctx));
    }
}
}  // namespace mir
