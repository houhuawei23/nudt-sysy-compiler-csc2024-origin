#include <queue>
#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "mir/utils.hpp"

namespace mir {
/*
 * @brief: eliminateStackLoads
 * @note: 
 *     消除无用的Load Register from Stack指令
 *     使用前面已经load之后的虚拟寄存器Copy指令来代替   
 *     NOTE: 后端属于SSA形式, 某一个虚拟寄存器不会被重复定义
 */
bool eliminateStackLoads(MIRFunction& mfunc, CodeGenContext& ctx) {
    /* Eliminate Stack Loads: 需要在寄存器分配之前进行优化处理  */
    if (!ctx.registerInfo || ctx.flags.preRA) return false;
    bool modified = false;
    for (auto& block : mfunc.blocks()) {
        auto& instructions = block->insts();

        uint32_t versionId = 0;
        std::unordered_map<MIROperand, uint32_t, MIROperandHasher> reg2Version;
        std::unordered_map<MIROperand, std::pair<MIROperand, uint32_t>, MIROperandHasher> stack2Reg;  // stack -> (reg, version)
        auto defReg = [&](MIROperand reg) { reg2Version[reg] = ++versionId; };

        for (auto inst : instructions) {
            if (inst->opcode() == InstStoreRegToStack) {
                auto& obj = inst->operand(0);
                auto& reg = inst->operand(1);
                if (auto iter = reg2Version.find(reg); iter != reg2Version.cend()) {
                    stack2Reg[obj] = { reg, iter->second };
                } else {
                    defReg(reg);
                    stack2Reg[obj] = { reg, versionId };
                }
            } else if (inst->opcode() == InstLoadRegFromStack) {
                auto& dst = inst->operand(0);
                auto& obj = inst->operand(1);
                if (auto iter = stack2Reg.find(obj); iter != stack2Reg.cend()) {
                    auto& [reg, ver] = stack2Reg[obj];
                    if (ver == reg2Version[reg]) {
                        // dst <- reg
                        inst->set_opcode(InstCopy);
                        obj = reg;
                        modified = true;
                    }
                }
            }

            auto& instInfo = ctx.instInfo.getInstInfo(inst);
            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                if (instInfo.operand_flag(idx) & OperandFlagDef) {
                    defReg(inst->operand(idx));
                }
            }

            /* NOTE: 但是物理寄存器可能会被重复定义, 此时需要更新物理寄存器相关的versionId */
            if (requireFlag(instInfo.inst_flag(), InstFlagCall)) {
                std::vector<MIROperand> nonVReg;
                for (auto [reg, ver] : reg2Version) {
                    if (isISAReg(reg.reg())) nonVReg.push_back(reg);
                }
                for (auto reg : nonVReg) defReg(reg);
            }

            /* NOTE: 更新 */
            if (inst->opcode() == InstLoadRegFromStack) {
                auto& dst = inst->operand(0);
                auto& obj = inst->operand(1);
                stack2Reg[obj] = { dst, reg2Version.at(dst) };
            }
        }
    }
    
    return modified;
}

// TODO
bool removeIndirectCopy(MIRFunction& mfunc, CodeGenContext& ctx) {
    return false;
}
bool removeIdentityCopies(MIRFunction& func, CodeGenContext& ctx) {
  bool modified = false;
  for (auto& block : func.blocks()) {
    block->insts().remove_if([&](MIRInst* inst) {
      MIROperand dst, src;
      ctx.instInfo.matchCopy(inst, dst, src);
      auto& info = ctx.instInfo.getInstInfo(inst);
      if (dst == src) {
        std::cerr << "remove identity copy: ";
        info.print(std::cerr, *inst, false);
        std::cerr << std::endl;
        modified = true;
        return true;
      }
      return false;
    });
  }
  return modified;
}
bool removeUnusedInsts(MIRFunction& func, CodeGenContext& ctx) {
    std::unordered_map<MIROperand, std::vector<MIRInst*>, MIROperandHasher> writers;

    /** specail insts: that cant be removed,
     * 1 InstFlagSideEffect,
     * 2 def reg is allocable type */
    std::queue<MIRInst*> q;

    auto isAllocableType = [](OperandType type) {
        return type <= OperandType::Float32;
    };

    for (auto& block : func.blocks()) {
        for (auto& inst : block->insts()) {
            auto& instInfo = ctx.instInfo.getInstInfo(inst);
            bool special = false;

            if (requireOneFlag(instInfo.inst_flag(), InstFlagSideEffect)) {
                special = true;
            }

            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                if (instInfo.operand_flag(idx) & OperandFlagDef) {
                    auto op = inst->operand(idx);
                    writers[op].push_back(inst);
                    if (op.isReg() && isISAReg(op.reg()) &&
                        isAllocableType(op.type())) {
                        special = true;
                    }
                }
            }
            if (special) {
                q.push(inst);
            }
        }
    }

    /*
    i1: sub b[Def], 0[Metadata], 1[Metadata]
    i2: add b[Def], 1[Metadata], 2[Metadata]
    i3: store a[Def], b[Use]

    q0 = {i1, i2, i3}
    writers: {
        a: {i3},
        b: {i1, i2},
    }

    inst = i1, pop i1, q = {i2, i3}
    writers not changed

    inst = i2, pop i2, q = {i3}
    writers not changed

    inst = i3, pop i3, q = {}
    writers[b] = {i1, i2}, q.push(i1, i2), q = {i1, i2}, writers erase b
    writers = { a: {i3} }

    inst = i1, pop i1, q = {i2}
    writers not changed

    inst = i2, pop i2, q = {}
    writers not changed

    finieshed, writers = { a: {i3} }

    */
    /*
    q has insts that cant be removed and not yet processed,
    if inst i cant be removed, then the insts define i's operand also cant be
    removed, so remove i's operand from writers, and add insts that define i's
    operand to q.
    */
    while (not q.empty()) {
        auto& inst = q.front();
        q.pop();

        auto& instInfo = ctx.instInfo.getInstInfo(inst);
        for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
            if (instInfo.operand_flag(idx) & OperandFlagUse) {
                /* if operand is used, remove from writers */
                auto op = inst->operand(idx);
                if (auto iter = writers.find(op); iter != writers.end()) {
                    for (auto& writer : iter->second) {
                        q.push(writer);
                    }
                    writers.erase(iter);
                }
            }
        }
    }
    /* after while, writers contain operands that are not used */

    std::unordered_set<MIRInst*> remove;
    for (auto& [op, writerList] : writers) {
        if (isISAReg(op.reg()) && isAllocableType(op.type())) {
            continue;
        }
        for (auto& writer : writerList) {
            auto& instInfo = ctx.instInfo.getInstInfo(writer);
            if (requireOneFlag(instInfo.inst_flag(),
                               InstFlagSideEffect | InstFlagMultiDef)) {
                continue;
            }
            remove.insert(writer);
        }
    }

    for (auto& block : func.blocks()) {
        block->insts().remove_if([&](MIRInst* inst) { return remove.count(inst); });
    }

    if(not remove.empty()) {
        int a = 12;
        int b = remove.size();
        std::cout << "removed " << b << " insts" << std::endl;
    }

    return !remove.empty();
}
bool applySSAPropagation(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}
bool machineConstantCSE(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}
bool machineConstantHoist(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}
bool machineInstCSE(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}
bool deadInstElimination(MIRFunction& func, CodeGenContext& ctx) {
    bool modified = false;
    for (auto& block : func.blocks()) {
        auto& instructions = block->insts();
        std::unordered_map<MIROperand, uint32_t, MIROperandHasher> version;
        uint32_t versionIdx = 0;

        auto getVersion = [&](MIROperand& op) ->uint32_t {
            if (!isOperandVRegORISAReg(op)) return 0;
            if (auto iter = version.find(op); iter != version.cend()) return iter->second;
            return version[op] = ++versionIdx;
        };
    }
    
    return modified;
}
bool removeInvisibleInsts(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}
bool genericPeepholeOpt(MIRFunction& func, CodeGenContext& ctx) {
    bool modified = false;
    modified |= eliminateStackLoads(func, ctx);
    // modified |= removeIndirectCopy(func, ctx);
    // modified |= removeIdentityCopies(func, ctx);
    // modified |= removeUnusedInsts(func, ctx);
    // modified |= applySSAPropagation(func, ctx);
    // modified |= machineConstantCSE(func, ctx);
    // modified |= machineConstantHoist(func, ctx);
    // modified |= machineInstCSE(func, ctx);
    // modified |= deadInstElimination(func, ctx);
    // modified |= removeInvisibleInsts(func, ctx);
    // modified |= ctx.scheduleModel->peepholeOpt(func, ctx);
    return modified;
}
}