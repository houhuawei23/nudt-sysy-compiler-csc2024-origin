#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "mir/utils.hpp"
#include <queue>

namespace mir {

bool eliminateStackLoads(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}

bool removeIndirectCopy(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}

bool removeIdentityCopies(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}

bool removeUnusedInsts(MIRFunction& func, CodeGenContext& ctx) {
    /* writers: map from operand to list of insts that write it */
    std::unordered_map<MIROperand*, std::vector<MIRInst*>> writers;

    /** specail insts: that cant be removed,
     * 1 InstFlagSideEffect,
     * 2 def reg is allocable type */
    std::queue<MIRInst*> q;

    auto isAllocableType = [](OperandType type) {
        return type <= OperandType::Float32;
    };

    for (auto& block : func.blocks()) {
        for (auto& inst : block->insts()) {
            auto& instInfo = ctx.instInfo.get_instinfo(inst);
            bool special = false;

            if (requireOneFlag(instInfo.inst_flag(), InstFlagSideEffect)) {
                special = true;
            }

            for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
                if (instInfo.operand_flag(idx) & OperandFlagDef) {
                    auto op = inst->operand(idx);
                    writers[op].push_back(inst);
                    if (op->is_reg() && isISAReg(op->reg()) &&
                        isAllocableType(op->type())) {
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

        auto& instInfo = ctx.instInfo.get_instinfo(inst);
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
        if (isISAReg(op->reg()) && isAllocableType(op->type())) {
            continue;
        }
        for (auto& writer : writerList) {
            auto& instInfo = ctx.instInfo.get_instinfo(writer);
            if (requireOneFlag(instInfo.inst_flag(),
                               InstFlagSideEffect | InstFlagMultiDef)) {
                continue;
            }
            remove.insert(writer);
        }
    }

    for (auto& block : func.blocks()) {
        block->insts().remove_if(
            [&](MIRInst* inst) { return remove.count(inst); });
    }

    return not remove.empty();
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
    return false;
}

bool removeInvisibleInsts(MIRFunction& func, CodeGenContext& ctx) {
    return false;
}

bool genericPeepholeOpt(MIRFunction& func, CodeGenContext& ctx) {
    bool modified = false;
    modified |= eliminateStackLoads(func, ctx);
    modified |= removeIndirectCopy(func, ctx);
    modified |= removeIdentityCopies(func, ctx);
    modified |= removeUnusedInsts(func, ctx);
    modified |= applySSAPropagation(func, ctx);
    modified |= machineConstantCSE(func, ctx);
    modified |= machineConstantHoist(func, ctx);
    modified |= machineInstCSE(func, ctx);
    modified |= deadInstElimination(func, ctx);
    modified |= removeInvisibleInsts(func, ctx);
    // modified |= ctx.scheduleModel->peepholeOpt(func, ctx);
    return modified;
}

}  // namespace mir
