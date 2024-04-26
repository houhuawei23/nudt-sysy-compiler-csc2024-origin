#include "mir/mir.hpp"
#include "mir/utils.hpp"

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
    return false;
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
    modified |= ctx.scheduleModel->peepholeOpt(func, ctx);
    return modified;
}

}  // namespace mir
