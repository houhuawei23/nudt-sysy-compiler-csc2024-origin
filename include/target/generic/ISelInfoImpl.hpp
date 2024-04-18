#pragma once

#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "mir/iselinfo.hpp"
#include "target/generic/InstInfoDecl.hpp"
#include "target/generic/ISelInfoDecl.hpp"

GENERIC_NAMESPACE_BEGIN

// matchInstXXXX
// static bool matchInstAdd() {

// }
static bool matchInstLoad(MIRInst* inst,
                          MIROperand* dst,
                          MIROperand* lhs,
                          MIROperand* rhs) {
    if (inst->opcode() != InstLoad) {
        return false;
    }
    dst = inst->operand(0);
    lhs = inst->operand(1);
    rhs = inst->operand(2);
    return true;
}

static bool matchAndSelectPattern1(MIRInst* inst, ISelContext& ctx) {
    uint32_t opcode = inst->opcode();
    /* match pattern */
    MIROperand* op1;
    MIROperand* op2;
    MIROperand* op3;
    if(not matchInstLoad(inst, op1, op2, op3)){
        return false;
    }
    // if(not)

    /* select inst */
    auto op4 = op1;
    auto op5 = op2;
    auto op6 = op3;
    // ctx.new_inst
    auto new_inst = new MIRInst(Load);
    new_inst->set_operand(0, op4);
    new_inst->set_operand(1, op5);
    new_inst->set_operand(2, op6);
    
    ctx.insert_inst(new_inst);
    ctx.replace_operand(ctx.get_inst_def(inst), ctx.get_inst_def(new_inst));
    ctx.remove_inst(inst);
    return true;
}

static bool matchAndSelectImpl(MIRInst* inst, ISelContext& ctx) {
    std::cout << "matchAndSelectImpl: " << inst->opcode() << std::endl;
    switch (inst->opcode()) {
        case InstLoad: {
            if (matchAndSelectPattern1(inst, ctx))
                return true;
            break;
        }

        default:
            break;
    }
    std::cout << "matchAndSelectImpl: not matched" << std::endl;
    return false;
}

class GENERICISelInfo final : public TargetISelInfo {
   public:
    bool is_legal_geninst(uint32_t opcode) const override;
    bool match_select(MIRInst* inst, ISelContext& ctx) const override;
};

TargetISelInfo& getGENERICISelInfo() {
    static GENERICISelInfo iselInfo;
    return iselInfo;
}

GENERIC_NAMESPACE_END
