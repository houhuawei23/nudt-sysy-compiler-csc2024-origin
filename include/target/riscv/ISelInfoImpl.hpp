// Automatically generated file, do not edit!

#pragma once
#include "mir/mir.hpp"
#include "mir/iselinfo.hpp"
#include "target/riscv/riscviselinfo.hpp"
#include "target/riscv/ISelInfoDecl.hpp"

RISCV_NAMESPACE_BEGIN

static bool matchInstJump(MIRInst* inst, MIROperand*& label) {
    if (inst->opcode() != InstJump)
        return false;
    label = inst->operand(0);
    return true;
}

static bool matchInstBranch(MIRInst* inst,
                            MIROperand*& cond,
                            MIROperand*& thenb,
                            MIROperand*& elseb) {
    if (inst->opcode() != InstBranch)
        return false;
    cond = inst->operand(0);
    thenb = inst->operand(1);
    elseb = inst->operand(2);
    return true;
}

static bool matchInstLoad(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& addr,
                          MIROperand*& align) {
    if (inst->opcode() != InstLoad)
        return false;
    dst = inst->operand(0);
    addr = inst->operand(1);
    align = inst->operand(2);
    return true;
}

static bool matchInstStore(MIRInst* inst,
                           MIROperand*& addr,
                           MIROperand*& src,
                           MIROperand*& align) {
    if (inst->opcode() != InstStore)
        return false;
    addr = inst->operand(0);
    src = inst->operand(1);
    align = inst->operand(2);
    return true;
}

static bool matchInstAdd(MIRInst* inst,
                         MIROperand*& dst,
                         MIROperand*& src1,
                         MIROperand*& src2) {
    if (inst->opcode() != InstAdd)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstSub(MIRInst* inst,
                         MIROperand*& dst,
                         MIROperand*& src1,
                         MIROperand*& src2) {
    if (inst->opcode() != InstSub)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstMul(MIRInst* inst,
                         MIROperand*& dst,
                         MIROperand*& src1,
                         MIROperand*& src2) {
    if (inst->opcode() != InstMul)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstUDiv(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
    if (inst->opcode() != InstUDiv)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstURem(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
    if (inst->opcode() != InstURem)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstAnd(MIRInst* inst,
                         MIROperand*& dst,
                         MIROperand*& src1,
                         MIROperand*& src2) {
    if (inst->opcode() != InstAnd)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstOr(MIRInst* inst,
                        MIROperand*& dst,
                        MIROperand*& src1,
                        MIROperand*& src2) {
    if (inst->opcode() != InstOr)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstXor(MIRInst* inst,
                         MIROperand*& dst,
                         MIROperand*& src1,
                         MIROperand*& src2) {
    if (inst->opcode() != InstXor)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstShl(MIRInst* inst,
                         MIROperand*& dst,
                         MIROperand*& src1,
                         MIROperand*& src2) {
    if (inst->opcode() != InstShl)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstLShr(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
    if (inst->opcode() != InstLShr)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstAShr(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
    if (inst->opcode() != InstAShr)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstSMin(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
    if (inst->opcode() != InstSMin)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstSMax(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
    if (inst->opcode() != InstSMax)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstNeg(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
    if (inst->opcode() != InstNeg)
        return false;
    dst = inst->operand(0);
    src = inst->operand(1);
    return true;
}

static bool matchInstAbs(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
    if (inst->opcode() != InstAbs)
        return false;
    dst = inst->operand(0);
    src = inst->operand(1);
    return true;
}

static bool matchInstFAdd(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
    if (inst->opcode() != InstFAdd)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstFSub(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
    if (inst->opcode() != InstFSub)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstFMul(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
    if (inst->opcode() != InstFMul)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstFDiv(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
    if (inst->opcode() != InstFDiv)
        return false;
    dst = inst->operand(0);
    src1 = inst->operand(1);
    src2 = inst->operand(2);
    return true;
}

static bool matchInstSExt(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
    if (inst->opcode() != InstSExt)
        return false;
    dst = inst->operand(0);
    src = inst->operand(1);
    return true;
}

static bool matchInstZExt(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
    if (inst->opcode() != InstZExt)
        return false;
    dst = inst->operand(0);
    src = inst->operand(1);
    return true;
}

static bool matchInstTrunc(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
    if (inst->opcode() != InstTrunc)
        return false;
    dst = inst->operand(0);
    src = inst->operand(1);
    return true;
}

static bool matchInstF2U(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
    if (inst->opcode() != InstF2U)
        return false;
    dst = inst->operand(0);
    src = inst->operand(1);
    return true;
}

static bool matchInstF2S(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
    if (inst->opcode() != InstF2S)
        return false;
    dst = inst->operand(0);
    src = inst->operand(1);
    return true;
}

static bool matchInstU2F(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
    if (inst->opcode() != InstU2F)
        return false;
    dst = inst->operand(0);
    src = inst->operand(1);
    return true;
}

static bool matchInstS2F(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
    if (inst->opcode() != InstS2F)
        return false;
    dst = inst->operand(0);
    src = inst->operand(1);
    return true;
}

static bool matchInstFCast(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
    if (inst->opcode() != InstFCast)
        return false;
    dst = inst->operand(0);
    src = inst->operand(1);
    return true;
}

static bool matchInstCopy(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
    if (inst->opcode() != InstCopy)
        return false;
    dst = inst->operand(0);
    src = inst->operand(1);
    return true;
}

static bool matchInstLoadGlobalAddress(MIRInst* inst,
                                       MIROperand*& dst,
                                       MIROperand*& addr) {
    if (inst->opcode() != InstLoadGlobalAddress)
        return false;
    dst = inst->operand(0);
    addr = inst->operand(1);
    return true;
}

static bool matchInstLoadImm(MIRInst* inst,
                             MIROperand*& dst,
                             MIROperand*& imm) {
    if (inst->opcode() != InstLoadImm)
        return false;
    dst = inst->operand(0);
    imm = inst->operand(1);
    return true;
}

static bool matchInstLoadStackObjectAddr(MIRInst* inst,
                                         MIROperand*& dst,
                                         MIROperand*& stack_obj) {
    if (inst->opcode() != InstLoadStackObjectAddr)
        return false;
    dst = inst->operand(0);
    stack_obj = inst->operand(1);
    return true;
}

static bool matchInstReturn(MIRInst* inst) {
    if (inst->opcode() != InstReturn)
        return false;

    return true;
}

/* InstLoad matchAndSelectPatternInstLoad begin */
static bool matchAndSelectPattern1(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstLoad;
    /** Match Inst **/
    /* match inst InstLoad */
    MIROperand* op1 = nullptr;
    MIROperand* op2 = nullptr;
    MIROperand* op3 = nullptr;
    if (not matchInstLoad(inst1, op1, op2, op3)) {
        return false;
    }

    /* match predicate for operands  */
    MIROperand* op4 = nullptr;
    MIROperand* op5 = nullptr;
    if (not(selectAddrOffset(op2, ctx, op4, op5))) {
        return false;
    }

    /** Select Inst **/
    auto op6 = (op1);
    auto op7 = (op4);
    auto op8 = (op5);
    /* select inst getLoadOpcode(op1) */
    auto inst2 = new MIRInst(getLoadOpcode(op1));
    inst2->set_operand(0, op6);
    inst2->set_operand(1, op7);
    inst2->set_operand(2, op8);
    ctx.insert_inst(inst2);

    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "    InstLoad success!" << std::endl;
    return true;
}

/* InstLoad matchAndSelectPatternInstLoadend */

/* InstStore matchAndSelectPatternInstStore begin */
static bool matchAndSelectPattern2(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstStore;
    /** Match Inst **/
    /* match inst InstStore */
    MIROperand* op1 = nullptr;
    MIROperand* op2 = nullptr;
    MIROperand* op3 = nullptr;
    if (not matchInstStore(inst1, op1, op2, op3)) {
        return false;
    }

    /* match predicate for operands  */
    MIROperand* op4 = nullptr;
    MIROperand* op5 = nullptr;
    if (not(isOperandVRegOrISAReg(op2) &&
            selectAddrOffset(op1, ctx, op4, op5))) {
        return false;
    }

    /** Select Inst **/
    auto op6 = (op2);
    auto op7 = (op4);
    auto op8 = (op5);
    /* select inst getStoreOpcode(op2) */
    auto inst2 = new MIRInst(getStoreOpcode(op2));
    inst2->set_operand(0, op6);
    inst2->set_operand(1, op8);
    inst2->set_operand(2, op7);
    ctx.insert_inst(inst2);

    ctx.remove_inst(inst1);
    std::cout << "    InstStore success!" << std::endl;
    return true;
}

/* InstStore matchAndSelectPatternInstStoreend */

/* InstJump matchAndSelectPatternInstJump begin */
static bool matchAndSelectPattern3(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstJump;
    /** Match Inst **/
    /* match inst InstJump */
    MIROperand* op1 = nullptr;
    if (not matchInstJump(inst1, op1)) {
        return false;
    }

    /** Select Inst **/
    auto op2 = (op1);
    /* select inst J */
    auto inst2 = new MIRInst(J);
    inst2->set_operand(0, op2);
    ctx.insert_inst(inst2);

    ctx.remove_inst(inst1);
    std::cout << "    InstJump success!" << std::endl;
    return true;
}

/* InstJump matchAndSelectPatternInstJumpend */

/* InstReturn matchAndSelectPatternInstReturn begin */
static bool matchAndSelectPattern4(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstReturn;
    /** Match Inst **/
    /* match inst InstReturn */

    if (not matchInstReturn(inst1)) {
        return false;
    }

    /** Select Inst **/
    /* select inst RET */
    auto inst2 = new MIRInst(RET);
    ctx.insert_inst(inst2);

    ctx.remove_inst(inst1);
    std::cout << "    InstReturn success!" << std::endl;
    return true;
}

/* InstReturn matchAndSelectPatternInstReturnend */

static bool matchAndSelectImpl(MIRInst* inst, ISelContext& ctx) {
    switch (inst->opcode()) {
        case InstLoad: {
            if (matchAndSelectPattern1(inst, ctx))
                return true;
            break;
        }
        case InstStore: {
            if (matchAndSelectPattern2(inst, ctx))
                return true;
            break;
        }
        case InstJump: {
            if (matchAndSelectPattern3(inst, ctx))
                return true;
            break;
        }
        case InstReturn: {
            if (matchAndSelectPattern4(inst, ctx))
                return true;
            break;
        }
        default:
            break;
    }
    return false;
}

class RISCVISelInfo final : public TargetISelInfo {
   public:
    bool is_legal_geninst(uint32_t opcode) const override;
    bool match_select(MIRInst* inst, ISelContext& ctx) const override;
};

TargetISelInfo& getRISCVISelInfo() {
    static RISCVISelInfo iselInfo;
    return iselInfo;
}

RISCV_NAMESPACE_END