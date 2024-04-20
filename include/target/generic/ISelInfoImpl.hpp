// Automatically generated file, do not edit!

#pragma once
#include "mir/mir.hpp"
#include "mir/iselinfo.hpp"
#include "target/generic/ISelInfoDecl.hpp"

GENERIC_NAMESPACE_BEGIN

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
                                       MIROperand*& reloc) {
    if (inst->opcode() != InstLoadGlobalAddress)
        return false;
    dst = inst->operand(0);
    reloc = inst->operand(1);
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

static bool matchInstReturn(MIRInst* inst

) {
    if (inst->opcode() != InstReturn)
        return false;

    return true;
}

/* InstAdd matchAndSelectPatternInstAdd begin */
static bool matchAndSelectPattern1(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstAdd;
    /* Match Inst */

    /* match inst InstAdd */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstAdd(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst Add */
    auto inst2 = new MIRInst(Add);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstAdd success!" << std::endl;
    return true;
}

/* InstAdd matchAndSelectPatternInstAddend */

/* InstSub matchAndSelectPatternInstSub begin */
static bool matchAndSelectPattern2(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstSub;
    /* Match Inst */

    /* match inst InstSub */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstSub(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst Sub */
    auto inst2 = new MIRInst(Sub);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstSub success!" << std::endl;
    return true;
}

/* InstSub matchAndSelectPatternInstSubend */

/* InstMul matchAndSelectPatternInstMul begin */
static bool matchAndSelectPattern3(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstMul;
    /* Match Inst */

    /* match inst InstMul */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstMul(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst Mul */
    auto inst2 = new MIRInst(Mul);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstMul success!" << std::endl;
    return true;
}

/* InstMul matchAndSelectPatternInstMulend */

/* InstUDiv matchAndSelectPatternInstUDiv begin */
static bool matchAndSelectPattern4(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstUDiv;
    /* Match Inst */

    /* match inst InstUDiv */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstUDiv(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst UDiv */
    auto inst2 = new MIRInst(UDiv);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstUDiv success!" << std::endl;
    return true;
}

/* InstUDiv matchAndSelectPatternInstUDivend */

/* InstURem matchAndSelectPatternInstURem begin */
static bool matchAndSelectPattern5(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstURem;
    /* Match Inst */

    /* match inst InstURem */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstURem(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst URem */
    auto inst2 = new MIRInst(URem);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstURem success!" << std::endl;
    return true;
}

/* InstURem matchAndSelectPatternInstURemend */

/* InstAnd matchAndSelectPatternInstAnd begin */
static bool matchAndSelectPattern6(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstAnd;
    /* Match Inst */

    /* match inst InstAnd */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstAnd(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst And */
    auto inst2 = new MIRInst(And);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstAnd success!" << std::endl;
    return true;
}

/* InstAnd matchAndSelectPatternInstAndend */

/* InstOr matchAndSelectPatternInstOr begin */
static bool matchAndSelectPattern7(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstOr;
    /* Match Inst */

    /* match inst InstOr */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstOr(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst Or */
    auto inst2 = new MIRInst(Or);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstOr success!" << std::endl;
    return true;
}

/* InstOr matchAndSelectPatternInstOrend */

/* InstXor matchAndSelectPatternInstXor begin */
static bool matchAndSelectPattern8(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstXor;
    /* Match Inst */

    /* match inst InstXor */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstXor(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst Xor */
    auto inst2 = new MIRInst(Xor);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstXor success!" << std::endl;
    return true;
}

/* InstXor matchAndSelectPatternInstXorend */

/* InstShl matchAndSelectPatternInstShl begin */
static bool matchAndSelectPattern9(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstShl;
    /* Match Inst */

    /* match inst InstShl */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstShl(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst Shl */
    auto inst2 = new MIRInst(Shl);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstShl success!" << std::endl;
    return true;
}

/* InstShl matchAndSelectPatternInstShlend */

/* InstLShr matchAndSelectPatternInstLShr begin */
static bool matchAndSelectPattern10(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstLShr;
    /* Match Inst */

    /* match inst InstLShr */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstLShr(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst LShr */
    auto inst2 = new MIRInst(LShr);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstLShr success!" << std::endl;
    return true;
}

/* InstLShr matchAndSelectPatternInstLShrend */

/* InstAShr matchAndSelectPatternInstAShr begin */
static bool matchAndSelectPattern11(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstAShr;
    /* Match Inst */

    /* match inst InstAShr */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstAShr(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst AShr */
    auto inst2 = new MIRInst(AShr);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstAShr success!" << std::endl;
    return true;
}

/* InstAShr matchAndSelectPatternInstAShrend */

/* InstSMin matchAndSelectPatternInstSMin begin */
static bool matchAndSelectPattern12(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstSMin;
    /* Match Inst */

    /* match inst InstSMin */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstSMin(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst SMin */
    auto inst2 = new MIRInst(SMin);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstSMin success!" << std::endl;
    return true;
}

/* InstSMin matchAndSelectPatternInstSMinend */

/* InstSMax matchAndSelectPatternInstSMax begin */
static bool matchAndSelectPattern13(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstSMax;
    /* Match Inst */

    /* match inst InstSMax */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstSMax(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst SMax */
    auto inst2 = new MIRInst(SMax);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstSMax success!" << std::endl;
    return true;
}

/* InstSMax matchAndSelectPatternInstSMaxend */

/* InstReturn matchAndSelectPatternInstReturn begin */
static bool matchAndSelectPattern14(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstReturn;
    /* Match Inst */

    /* match inst InstReturn */

    if (not matchInstReturn(inst1)) {
        return false;
    }

    /** Select Inst **/
    /* select inst Return */
    auto inst2 = new MIRInst(Return);

    ctx.insert_inst(inst2);
    ctx.remove_inst(inst1);
    std::cout << "match and select InstReturn success!" << std::endl;
    return true;
}

/* InstReturn matchAndSelectPatternInstReturnend */

/* InstLoad matchAndSelectPatternInstLoad begin */
static bool matchAndSelectPattern15(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstLoad;
    /* Match Inst */

    /* match inst InstLoad */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstLoad(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst Load */
    auto inst2 = new MIRInst(Load);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstLoad success!" << std::endl;
    return true;
}

/* InstLoad matchAndSelectPatternInstLoadend */

/* InstStore matchAndSelectPatternInstStore begin */
static bool matchAndSelectPattern16(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstStore;
    /* Match Inst */

    /* match inst InstStore */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstStore(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst Store */
    auto inst2 = new MIRInst(Store);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    ctx.remove_inst(inst1);
    std::cout << "match and select InstStore success!" << std::endl;
    return true;
}

/* InstStore matchAndSelectPatternInstStoreend */

/* InstJump matchAndSelectPatternInstJump begin */
static bool matchAndSelectPattern17(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstJump;
    /* Match Inst */

    /* match inst InstJump */
    MIROperand* operand1 = nullptr;
    if (not matchInstJump(inst1, operand1)) {
        return false;
    }

    /** Select Inst **/
    auto operand2 = (operand1);
    /* select inst Jump */
    auto inst2 = new MIRInst(Jump);
    inst2->set_operand(0, operand2);

    ctx.insert_inst(inst2);
    ctx.remove_inst(inst1);
    std::cout << "match and select InstJump success!" << std::endl;
    return true;
}

/* InstJump matchAndSelectPatternInstJumpend */

/* InstBranch matchAndSelectPatternInstBranch begin */
static bool matchAndSelectPattern18(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstBranch;
    /* Match Inst */

    /* match inst InstBranch */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    MIROperand* operand3 = nullptr;
    if (not matchInstBranch(inst1, operand1, operand2, operand3)) {
        return false;
    }

    /** Select Inst **/
    auto operand4 = (operand1);
    auto operand5 = (operand2);
    auto operand6 = (operand3);
    /* select inst Branch */
    auto inst2 = new MIRInst(Branch);
    inst2->set_operand(0, operand4);
    inst2->set_operand(1, operand5);
    inst2->set_operand(2, operand6);

    ctx.insert_inst(inst2);
    ctx.remove_inst(inst1);
    std::cout << "match and select InstBranch success!" << std::endl;
    return true;
}

/* InstBranch matchAndSelectPatternInstBranchend */

/* InstCopy matchAndSelectPatternInstCopy begin */
static bool matchAndSelectPattern19(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstCopy;
    /* Match Inst */

    /* match inst InstCopy */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    if (not matchInstCopy(inst1, operand1, operand2)) {
        return false;
    }

    /** Select Inst **/
    auto operand3 = (operand1);
    auto operand4 = (operand2);
    /* select inst Copy */
    auto inst2 = new MIRInst(Copy);
    inst2->set_operand(0, operand3);
    inst2->set_operand(1, operand4);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstCopy success!" << std::endl;
    return true;
}

/* InstCopy matchAndSelectPatternInstCopyend */

/* InstLoadGlobalAddress matchAndSelectPatternInstLoadGlobalAddress begin */
static bool matchAndSelectPattern20(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstLoadGlobalAddress;
    /* Match Inst */

    /* match inst InstLoadGlobalAddress */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    if (not matchInstLoadGlobalAddress(inst1, operand1, operand2)) {
        return false;
    }

    /** Select Inst **/
    auto operand3 = (operand1);
    auto operand4 = (operand2);
    /* select inst LoadGlobalAddress */
    auto inst2 = new MIRInst(LoadGlobalAddress);
    inst2->set_operand(0, operand3);
    inst2->set_operand(1, operand4);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstLoadGlobalAddress success!" << std::endl;
    return true;
}

/* InstLoadGlobalAddress matchAndSelectPatternInstLoadGlobalAddressend */

/* InstLoadImm matchAndSelectPatternInstLoadImm begin */
static bool matchAndSelectPattern21(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstLoadImm;
    /* Match Inst */

    /* match inst InstLoadImm */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    if (not matchInstLoadImm(inst1, operand1, operand2)) {
        return false;
    }

    /** Select Inst **/
    auto operand3 = (operand1);
    auto operand4 = (operand2);
    /* select inst LoadImm */
    auto inst2 = new MIRInst(LoadImm);
    inst2->set_operand(0, operand3);
    inst2->set_operand(1, operand4);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstLoadImm success!" << std::endl;
    return true;
}

/* InstLoadImm matchAndSelectPatternInstLoadImmend */

/* InstLoadStackObjectAddr matchAndSelectPatternInstLoadStackObjectAddr begin */
static bool matchAndSelectPattern22(MIRInst* inst1, ISelContext& ctx) {
    uint32_t opcode = InstLoadStackObjectAddr;
    /* Match Inst */

    /* match inst InstLoadStackObjectAddr */
    MIROperand* operand1 = nullptr;
    MIROperand* operand2 = nullptr;
    if (not matchInstLoadStackObjectAddr(inst1, operand1, operand2)) {
        return false;
    }

    /** Select Inst **/
    auto operand3 = (operand1);
    auto operand4 = (operand2);
    /* select inst LoadStackObjectAddr */
    auto inst2 = new MIRInst(LoadStackObjectAddr);
    inst2->set_operand(0, operand3);
    inst2->set_operand(1, operand4);

    ctx.insert_inst(inst2);
    /* Replace Operand */
    ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
    ctx.remove_inst(inst1);
    std::cout << "match and select InstLoadStackObjectAddr success!"
              << std::endl;
    return true;
}

/* InstLoadStackObjectAddr matchAndSelectPatternInstLoadStackObjectAddrend */

static bool matchAndSelectImpl(MIRInst* inst, ISelContext& ctx) {
    switch (inst->opcode()) {
        case InstAdd: {
            if (matchAndSelectPattern1(inst, ctx))
                return true;
            break;
        }
        case InstSub: {
            if (matchAndSelectPattern2(inst, ctx))
                return true;
            break;
        }
        case InstMul: {
            if (matchAndSelectPattern3(inst, ctx))
                return true;
            break;
        }
        case InstUDiv: {
            if (matchAndSelectPattern4(inst, ctx))
                return true;
            break;
        }
        case InstURem: {
            if (matchAndSelectPattern5(inst, ctx))
                return true;
            break;
        }
        case InstAnd: {
            if (matchAndSelectPattern6(inst, ctx))
                return true;
            break;
        }
        case InstOr: {
            if (matchAndSelectPattern7(inst, ctx))
                return true;
            break;
        }
        case InstXor: {
            if (matchAndSelectPattern8(inst, ctx))
                return true;
            break;
        }
        case InstShl: {
            if (matchAndSelectPattern9(inst, ctx))
                return true;
            break;
        }
        case InstLShr: {
            if (matchAndSelectPattern10(inst, ctx))
                return true;
            break;
        }
        case InstAShr: {
            if (matchAndSelectPattern11(inst, ctx))
                return true;
            break;
        }
        case InstSMin: {
            if (matchAndSelectPattern12(inst, ctx))
                return true;
            break;
        }
        case InstSMax: {
            if (matchAndSelectPattern13(inst, ctx))
                return true;
            break;
        }
        case InstReturn: {
            if (matchAndSelectPattern14(inst, ctx))
                return true;
            break;
        }
        case InstLoad: {
            if (matchAndSelectPattern15(inst, ctx))
                return true;
            break;
        }
        case InstStore: {
            if (matchAndSelectPattern16(inst, ctx))
                return true;
            break;
        }
        case InstJump: {
            if (matchAndSelectPattern17(inst, ctx))
                return true;
            break;
        }
        case InstBranch: {
            if (matchAndSelectPattern18(inst, ctx))
                return true;
            break;
        }
        case InstCopy: {
            if (matchAndSelectPattern19(inst, ctx))
                return true;
            break;
        }
        case InstLoadGlobalAddress: {
            if (matchAndSelectPattern20(inst, ctx))
                return true;
            break;
        }
        case InstLoadImm: {
            if (matchAndSelectPattern21(inst, ctx))
                return true;
            break;
        }
        case InstLoadStackObjectAddr: {
            if (matchAndSelectPattern22(inst, ctx))
                return true;
            break;
        }
        default:
            break;
    }
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