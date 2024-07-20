// Automatically generated file, do not edit!

#include "autogen/riscv/ISelInfoDecl.hpp"

RISCV_NAMESPACE_BEGIN

static bool matchInstJump(MIRInst* inst, MIROperand*& target) {
  if (inst->opcode() != InstJump) return false;
  target = inst->operand(0);
  return true;
}

static bool matchInstBranch(MIRInst* inst,
                            MIROperand*& cond,
                            MIROperand*& target,
                            MIROperand*& prob) {
  if (inst->opcode() != InstBranch) return false;
  cond = inst->operand(0);
  target = inst->operand(1);
  prob = inst->operand(2);
  return true;
}

static bool matchInstUnreachable(MIRInst* inst) {
  if (inst->opcode() != InstUnreachable) return false;

  return true;
}

static bool matchInstLoad(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& addr,
                          MIROperand*& align) {
  if (inst->opcode() != InstLoad) return false;
  dst = inst->operand(0);
  addr = inst->operand(1);
  align = inst->operand(2);
  return true;
}

static bool matchInstStore(MIRInst* inst,
                           MIROperand*& addr,
                           MIROperand*& src,
                           MIROperand*& align) {
  if (inst->opcode() != InstStore) return false;
  addr = inst->operand(0);
  src = inst->operand(1);
  align = inst->operand(2);
  return true;
}

static bool matchInstAdd(MIRInst* inst,
                         MIROperand*& dst,
                         MIROperand*& src1,
                         MIROperand*& src2) {
  if (inst->opcode() != InstAdd) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstSub(MIRInst* inst,
                         MIROperand*& dst,
                         MIROperand*& src1,
                         MIROperand*& src2) {
  if (inst->opcode() != InstSub) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstMul(MIRInst* inst,
                         MIROperand*& dst,
                         MIROperand*& src1,
                         MIROperand*& src2) {
  if (inst->opcode() != InstMul) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstUDiv(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
  if (inst->opcode() != InstUDiv) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstURem(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
  if (inst->opcode() != InstURem) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstAnd(MIRInst* inst,
                         MIROperand*& dst,
                         MIROperand*& src1,
                         MIROperand*& src2) {
  if (inst->opcode() != InstAnd) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstOr(MIRInst* inst,
                        MIROperand*& dst,
                        MIROperand*& src1,
                        MIROperand*& src2) {
  if (inst->opcode() != InstOr) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstXor(MIRInst* inst,
                         MIROperand*& dst,
                         MIROperand*& src1,
                         MIROperand*& src2) {
  if (inst->opcode() != InstXor) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstShl(MIRInst* inst,
                         MIROperand*& dst,
                         MIROperand*& src1,
                         MIROperand*& src2) {
  if (inst->opcode() != InstShl) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstLShr(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
  if (inst->opcode() != InstLShr) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstAShr(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
  if (inst->opcode() != InstAShr) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstSDiv(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
  if (inst->opcode() != InstSDiv) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstSRem(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
  if (inst->opcode() != InstSRem) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstSMin(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
  if (inst->opcode() != InstSMin) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstSMax(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
  if (inst->opcode() != InstSMax) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstNeg(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
  if (inst->opcode() != InstNeg) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstAbs(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
  if (inst->opcode() != InstAbs) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstFAdd(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
  if (inst->opcode() != InstFAdd) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstFSub(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
  if (inst->opcode() != InstFSub) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstFMul(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
  if (inst->opcode() != InstFMul) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstFDiv(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2) {
  if (inst->opcode() != InstFDiv) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  return true;
}

static bool matchInstFNeg(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
  if (inst->opcode() != InstFNeg) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstFAbs(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
  if (inst->opcode() != InstFAbs) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstFFma(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2,
                          MIROperand*& acc) {
  if (inst->opcode() != InstFFma) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  acc = inst->operand(3);
  return true;
}

static bool matchInstICmp(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2,
                          MIROperand*& op) {
  if (inst->opcode() != InstICmp) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  op = inst->operand(3);
  return true;
}

static bool matchInstFCmp(MIRInst* inst,
                          MIROperand*& dst,
                          MIROperand*& src1,
                          MIROperand*& src2,
                          MIROperand*& op) {
  if (inst->opcode() != InstFCmp) return false;
  dst = inst->operand(0);
  src1 = inst->operand(1);
  src2 = inst->operand(2);
  op = inst->operand(3);
  return true;
}

static bool matchInstSExt(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
  if (inst->opcode() != InstSExt) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstZExt(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
  if (inst->opcode() != InstZExt) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstTrunc(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
  if (inst->opcode() != InstTrunc) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstF2U(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
  if (inst->opcode() != InstF2U) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstF2S(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
  if (inst->opcode() != InstF2S) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstU2F(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
  if (inst->opcode() != InstU2F) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstS2F(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
  if (inst->opcode() != InstS2F) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstFCast(MIRInst* inst, MIROperand*& dst) {
  if (inst->opcode() != InstFCast) return false;
  dst = inst->operand(0);
  return true;
}

static bool matchInstCopy(MIRInst* inst, MIROperand*& dst, MIROperand*& src) {
  if (inst->opcode() != InstCopy) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstSelect(MIRInst* inst,
                            MIROperand*& dst,
                            MIROperand*& cond,
                            MIROperand*& src1,
                            MIROperand*& src2) {
  if (inst->opcode() != InstSelect) return false;
  dst = inst->operand(0);
  cond = inst->operand(1);
  src1 = inst->operand(2);
  src2 = inst->operand(3);
  return true;
}

static bool matchInstLoadGlobalAddress(MIRInst* inst,
                                       MIROperand*& dst,
                                       MIROperand*& addr) {
  if (inst->opcode() != InstLoadGlobalAddress) return false;
  dst = inst->operand(0);
  addr = inst->operand(1);
  return true;
}

static bool matchInstLoadImm(MIRInst* inst,
                             MIROperand*& dst,
                             MIROperand*& imm) {
  if (inst->opcode() != InstLoadImm) return false;
  dst = inst->operand(0);
  imm = inst->operand(1);
  return true;
}

static bool matchInstLoadStackObjectAddr(MIRInst* inst,
                                         MIROperand*& dst,
                                         MIROperand*& obj) {
  if (inst->opcode() != InstLoadStackObjectAddr) return false;
  dst = inst->operand(0);
  obj = inst->operand(1);
  return true;
}

static bool matchInstCopyFromReg(MIRInst* inst,
                                 MIROperand*& dst,
                                 MIROperand*& src) {
  if (inst->opcode() != InstCopyFromReg) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstCopyToReg(MIRInst* inst,
                               MIROperand*& dst,
                               MIROperand*& src) {
  if (inst->opcode() != InstCopyToReg) return false;
  dst = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstLoadImmToReg(MIRInst* inst,
                                  MIROperand*& dst,
                                  MIROperand*& imm) {
  if (inst->opcode() != InstLoadImmToReg) return false;
  dst = inst->operand(0);
  imm = inst->operand(1);
  return true;
}

static bool matchInstLoadRegFromStack(MIRInst* inst,
                                      MIROperand*& dst,
                                      MIROperand*& obj) {
  if (inst->opcode() != InstLoadRegFromStack) return false;
  dst = inst->operand(0);
  obj = inst->operand(1);
  return true;
}

static bool matchInstStoreRegToStack(MIRInst* inst,
                                     MIROperand*& obj,
                                     MIROperand*& src) {
  if (inst->opcode() != InstStoreRegToStack) return false;
  obj = inst->operand(0);
  src = inst->operand(1);
  return true;
}

static bool matchInstReturn(MIRInst* inst) {
  if (inst->opcode() != InstReturn) return false;

  return true;
}

/* InstLoadGlobalAddress matchAndSelectPatternInstLoadGlobalAddress begin */
static bool matchAndSelectPattern1(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstLoadGlobalAddress;
  /** Match Inst **/
  /* match inst InstLoadGlobalAddress */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  if (not matchInstLoadGlobalAddress(inst1, op1, op2)) {
    return false;
  }

  /** Select Inst **/
  auto op4 = (getVRegAs(ctx, op1));
  auto op5 = (op2);
  /* select inst LLA */
  auto inst2 = new MIRInst(LLA);
  inst2->set_operand(0, op4);
  inst2->set_operand(1, op5);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern2(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstLoadGlobalAddress;
  /** Match Inst **/
  /* match inst InstLoadGlobalAddress */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  if (not matchInstLoadGlobalAddress(inst1, op1, op2)) {
    return false;
  }

  /** Select Inst **/
  auto op4 = (op1);
  auto op6 = (getVRegAs(ctx, op1));
  auto op7 = (getHighBits(op2));
  /* select inst AUIPC */
  auto inst2 = new MIRInst(AUIPC);
  inst2->set_operand(0, op6);
  inst2->set_operand(1, op7);
  ctx.insert_inst(inst2);

  auto op5 = ctx.get_inst_def(inst2);

  auto op8 = (getLowBits(op2));
  /* select inst ADDI */
  auto inst3 = new MIRInst(ADDI);
  inst3->set_operand(0, op4);
  inst3->set_operand(1, op5);
  inst3->set_operand(2, op8);
  ctx.insert_inst(inst3);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst3));
  ctx.remove_inst(inst1);
  return true;
}

/* InstLoadGlobalAddress matchAndSelectPatternInstLoadGlobalAddressend */

/* InstLoad matchAndSelectPatternInstLoad begin */
static bool matchAndSelectPattern3(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstLoad;
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
  auto op7 = (op1);
  auto op8 = (op4);
  auto op9 = (op5);
  /* select inst getLoadOpcode(op1) */
  auto inst2 = new MIRInst(getLoadOpcode(op1));
  inst2->set_operand(0, op7);
  inst2->set_operand(1, op9);
  inst2->set_operand(2, op8);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstLoad matchAndSelectPatternInstLoadend */

/* InstStore matchAndSelectPatternInstStore begin */
static bool matchAndSelectPattern4(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstStore;
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
  if (not(isOperandVRegOrISAReg(op2) && selectAddrOffset(op1, ctx, op4, op5))) {
    return false;
  }

  /** Select Inst **/
  auto op7 = (op2);
  auto op8 = (op4);
  auto op9 = (op5);
  /* select inst getStoreOpcode(op2) */
  auto inst2 = new MIRInst(getStoreOpcode(op2));
  inst2->set_operand(0, op7);
  inst2->set_operand(1, op9);
  inst2->set_operand(2, op8);
  ctx.insert_inst(inst2);

  ctx.remove_inst(inst1);
  return true;
}

/* InstStore matchAndSelectPatternInstStoreend */

/* InstJump matchAndSelectPatternInstJump begin */
static bool matchAndSelectPattern5(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstJump;
  /** Match Inst **/
  /* match inst InstJump */
  MIROperand* op1 = nullptr;
  if (not matchInstJump(inst1, op1)) {
    return false;
  }

  /** Select Inst **/
  auto op3 = (op1);
  /* select inst J */
  auto inst2 = new MIRInst(J);
  inst2->set_operand(0, op3);
  ctx.insert_inst(inst2);

  ctx.remove_inst(inst1);
  return true;
}

/* InstJump matchAndSelectPatternInstJumpend */

/* InstReturn matchAndSelectPatternInstReturn begin */
static bool matchAndSelectPattern6(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstReturn;
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
  return true;
}

/* InstReturn matchAndSelectPatternInstReturnend */

/* InstLoadImm matchAndSelectPatternInstLoadImm begin */
static bool matchAndSelectPattern7(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstLoadImm;
  /** Match Inst **/
  /* match inst InstLoadImm */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  if (not matchInstLoadImm(inst1, op1, op2)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isZero(op2))) {
    return false;
  }

  /** Select Inst **/
  auto op4 = (op1);
  auto op5 = (getZero(op1));
  /* select inst MV */
  auto inst2 = new MIRInst(MV);
  inst2->set_operand(0, op4);
  inst2->set_operand(1, op5);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern9(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstLoadImm;
  /** Match Inst **/
  /* match inst InstLoadImm */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  if (not matchInstLoadImm(inst1, op1, op2)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandNonZeroImm12(op2))) {
    return false;
  }

  /** Select Inst **/
  auto op4 = (op1);
  auto op5 = (op2);
  /* select inst LoadImm12 */
  auto inst2 = new MIRInst(LoadImm12);
  inst2->set_operand(0, op4);
  inst2->set_operand(1, op5);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstLoadImm matchAndSelectPatternInstLoadImmend */

/* InstLoadImmToReg matchAndSelectPatternInstLoadImmToReg begin */
static bool matchAndSelectPattern8(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstLoadImmToReg;
  /** Match Inst **/
  /* match inst InstLoadImmToReg */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  if (not matchInstLoadImmToReg(inst1, op1, op2)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isZero(op2))) {
    return false;
  }

  /** Select Inst **/
  auto op4 = (op1);
  auto op5 = (getZero(op1));
  /* select inst MV */
  auto inst2 = new MIRInst(MV);
  inst2->set_operand(0, op4);
  inst2->set_operand(1, op5);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern10(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstLoadImmToReg;
  /** Match Inst **/
  /* match inst InstLoadImmToReg */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  if (not matchInstLoadImmToReg(inst1, op1, op2)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandNonZeroImm12(op2))) {
    return false;
  }

  /** Select Inst **/
  auto op4 = (op1);
  auto op5 = (op2);
  /* select inst LoadImm12 */
  auto inst2 = new MIRInst(LoadImm12);
  inst2->set_operand(0, op4);
  inst2->set_operand(1, op5);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstLoadImmToReg matchAndSelectPatternInstLoadImmToRegend */

/* InstAdd matchAndSelectPatternInstAdd begin */
static bool matchAndSelectPattern11(MIRInst* inst5, ISelContext& ctx) {
  uint32_t rootOpcode = InstAdd;
  /** Match Inst **/
  /* match inst InstAdd */
  MIROperand* op21 = nullptr;
  MIROperand* op22 = nullptr;
  MIROperand* op23 = nullptr;
  if (not matchInstAdd(inst5, op21, op22, op23)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI64(op21) && isOperandIReg(op22) && isOperandIReg(op23))) {
    return false;
  }

  /** Select Inst **/
  auto op25 = (op21);
  auto op26 = (op22);
  auto op27 = (op23);
  /* select inst ADD */
  auto inst6 = new MIRInst(ADD);
  inst6->set_operand(0, op25);
  inst6->set_operand(1, op26);
  inst6->set_operand(2, op27);
  ctx.insert_inst(inst6);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst5), ctx.get_inst_def(inst6));
  ctx.remove_inst(inst5);
  return true;
}
static bool matchAndSelectPattern12(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstAdd;
  /** Match Inst **/
  /* match inst InstAdd */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstAdd(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI32(op1) && isOperandIReg(op2) && isOperandIReg(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst ADDW */
  auto inst2 = new MIRInst(ADDW);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstAdd matchAndSelectPatternInstAddend */

/* InstSub matchAndSelectPatternInstSub begin */
static bool matchAndSelectPattern13(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstSub;
  /** Match Inst **/
  /* match inst InstSub */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstSub(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI64(op1) && isOperandIReg(op2) && isOperandIReg(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst SUB */
  auto inst2 = new MIRInst(SUB);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern14(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstSub;
  /** Match Inst **/
  /* match inst InstSub */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstSub(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI32(op1) && isOperandIReg(op2) && isOperandIReg(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst SUBW */
  auto inst2 = new MIRInst(SUBW);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstSub matchAndSelectPatternInstSubend */

/* InstMul matchAndSelectPatternInstMul begin */
static bool matchAndSelectPattern15(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstMul;
  /** Match Inst **/
  /* match inst InstMul */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstMul(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI64(op1) && isOperandIReg(op2) && isOperandIReg(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst MUL */
  auto inst2 = new MIRInst(MUL);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern16(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstMul;
  /** Match Inst **/
  /* match inst InstMul */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstMul(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI32(op1) && isOperandIReg(op2) && isOperandIReg(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst MULW */
  auto inst2 = new MIRInst(MULW);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstMul matchAndSelectPatternInstMulend */

/* InstUDiv matchAndSelectPatternInstUDiv begin */
static bool matchAndSelectPattern17(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstUDiv;
  /** Match Inst **/
  /* match inst InstUDiv */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstUDiv(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI64(op1) && isOperandIReg(op2) && isOperandIReg(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst DIVU */
  auto inst2 = new MIRInst(DIVU);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstUDiv matchAndSelectPatternInstUDivend */

/* InstURem matchAndSelectPatternInstURem begin */
static bool matchAndSelectPattern18(MIRInst* inst2, ISelContext& ctx) {
  uint32_t rootOpcode = InstURem;
  /** Match Inst **/
  /* match inst InstURem */
  MIROperand* op8 = nullptr;
  MIROperand* op9 = nullptr;
  MIROperand* op10 = nullptr;
  if (not matchInstURem(inst2, op8, op9, op10)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI64(op8) && isOperandIReg(op9) && isOperandIReg(op10))) {
    return false;
  }

  /** Select Inst **/
  auto op12 = (op8);
  auto op13 = (op9);
  auto op14 = (op10);
  /* select inst REMU */
  auto inst3 = new MIRInst(REMU);
  inst3->set_operand(0, op12);
  inst3->set_operand(1, op13);
  inst3->set_operand(2, op14);
  ctx.insert_inst(inst3);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst2), ctx.get_inst_def(inst3));
  ctx.remove_inst(inst2);
  return true;
}

/* InstURem matchAndSelectPatternInstURemend */

/* InstAnd matchAndSelectPatternInstAnd begin */
static bool matchAndSelectPattern19(MIRInst* inst2, ISelContext& ctx) {
  uint32_t rootOpcode = InstAnd;
  /** Match Inst **/
  /* match inst InstAnd */
  MIROperand* op8 = nullptr;
  MIROperand* op9 = nullptr;
  MIROperand* op10 = nullptr;
  if (not matchInstAnd(inst2, op8, op9, op10)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op9) && isOperandIReg(op10))) {
    return false;
  }

  /** Select Inst **/
  auto op12 = (op8);
  auto op13 = (op9);
  auto op14 = (op10);
  /* select inst getIntegerBinaryRegOpcode(rootOpcode) */
  auto inst3 = new MIRInst(getIntegerBinaryRegOpcode(rootOpcode));
  inst3->set_operand(0, op12);
  inst3->set_operand(1, op13);
  inst3->set_operand(2, op14);
  ctx.insert_inst(inst3);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst2), ctx.get_inst_def(inst3));
  ctx.remove_inst(inst2);
  return true;
}
static bool matchAndSelectPattern22(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstAnd;
  /** Match Inst **/
  /* match inst InstAnd */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstAnd(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op2) && isOperandImm12(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst getIntegerBinaryImmOpcode(rootOpcode) */
  auto inst2 = new MIRInst(getIntegerBinaryImmOpcode(rootOpcode));
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstAnd matchAndSelectPatternInstAndend */

/* InstOr matchAndSelectPatternInstOr begin */
static bool matchAndSelectPattern20(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstOr;
  /** Match Inst **/
  /* match inst InstOr */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstOr(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op2) && isOperandIReg(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst getIntegerBinaryRegOpcode(rootOpcode) */
  auto inst2 = new MIRInst(getIntegerBinaryRegOpcode(rootOpcode));
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern23(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstOr;
  /** Match Inst **/
  /* match inst InstOr */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstOr(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op2) && isOperandImm12(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst getIntegerBinaryImmOpcode(rootOpcode) */
  auto inst2 = new MIRInst(getIntegerBinaryImmOpcode(rootOpcode));
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstOr matchAndSelectPatternInstOrend */

/* InstXor matchAndSelectPatternInstXor begin */
static bool matchAndSelectPattern21(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstXor;
  /** Match Inst **/
  /* match inst InstXor */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstXor(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op2) && isOperandIReg(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst getIntegerBinaryRegOpcode(rootOpcode) */
  auto inst2 = new MIRInst(getIntegerBinaryRegOpcode(rootOpcode));
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern24(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstXor;
  /** Match Inst **/
  /* match inst InstXor */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstXor(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op2) && isOperandImm12(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst getIntegerBinaryImmOpcode(rootOpcode) */
  auto inst2 = new MIRInst(getIntegerBinaryImmOpcode(rootOpcode));
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstXor matchAndSelectPatternInstXorend */

/* InstShl matchAndSelectPatternInstShl begin */
static bool matchAndSelectPattern25(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstShl;
  /** Match Inst **/
  /* match inst InstShl */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstShl(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI64(op1) && isOperandIReg(op2) && isOperandIReg(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst SLL */
  auto inst2 = new MIRInst(SLL);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstShl matchAndSelectPatternInstShlend */

/* InstLShr matchAndSelectPatternInstLShr begin */
static bool matchAndSelectPattern26(MIRInst* inst2, ISelContext& ctx) {
  uint32_t rootOpcode = InstLShr;
  /** Match Inst **/
  /* match inst InstLShr */
  MIROperand* op8 = nullptr;
  MIROperand* op9 = nullptr;
  MIROperand* op10 = nullptr;
  if (not matchInstLShr(inst2, op8, op9, op10)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI64(op8) && isOperandIReg(op9) && isOperandIReg(op10))) {
    return false;
  }

  /** Select Inst **/
  auto op12 = (op8);
  auto op13 = (op9);
  auto op14 = (op10);
  /* select inst SRL */
  auto inst3 = new MIRInst(SRL);
  inst3->set_operand(0, op12);
  inst3->set_operand(1, op13);
  inst3->set_operand(2, op14);
  ctx.insert_inst(inst3);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst2), ctx.get_inst_def(inst3));
  ctx.remove_inst(inst2);
  return true;
}

/* InstLShr matchAndSelectPatternInstLShrend */

/* InstAShr matchAndSelectPatternInstAShr begin */
static bool matchAndSelectPattern27(MIRInst* inst2, ISelContext& ctx) {
  uint32_t rootOpcode = InstAShr;
  /** Match Inst **/
  /* match inst InstAShr */
  MIROperand* op8 = nullptr;
  MIROperand* op9 = nullptr;
  MIROperand* op10 = nullptr;
  if (not matchInstAShr(inst2, op8, op9, op10)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI64(op8) && isOperandIReg(op9) && isOperandIReg(op10))) {
    return false;
  }

  /** Select Inst **/
  auto op12 = (op8);
  auto op13 = (op9);
  auto op14 = (op10);
  /* select inst SRA */
  auto inst3 = new MIRInst(SRA);
  inst3->set_operand(0, op12);
  inst3->set_operand(1, op13);
  inst3->set_operand(2, op14);
  ctx.insert_inst(inst3);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst2), ctx.get_inst_def(inst3));
  ctx.remove_inst(inst2);
  return true;
}

/* InstAShr matchAndSelectPatternInstAShrend */

/* InstSDiv matchAndSelectPatternInstSDiv begin */
static bool matchAndSelectPattern28(MIRInst* inst2, ISelContext& ctx) {
  uint32_t rootOpcode = InstSDiv;
  /** Match Inst **/
  /* match inst InstSDiv */
  MIROperand* op8 = nullptr;
  MIROperand* op9 = nullptr;
  MIROperand* op10 = nullptr;
  if (not matchInstSDiv(inst2, op8, op9, op10)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI64(op8) && isOperandIReg(op9) && isOperandIReg(op10))) {
    return false;
  }

  /** Select Inst **/
  auto op12 = (op8);
  auto op13 = (op9);
  auto op14 = (op10);
  /* select inst DIV */
  auto inst3 = new MIRInst(DIV);
  inst3->set_operand(0, op12);
  inst3->set_operand(1, op13);
  inst3->set_operand(2, op14);
  ctx.insert_inst(inst3);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst2), ctx.get_inst_def(inst3));
  ctx.remove_inst(inst2);
  return true;
}
static bool matchAndSelectPattern29(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstSDiv;
  /** Match Inst **/
  /* match inst InstSDiv */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstSDiv(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI32(op1) && isOperandIReg(op2) && isOperandIReg(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst DIVW */
  auto inst2 = new MIRInst(DIVW);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstSDiv matchAndSelectPatternInstSDivend */

/* InstSRem matchAndSelectPatternInstSRem begin */
static bool matchAndSelectPattern30(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstSRem;
  /** Match Inst **/
  /* match inst InstSRem */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstSRem(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI64(op1) && isOperandIReg(op2) && isOperandIReg(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst REM */
  auto inst2 = new MIRInst(REM);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern31(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstSRem;
  /** Match Inst **/
  /* match inst InstSRem */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstSRem(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandI32(op1) && isOperandIReg(op2) && isOperandIReg(op3))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst REMW */
  auto inst2 = new MIRInst(REMW);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstSRem matchAndSelectPatternInstSRemend */

/* InstICmp matchAndSelectPatternInstICmp begin */
static bool matchAndSelectPattern32(MIRInst* inst5, ISelContext& ctx) {
  uint32_t rootOpcode = InstICmp;
  /** Match Inst **/
  /* match inst InstICmp */
  MIROperand* op24 = nullptr;
  MIROperand* op25 = nullptr;
  MIROperand* op26 = nullptr;
  MIROperand* op27 = nullptr;
  if (not matchInstICmp(inst5, op24, op25, op26, op27)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op25) && isOperandIReg(op26) &&
          isCompareOp(op27, CompareOp::ICmpSignedLessThan))) {
    return false;
  }

  /** Select Inst **/
  auto op29 = (op24);
  auto op30 = (op25);
  auto op31 = (op26);
  /* select inst SLT */
  auto inst6 = new MIRInst(SLT);
  inst6->set_operand(0, op29);
  inst6->set_operand(1, op30);
  inst6->set_operand(2, op31);
  ctx.insert_inst(inst6);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst5), ctx.get_inst_def(inst6));
  ctx.remove_inst(inst5);
  return true;
}
static bool matchAndSelectPattern33(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstICmp;
  /** Match Inst **/
  /* match inst InstICmp */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  MIROperand* op4 = nullptr;
  if (not matchInstICmp(inst1, op1, op2, op3, op4)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op2) && isOperandIReg(op3) &&
          isCompareOp(op4, CompareOp::ICmpSignedGreaterEqual))) {
    return false;
  }

  /** Select Inst **/
  auto op6 = (op1);
  auto op7 = (op3);
  auto op8 = (op2);
  /* select inst SLT */
  auto inst2 = new MIRInst(SLT);
  inst2->set_operand(0, op6);
  inst2->set_operand(1, op7);
  inst2->set_operand(2, op8);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern34(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstICmp;
  /** Match Inst **/
  /* match inst InstICmp */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  MIROperand* op4 = nullptr;
  if (not matchInstICmp(inst1, op1, op2, op3, op4)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op2) && isOperandIReg(op3) &&
          isCompareOp(op4, CompareOp::ICmpUnsignedLessThan))) {
    return false;
  }

  /** Select Inst **/
  auto op6 = (op1);
  auto op7 = (op2);
  auto op8 = (op3);
  /* select inst SLTU */
  auto inst2 = new MIRInst(SLTU);
  inst2->set_operand(0, op6);
  inst2->set_operand(1, op7);
  inst2->set_operand(2, op8);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern35(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstICmp;
  /** Match Inst **/
  /* match inst InstICmp */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  MIROperand* op4 = nullptr;
  if (not matchInstICmp(inst1, op1, op2, op3, op4)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op2) && isOperandIReg(op3) &&
          isCompareOp(op4, CompareOp::ICmpSignedGreaterThan))) {
    return false;
  }

  /** Select Inst **/
  auto op6 = (op1);
  auto op7 = (op3);
  auto op8 = (op2);
  /* select inst SLT */
  auto inst2 = new MIRInst(SLT);
  inst2->set_operand(0, op6);
  inst2->set_operand(1, op7);
  inst2->set_operand(2, op8);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern36(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstICmp;
  /** Match Inst **/
  /* match inst InstICmp */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  MIROperand* op4 = nullptr;
  if (not matchInstICmp(inst1, op1, op2, op3, op4)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op2) && isOperandIReg(op3) &&
          isCompareOp(op4, CompareOp::ICmpUnsignedGreaterThan))) {
    return false;
  }

  /** Select Inst **/
  auto op6 = (op1);
  auto op7 = (op3);
  auto op8 = (op2);
  /* select inst SLTU */
  auto inst2 = new MIRInst(SLTU);
  inst2->set_operand(0, op6);
  inst2->set_operand(1, op7);
  inst2->set_operand(2, op8);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern37(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstICmp;
  /** Match Inst **/
  /* match inst InstICmp */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  MIROperand* op4 = nullptr;
  if (not matchInstICmp(inst1, op1, op2, op3, op4)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op2) && isZero(op3) &&
          isCompareOp(op4, CompareOp::ICmpEqual))) {
    return false;
  }

  /** Select Inst **/
  auto op6 = (op1);
  auto op7 = (op2);
  auto op8 = (getOne(op3));
  /* select inst SLTIU */
  auto inst2 = new MIRInst(SLTIU);
  inst2->set_operand(0, op6);
  inst2->set_operand(1, op7);
  inst2->set_operand(2, op8);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern38(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstICmp;
  /** Match Inst **/
  /* match inst InstICmp */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  MIROperand* op4 = nullptr;
  if (not matchInstICmp(inst1, op1, op2, op3, op4)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op2) && isZero(op3) &&
          isCompareOp(op4, CompareOp::ICmpNotEqual))) {
    return false;
  }

  /** Select Inst **/
  auto op6 = (op1);
  auto op7 = (getZero(op3));
  auto op8 = (op2);
  /* select inst SLTU */
  auto inst2 = new MIRInst(SLTU);
  inst2->set_operand(0, op6);
  inst2->set_operand(1, op7);
  inst2->set_operand(2, op8);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}
static bool matchAndSelectPattern39(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstICmp;
  /** Match Inst **/
  /* match inst InstICmp */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  MIROperand* op4 = nullptr;
  if (not matchInstICmp(inst1, op1, op2, op3, op4)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op2) && isOperandIReg(op3) &&
          isCompareOp(op4, CompareOp::ICmpSignedLessEqual))) {
    return false;
  }

  /** Select Inst **/
  auto op6 = (op1);
  auto op8 = (getVRegAs(ctx, op1));
  auto op9 = (op3);
  auto op10 = (op2);
  /* select inst SLT */
  auto inst2 = new MIRInst(SLT);
  inst2->set_operand(0, op8);
  inst2->set_operand(1, op9);
  inst2->set_operand(2, op10);
  ctx.insert_inst(inst2);

  auto op7 = ctx.get_inst_def(inst2);

  auto op11 = (getOne(op3));
  /* select inst XORI */
  auto inst3 = new MIRInst(XORI);
  inst3->set_operand(0, op6);
  inst3->set_operand(1, op7);
  inst3->set_operand(2, op11);
  ctx.insert_inst(inst3);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst3));
  ctx.remove_inst(inst1);
  return true;
}

/* InstICmp matchAndSelectPatternInstICmpend */

/* InstBranch matchAndSelectPatternInstBranch begin */
static bool matchAndSelectPattern40(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstBranch;
  /** Match Inst **/
  /* match inst InstBranch */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstBranch(inst1, op1, op2, op3)) {
    return false;
  }

  /* match predicate for operands  */
  if (not(isOperandIReg(op1))) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (getZero(op1));
  auto op7 = (op2);
  auto op8 = (op3);
  /* select inst BNE */
  auto inst2 = new MIRInst(BNE);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  inst2->set_operand(3, op8);
  ctx.insert_inst(inst2);

  ctx.remove_inst(inst1);
  return true;
}

/* InstBranch matchAndSelectPatternInstBranchend */

/* InstF2S matchAndSelectPatternInstF2S begin */
static bool matchAndSelectPattern41(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstF2S;
  /** Match Inst **/
  /* match inst InstF2S */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  if (not matchInstF2S(inst1, op1, op2)) {
    return false;
  }

  /** Select Inst **/
  auto op4 = (op1);
  auto op5 = (op2);
  /* select inst FCVT_W_S */
  auto inst2 = new MIRInst(FCVT_W_S);
  inst2->set_operand(0, op4);
  inst2->set_operand(1, op5);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstF2S matchAndSelectPatternInstF2Send */

/* InstS2F matchAndSelectPatternInstS2F begin */
static bool matchAndSelectPattern42(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstS2F;
  /** Match Inst **/
  /* match inst InstS2F */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  if (not matchInstS2F(inst1, op1, op2)) {
    return false;
  }

  /** Select Inst **/
  auto op4 = (op1);
  auto op5 = (op2);
  /* select inst FCVT_S_W */
  auto inst2 = new MIRInst(FCVT_S_W);
  inst2->set_operand(0, op4);
  inst2->set_operand(1, op5);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstS2F matchAndSelectPatternInstS2Fend */

/* InstFCmp matchAndSelectPatternInstFCmp begin */
static bool matchAndSelectPattern43(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstFCmp;
  /** Match Inst **/
  /* match inst InstFCmp */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  MIROperand* op4 = nullptr;
  if (not matchInstFCmp(inst1, op1, op2, op3, op4)) {
    return false;
  }

  /* match predicate for operands  */
  MIROperand* op5 = nullptr;
  MIROperand* op6 = nullptr;
  MIROperand* op7 = nullptr;
  if (not(selectFCmpOpcode(op4, op2, op3, op5, op6, op7))) {
    return false;
  }

  /** Select Inst **/
  auto op9 = (op1);
  auto op10 = (op5);
  auto op11 = (op6);
  /* select inst static_cast<uint32_t>((op7)->imm()) */
  auto inst2 = new MIRInst(static_cast<uint32_t>((op7)->imm()));
  inst2->set_operand(0, op9);
  inst2->set_operand(1, op10);
  inst2->set_operand(2, op11);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstFCmp matchAndSelectPatternInstFCmpend */

/* InstFNeg matchAndSelectPatternInstFNeg begin */
static bool matchAndSelectPattern44(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstFNeg;
  /** Match Inst **/
  /* match inst InstFNeg */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  if (not matchInstFNeg(inst1, op1, op2)) {
    return false;
  }

  /** Select Inst **/
  auto op4 = (op1);
  auto op5 = (op2);
  /* select inst FNEG_S */
  auto inst2 = new MIRInst(FNEG_S);
  inst2->set_operand(0, op4);
  inst2->set_operand(1, op5);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstFNeg matchAndSelectPatternInstFNegend */

/* InstFAdd matchAndSelectPatternInstFAdd begin */
static bool matchAndSelectPattern45(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstFAdd;
  /** Match Inst **/
  /* match inst InstFAdd */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstFAdd(inst1, op1, op2, op3)) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst FADD_S */
  auto inst2 = new MIRInst(FADD_S);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstFAdd matchAndSelectPatternInstFAddend */

/* InstFSub matchAndSelectPatternInstFSub begin */
static bool matchAndSelectPattern46(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstFSub;
  /** Match Inst **/
  /* match inst InstFSub */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstFSub(inst1, op1, op2, op3)) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst FSUB_S */
  auto inst2 = new MIRInst(FSUB_S);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstFSub matchAndSelectPatternInstFSubend */

/* InstFMul matchAndSelectPatternInstFMul begin */
static bool matchAndSelectPattern47(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstFMul;
  /** Match Inst **/
  /* match inst InstFMul */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstFMul(inst1, op1, op2, op3)) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst FMUL_S */
  auto inst2 = new MIRInst(FMUL_S);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstFMul matchAndSelectPatternInstFMulend */

/* InstFDiv matchAndSelectPatternInstFDiv begin */
static bool matchAndSelectPattern48(MIRInst* inst1, ISelContext& ctx) {
  uint32_t rootOpcode = InstFDiv;
  /** Match Inst **/
  /* match inst InstFDiv */
  MIROperand* op1 = nullptr;
  MIROperand* op2 = nullptr;
  MIROperand* op3 = nullptr;
  if (not matchInstFDiv(inst1, op1, op2, op3)) {
    return false;
  }

  /** Select Inst **/
  auto op5 = (op1);
  auto op6 = (op2);
  auto op7 = (op3);
  /* select inst FDIV_S */
  auto inst2 = new MIRInst(FDIV_S);
  inst2->set_operand(0, op5);
  inst2->set_operand(1, op6);
  inst2->set_operand(2, op7);
  ctx.insert_inst(inst2);

  /* Replace Operand */
  ctx.replace_operand(ctx.get_inst_def(inst1), ctx.get_inst_def(inst2));
  ctx.remove_inst(inst1);
  return true;
}

/* InstFDiv matchAndSelectPatternInstFDivend */

static bool matchAndSelectImpl(MIRInst* inst,
                               ISelContext& ctx,
                               bool debugMatchSelect) {
  bool success = false;
  switch (inst->opcode()) {
    case InstLoadGlobalAddress: {
      if (matchAndSelectPattern1(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern2(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstLoad: {
      if (matchAndSelectPattern3(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstStore: {
      if (matchAndSelectPattern4(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstJump: {
      if (matchAndSelectPattern5(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstReturn: {
      if (matchAndSelectPattern6(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstLoadImm: {
      if (matchAndSelectPattern7(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern9(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstLoadImmToReg: {
      if (matchAndSelectPattern8(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern10(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstAdd: {
      if (matchAndSelectPattern11(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern12(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstSub: {
      if (matchAndSelectPattern13(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern14(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstMul: {
      if (matchAndSelectPattern15(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern16(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstUDiv: {
      if (matchAndSelectPattern17(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstURem: {
      if (matchAndSelectPattern18(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstAnd: {
      if (matchAndSelectPattern19(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern22(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstOr: {
      if (matchAndSelectPattern20(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern23(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstXor: {
      if (matchAndSelectPattern21(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern24(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstShl: {
      if (matchAndSelectPattern25(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstLShr: {
      if (matchAndSelectPattern26(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstAShr: {
      if (matchAndSelectPattern27(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstSDiv: {
      if (matchAndSelectPattern28(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern29(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstSRem: {
      if (matchAndSelectPattern30(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern31(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstICmp: {
      if (matchAndSelectPattern32(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern33(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern34(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern35(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern36(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern37(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern38(inst, ctx)) {
        success = true;
        break;
      }
      if (matchAndSelectPattern39(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstBranch: {
      if (matchAndSelectPattern40(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstF2S: {
      if (matchAndSelectPattern41(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstS2F: {
      if (matchAndSelectPattern42(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstFCmp: {
      if (matchAndSelectPattern43(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstFNeg: {
      if (matchAndSelectPattern44(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstFAdd: {
      if (matchAndSelectPattern45(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstFSub: {
      if (matchAndSelectPattern46(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstFMul: {
      if (matchAndSelectPattern47(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    case InstFDiv: {
      if (matchAndSelectPattern48(inst, ctx)) {
        success = true;
        break;
      }
      break;
    }
    default: break;
  }
  if (debugMatchSelect) {
    auto& instInfo = ctx.codegen_ctx().instInfo.get_instinfo(inst);
    std::cerr << instInfo.name();
    if (success)
      std::cerr << " success." << std::endl;
    else
      std::cerr << " failed." << std::endl;
  }
  return success;
}

class RISCVISelInfo final : public TargetISelInfo {
public:
  bool is_legal_geninst(uint32_t opcode) const override;
  bool match_select(MIRInst* inst, ISelContext& ctx) const override;
  void legalizeInstWithStackOperand(InstLegalizeContext& ctx,
                                    MIROperand* op,
                                    StackObject& obj) const override;
  void postLegalizeInst(const InstLegalizeContext& ctx) const override;
  MIROperand* materializeFPConstant(float fpVal, LoweringContext& loweringCtx)
    const override;
};

TargetISelInfo& getRISCVISelInfo() {
  static RISCVISelInfo iselInfo;
  return iselInfo;
}

RISCV_NAMESPACE_END