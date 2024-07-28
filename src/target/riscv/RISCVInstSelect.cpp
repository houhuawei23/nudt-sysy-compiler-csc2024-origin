#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/iselinfo.hpp"
#include "target/riscv/RISCV.hpp"
#include "autogen/riscv/InstInfoDecl.hpp"
#include "autogen/riscv/ISelInfoDecl.hpp"
#include "support/StaticReflection.hpp"
#include <cstring>
namespace mir::RISCV {

static MIROperand getVRegAs(ISelContext& ctx, MIROperand ref) {
  return MIROperand::asVReg(ctx.codegen_ctx().nextId(), ref.type());
}

constexpr RISCVInst getIntegerBinaryRegOpcode(uint32_t opcode) {
  switch (opcode) {
    case InstAnd:
      return AND;
    case InstOr:
      return OR;
    case InstXor:
      return XOR;
    default:
      assert(false && "Unsupported binary register instruction");
  }
}
constexpr RISCVInst getIntegerBinaryImmOpcode(uint32_t opcode) {
  switch (opcode) {
    case InstAnd:
      return ANDI;
    case InstOr:
      return ORI;
    case InstXor:
      return XORI;
    default:
      assert(false && "Unsupported binary immediate instruction");
  }
}

static RISCVInst getLoadOpcode(MIROperand dst) {
  switch (dst.type()) {
    case OperandType::Bool:
    case OperandType::Int8:
      return LB;
    case OperandType::Int16:
      return LH;
    case OperandType::Int32:
      return LW;
    case OperandType::Int64:
      return LD;
    case OperandType::Float32:
      return FLW;
    default:
      assert(false && "Unsupported operand type for load instruction");
  }
}
static RISCVInst getStoreOpcode(MIROperand src) {
  switch (src.type()) {
    case OperandType::Bool:
    case OperandType::Int8:
      return SB;
    case OperandType::Int16:
      return SH;
    case OperandType::Int32:
      return SW;
    case OperandType::Int64:
      return SD;
    case OperandType::Float32:
      return FSW;
    default:
      assert(false && "Unsupported operand type for store instruction");
  }
}

static bool selectFCmpOpcode(MIROperand opcode,
                             MIROperand lhs,
                             MIROperand rhs,
                             MIROperand& outlhs,
                             MIROperand& outrhs,
                             MIROperand& outOpcode) {
  // std::cerr << "selectFCmpOpcode" << std::endl;
  const auto op = static_cast<CompareOp>(opcode.imm());
  // std::cerr << "comapre op: " << utils::enumName(op) << std::endl;
  if (!isOperandFPR(lhs) or !isOperandFPR(rhs)) {
    return false;
  }
  outlhs = lhs;
  outrhs = rhs;
  RISCVInst newOpcode;
  switch (op) {
    case CompareOp::FCmpOrderedLessThan:
      newOpcode = FLT_S;
      break;
    case CompareOp::FCmpOrderedLessEqual:
      newOpcode = FLE_S;
      break;
    case CompareOp::FCmpOrderedGreaterThan:
      outlhs = rhs;
      outrhs = lhs;
      newOpcode = FLT_S;
      break;
    case CompareOp::FCmpOrderedGreaterEqual:
      outlhs = rhs;
      outrhs = lhs;
      newOpcode = FLE_S;
      break;
    case CompareOp::FCmpOrderedEqual:
      newOpcode = FEQ_S;
      break;
    default:
      return false;
  }
  outOpcode = MIROperand::asImm(newOpcode, OperandType::Special);
  // std::cerr << "newOpcode: " << newOpcode << std::endl;
  return true;
}

static bool selectAddrOffset(MIROperand addr,
                             ISelContext& ctx,
                             MIROperand& base,
                             MIROperand& offset) {
  bool debug = false;
  auto dumpInst = [&](MIRInst* inst) {
    auto& instInfo = ctx.codegen_ctx().instInfo.getInstInfo(inst);
    instInfo.print(std::cerr << "selectAddrOffset: ", *inst, false);
    std::cerr << std::endl;
  };

  const auto addrInst = ctx.lookupDef(addr);
  if (addrInst) {
    if (debug) dumpInst(addrInst);
    if (addrInst->opcode() == InstLoadStackObjectAddr) {
      base = addrInst->operand(1);  // obj
      offset = MIROperand::asImm(0, OperandType::Int64);
      return true;
    }
    if (addrInst->opcode() == InstLoadGlobalAddress) {
    }
  }
  if (isOperandIReg(addr)) {
    base = addr;
    offset = MIROperand::asImm(0, OperandType::Int64);
    return true;
  }
  return false;
}

static bool isOperandI64(MIROperand op) {
  return op.type() == OperandType::Int64;
}
static bool isOperandI32(MIROperand op) {
  return op.type() == OperandType::Int32;
}

static bool isZero(MIROperand operand) {
  if (operand.isReg() && operand.reg() == RISCV::X0) return true;
  return operand.isImm() && operand.imm() == 0;
}

static MIROperand getZero(MIROperand operand) {
  return MIROperand::asISAReg(RISCV::X0, operand.type());
}

static MIROperand getOne(MIROperand operand) {
  return MIROperand::asImm(1, operand.type());
}

}  // namespace mir::RISCV

//! Dont Change !!
#include "autogen/riscv/ISelInfoImpl.hpp"

namespace mir::RISCV {

bool RISCVISelInfo::isLegalInst(uint32_t opcode) const {
  return true;
}
static bool legalizeInst(MIRInst* inst, ISelContext& ctx) {
  bool modified = false;

  const auto imm2reg = [&](MIROperand& operand) {
    if (operand.isImm()) {
      if (operand.imm() == 0) {
        operand = getZero(operand);
      } else {
        const auto reg = getVRegAs(ctx, operand);
        // ctx.newInst(InstLoadImm).setOperand<0>(reg).setOperand<1>(operand);
        ctx.insertMIRInst(InstLoadImm, {reg, operand});
        operand = reg;
      }
      modified = true;
    }
  };
  switch (inst->opcode()) {
    case InstAdd:
    case InstAnd:
    case InstOr:
    case InstXor: {
      imm2reg(inst->operand(1));
      imm2reg(inst->operand(2));
      break;
    }
    case InstSub: { /* InstSub dst, src1, src2 */
      auto& src1 = inst->operand(1);
      auto& src2 = inst->operand(2);
      imm2reg(src1);

      if (src2.isImm()) { /* sub to add */
        auto neg = getNeg(src2);
        if (isOperandImm12(neg)) {
          inst->set_opcode(InstAdd);
          inst->set_operand(2, neg);
          modified = true;
          break;
        }
      }
      imm2reg(src2);
      break;
    }
    case InstNeg: {
      auto val = inst->operand(1);
      /* neg dst, val
      ->
      sub dst, x0, val */
      inst->set_opcode(InstSub);
      inst->set_operand(1, getZero(val));
      inst->set_operand(2, val);
      break;
    }
    case InstAbs: {
      /* abs dst, val */
      imm2reg(inst->operand(1));
      break;
    }
    case InstMul:
    case InstSDiv:
    case InstSRem:
    case InstUDiv:
    case InstURem: {
      imm2reg(inst->operand(1));
      imm2reg(inst->operand(2));

      break;
    }
    case InstICmp: {
      /* InstICmp dst, src1, src2, op */
      auto op = inst->operand(3);
      if (isICmpEqualityOp(op)) {
        /**
         * ICmp dst, src1, src2, EQ/NE
         * legalize ->
         * xor newdst, src1, src2
         * ICmp dst, newdst, 0, EQ/NE
         * instSelect ->
         * xor newdst, src1, src2
         * sltiu dst, newdst, 1
         */
        auto newDst = getVRegAs(ctx, inst->operand(0));

        ctx.insertMIRInst(InstXor, {newDst, inst->operand(1), inst->operand(2)});

        inst->set_operand(1, newDst); /* icmp */
        inst->set_operand(2, getZero(inst->operand(2)));
        modified = true;
      } else {
        imm2reg(inst->operand(1));
        imm2reg(inst->operand(2));
      }
      break;
    }
    case InstS2F: {
      imm2reg(inst->operand(1));
      break;
    }
    case InstFCmp: {
      const auto op = static_cast<CompareOp>(inst->operand(3).imm());
      if (op == CompareOp::FCmpOrderedNotEqual) {
        /**
         * FCmp dst, src1, src2, NE
         * ->
         * FCmp newdst, src1, src2, EQ
         * xor dst, newdst, 1
         */
        auto newDst = getVRegAs(ctx, inst->operand(0));

        auto dst = inst->operand(0);
        inst->set_operand(0, newDst);
        inst->set_operand(3, MIROperand::asImm(CompareOp::FCmpOrderedEqual, OperandType::Special));

        ctx.insertMIRInst(++ctx.insertPoint(), InstXor, {dst, newDst, getOne(newDst)});
        modified = true;
        break;
      }
      // switch (op) {
      //   case CompareOp::FCmpOrderedNotEqual:
      //   {

      //   }
      // }
    }
    case InstStore: { /* InstStore addr, src, align*/

      imm2reg(inst->operand(1));
      break;
    }
  }
  return modified;
}

bool RISCVISelInfo::match_select(MIRInst* inst, ISelContext& ctx) const {
  bool debugMatchSelect = false;
  bool ret = legalizeInst(inst, ctx);
  return ret | matchAndSelectImpl(inst, ctx, debugMatchSelect);
}

static MIROperand getAlign(int64_t immVal) {
  return MIROperand::asImm(immVal, OperandType::Special);
}

// void RISCVISelInfo::adjustReg(MIRInstList& insts,
//                               MIRInstList::iterator& iter,
//                               MIROperand& dst,
//                               MIROperand& src,
//                               int64_t& imm) const {
//   //
//   if (-2048 <= imm && imm <= 2047) {
//     // imm 12
//     return;
//   }
//   // else
//   // imm 32
// }

/**
 * sw rs2, offset(rs1)
 * M[x[rs1] + sext(offset)] = x[rs2][31: 0]
 *
 * lw rd, offset(rs1) or lw rd, rs1, offset
 * x[rd] = sext(M[x[rs1] + sext(offset)][31:0])
 */
void RISCVISelInfo::legalizeInstWithStackOperand(const InstLegalizeContext& ctx,
                                                 MIROperand op,
                                                 StackObject& obj) const {
  bool debugLISO = false;

  const auto& inst = ctx.inst;
  auto& insts = ctx.instructions;
  auto& iter = ctx.iter;
  auto& instInfo = ctx.codeGenCtx.instInfo.getInstInfo(inst);
  auto dumpInst = [&](MIRInst* inst) {
    instInfo.print(std::cerr, *inst, true);
    std::cerr << std::endl;
  };
  if (debugLISO) {
    instInfo.print(std::cerr, *inst, true);
  }
  int64_t immVal = obj.offset;  // rs1
  switch (inst->opcode()) {
    case SD:
    case SW:
    case SH:
    case SB:
    case FSW: {
      // rel addr = obj.offset + offset
      immVal += inst->operand(1).imm();
      break;
    }
    case LD:
    case LW:
    // case LWU:
    case LH:
    case LHU:
    case LB:
    case LBU:
    case FLW: {
      // rel addr = obj.offset + offset
      immVal += inst->operand(1).imm();
      break;
    }
    default:
      break;
  }

  MIROperand base = sp;

  legalizeAddrBaseOffsetPostRA(ctx.instructions, ctx.iter, base, immVal);
  auto offset = MIROperand::asImm(immVal, OperandType::Int64);

  switch (inst->opcode()) {
    case InstLoadStackObjectAddr: {
      /**
       * LoadStackObjAddr {dst:INTREG, Def}, {obj:StackObject, Metadata}
       *
       * LoadStackObjAddr dst, obj
       * ->
       * addi dst, sp, offset
       *
       * addi rd, rs1, imm
       * x[rd] = x[rs1] + sext(imm)
       */
      if (debugLISO) std::cout << "addi rd, rs1, imm" << std::endl;

      inst->set_opcode(ADDI);
      inst->set_operand(1, base);
      inst->set_operand(2, offset);
      break;
    }
    case InstStoreRegToStack: {
      /**
       * StoreRegToStack obj[0 Metadata], src[1 Use]
       *
       * sw rs2[0 Use], offset[1 Metadata](rs1[2 Use])
       * M[x[rs1] + sext(offset)] = x[rs2][31: 0]
       *
       * StoreRegToStack obj, src
       * ->
       * sw src, offset(sp)
       */
      if (debugLISO) std::cout << "sw rs2, offset(rs1)" << std::endl;
      inst->set_opcode(isOperandGR(inst->operand(1)) ? SD : FSW);
      auto oldSrc = inst->operand(1);
      inst->set_operand(0, oldSrc);                                          /* src2 := src */
      inst->set_operand(1, offset);                                          /* offset */
      inst->set_operand(2, base);                                            /* base = sp */
      inst->set_operand(3, getAlign(isOperandGR(inst->operand(0)) ? 8 : 4)); /* align */

      break;
    }
    case InstLoadRegFromStack: {
      /**
       * LoadRegFromStack dst[0 Def], obj[1 Metadata]
       * lw rd, offset(rs1)
       *
       * LoadRegFromStack dst, obj
       * ->
       * lw dst, offset(sp)
       */
      if (debugLISO) std::cout << "lw rd, offset(rs1)" << std::endl;

      inst->set_opcode(isOperandGR(inst->operand(0)) ? LD : FLW);

      inst->set_operand(1, offset);
      inst->set_operand(2, base);
      inst->set_operand(3, getAlign(isOperandGR(inst->operand(0)) ? 8 : 4));
      break;
    }
    case SD:
    case SW:
    case SH:
    case SB:
    case FSW: {
      /**
       * sw rs2, offset(rs1)
       *
       * sw rs2, obj
       * ->
       * sw rs2, offset(sp)
       */
      if (debugLISO) std::cout << "sw rs2, offset(rs1)" << std::endl;
      inst->set_operand(1, offset);
      inst->set_operand(2, base);
      break;
    }
    case LD:
    case LW:
    // case LWU:
    case LH:
    case LHU:
    case LB:
    case LBU:
    case FLW: {
      /**
       * lw rd, offset(rs1)
       *
       * lw rd, obj
       * ->
       * lw rd, offset(sp)
       */
      if (debugLISO) std::cout << "lw rd, offset(rs1)" << std::endl;
      //! careful with the order of operands,
      //! sw and lw have different order
      inst->set_operand(1, offset);
      inst->set_operand(2, base);
      break;
    }
    default:
      std::cerr << "Unsupported instruction for legalizeInstWithStackOperand" << std::endl;
  }

  // lw rd, offset(rs1) or sw rs2, offset(rs1)
  // their offset may overflow, need to adjust it
  // RISCV::adjust_reg()
}

void RISCVISelInfo::postLegalizeInst(const InstLegalizeContext& ctx) const {
  auto& inst = ctx.inst;
  switch (inst->opcode()) {
    case InstCopy:
    case InstCopyFromReg:
    case InstCopyToReg: {
      const auto dst = inst->operand(0);
      const auto src = inst->operand(1);
      if (isOperandIReg(dst) && isOperandIReg(src)) {
        inst->set_opcode(MV);
      } else if (isOperandFPR(dst) && isOperandFPR(src)) {
        inst->set_opcode(FMV_S);
      } else {
        std::cerr << "Unsupported InstCopyToReg for postLegalizeInst" << std::endl;
        assert(false);
      }
      break;
    }
    case InstLoadImm: {
      const auto dst = inst->operand(0);
      const auto src = inst->operand(1);
      if (isOperandIReg(dst)) {
        if (isZero(src)) {
          inst->set_opcode(MV);
          inst->set_operand(1, getZero(src));
          return;
        } else if (isOperandImm12(src)) {
          inst->set_opcode(LoadImm12);
          return;
        } else if (isOperandImm32(src)) {
          inst->set_opcode(LoadImm32);
          return;
        }
        std::cerr << "Unsupported InstLoadImm for postLegalizeInst" << std::endl;
        assert(false);
      }
    }
    default:
      std::cerr << "Unsupported opcode for postLegalizeInst" << std::endl;
  }
}

MIROperand RISCVISelInfo::materializeFPConstant(float fpVal, LoweringContext& loweringCtx) const {
  const auto val = fpVal;
  uint32_t rep;
  memcpy(&rep, &val, sizeof(float));
  if (rep == 0) {
    // fmv.w.x
    const auto dst = loweringCtx.newVReg(OperandType::Float32);

    loweringCtx.emitInstBeta(FMV_W_X, {dst, MIROperand::asISAReg(RISCV::X0, OperandType::Int32)});
    return dst;
  }
  if ((rep & 0xfff) == 0) {
    // lui + fmv.w.x
    const auto high = (rep >> 12);
    const auto gpr = loweringCtx.newVReg(OperandType::Int32);
    const auto fpr = loweringCtx.newVReg(OperandType::Float32);

    loweringCtx.emitInstBeta(LUI, {gpr, MIROperand::asImm(high, OperandType::Int32)});

    loweringCtx.emitInstBeta(FMV_W_X, {fpr, gpr});
    return fpr;
  }
  return MIROperand();
}
}  // namespace mir::RISCV