#include "ir/ir.hpp"
#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "mir/target.hpp"
#include "target/riscv/RISCV.hpp"
#include "autogen/riscv/ScheduleModelDecl.hpp"
#include "target/riscv/RISCVScheduleModel.hpp"
#include "autogen/riscv/ScheduleModelImpl.hpp"
#include <deque>

namespace mir::RISCV {
MicroArchInfo& RISCVScheduleModel_sifive_u74::getMicroArchInfo() {
  static MicroArchInfo info{
    .enablePostRAScheduling = true,
    .hasRegRenaming = false,
    .hasMacroFusion = false,
    .issueWidth = 2,
    .outOfOrder = false,
    .hardwarePrefetch = true,
    .maxDataStreams = 8,
    .maxStrideByBytes = 256,
  };
  return info;
}

/*
 * @brief: branch2jump
 * @note: bxx zero, zero -> j
 */
static bool branch2jump(MIRFunction& func, const CodeGenContext& ctx) {
  bool modified = false;
  for (auto iter = func.blocks().begin(); iter != func.blocks().end(); ) {
    auto next = std::next(iter);

    for (auto inst : iter->get()->insts()) {
      const auto isZero = [](MIROperand& op) { return op.isReg() && op.reg() == RISCV::X0; };
      if (!(isZero(inst->operand(0)) && isZero(inst->operand(1)))) continue;
      
      uint32_t status = 0;
      switch (inst->opcode()) {
        case RISCV::BEQ:
        case RISCV::BLE:
        case RISCV::BGE:
        case RISCV::BLEU:
        case RISCV::BGEU:
          status = 1;
          break;
        case RISCV::BNE:
        case RISCV::BLT:
        case RISCV::BGT:
        case RISCV::BLTU:
        case RISCV::BGTU:
          status = 2;
          break;
        default:
          break;
      }

      if (status) {  // 跳转指令
        auto b_true = inst->operand(2);
        if (status == 1) {
          auto tmp = MIRInst{ RISCV::J }; tmp.set_operand(0, b_true);
          *inst = tmp;
        } else {
          if (ctx.flags.endsWithTerminator) {
            assert(next != func.blocks().end());
            auto b_false = MIROperand::asReloc(next->get());
            auto tmp = MIRInst{ RISCV::J }; tmp.set_operand(0, b_false);
            *inst = tmp;
          } else {
            inst->set_opcode(InstCopy); inst->set_operand(2, MIROperand{});
          }
        }

        modified = true;
      }
    }
    iter = next;
  }
  
  return modified;
}

/*
 * @brief: removeDeadBranch function
 * @note: 
 */
static bool removeDeadBranch(MIRFunction& func, const CodeGenContext& ctx) {
  if (ctx.flags.endsWithTerminator) return false;
  bool modified = false;
  for (auto& block : func.blocks()) {
    auto& instructions = block->insts();
    
    std::list<MIRInst*> branches;
    const auto invalidBranches = [&](MIRInst* inst) {
      auto& instInfo = ctx.instInfo.getInstInfo(inst);

      if (requireOneFlag(instInfo.inst_flag(), InstFlagCall)) {
        branches.clear();
      }

      for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
        if (instInfo.operand_flag(idx) & OperandFlagDef) {
          auto& op = inst->operand(idx);
          branches.remove_if(
            [&](MIRInst* branch) {
              return branch->operand(0) == op || branch->operand(1) == op;
            }
          );
        }
      }
    };

    for (auto iter = instructions.begin(); iter != instructions.end();) {
      auto inst = *iter;
      auto next = std::next(iter);

      invalidBranches(inst);

      const auto isBranch = [](uint32_t opcode) {
        switch (opcode) {
          case BEQ:
          case BNE:
          case BLT:
          case BLE:
          case BGT:
          case BGE:
          case BLTU:
          case BLEU:
          case BGTU:
          case BGEU: return true;
          default: return false;
        }
      };
      const auto isSameBranch = [](MIRInst* lhs, MIRInst* rhs) {
        if (lhs->opcode() != rhs->opcode()) return false;
        for (uint32_t idx = 0; idx < 2; idx++) {
          if (lhs->operand(idx) != rhs->operand(idx)) {
            return false;
          }
        }
        return true;
      };
    
      if (isBranch(inst->opcode())) {
        /* 条件跳转指令 */
        bool remove = false;
        for (auto rhs : branches) {
          if (isSameBranch(inst, rhs)) {
            remove = true;
          }
        }

        if (remove) {
          instructions.erase(iter);
          modified = true;
        } else {
          branches.push_back(inst);
        }
      }
      iter = next;
    }
  }

  return modified;
}

/*
 * @brief: largeImmMaterialize function
 */
static bool largeImmMaterialize(MIRBlock& block) {
  constexpr uint32_t windowSize = 4;
  std::deque<std::pair<intmax_t, MIROperand>> immQueue;
  const auto addImm = [&](intmax_t val, MIROperand& imm) {
    if (isOperandVReg(imm)) {
      immQueue.emplace_back(val, imm);
      while (immQueue.size() > windowSize) {
        immQueue.pop_front();
      }
    }
  };
  const auto reuseImm = [&](intmax_t val, MIRInst* inst) {
    for (auto iter = immQueue.begin(); iter != immQueue.end(); iter++) {
      auto& [rhs, rhsOp] = *iter;
      /* equal */
      if (val == rhs) {
        inst->set_opcode(InstCopy);
        inst->set_operand(1, rhsOp);
        return true;
      }

      /* shift */
      {
        const int32_t maxK = 8;
        for (auto k = 1; k <= maxK; k++) {
          if ((rhs << k) == val) {
            inst->set_opcode(SLLI);
            inst->set_operand(1, rhsOp); inst->set_operand(2, MIROperand::asImm(k, OperandType::Int32));
            return true;
          }
        }
      }
    
      /* bias */
      {
        const auto offset = val - rhs;
        if (isSignedImm<12>(offset)) {
          inst->set_opcode(ADDI);
          inst->set_operand(1, rhsOp); inst->set_operand(2, MIROperand::asImm(offset, OperandType::Int32));
          return true;
        }
      }

      /* negative */
      if (-rhs == val) {
        inst->set_opcode(SUB);
        inst->set_operand(1, MIROperand::asISAReg(RISCV::X0, rhsOp.type())); inst->set_operand(2, rhsOp);
        return true;
      }

      /* xor */
      if (~rhs == val) {
        inst->set_opcode(XORI);
        inst->set_operand(1, rhsOp); inst->set_operand(2, MIROperand::asImm(-1, OperandType::Int32));
        return true;
      }

      // /* *3 -> sh1add */
      // if (rhs * 3 == val) {
      //   inst->set_opcode(SH1ADD);
      //   inst->set_operand(1, rhsOp); inst->set_operand(2, rhsOp);
      //   return true;
      // }

      // /* *5 -> sh2add */
      // if (rhs * 5 == val) {
      //   inst->set_opcode(SH2ADD);
      //   inst->set_operand(1, rhsOp); inst->set_operand(2, rhsOp);
      //   return true;
      // }

      // /* *9 -> sh3add */
      // if (rhs * 9 == val) {
      //   inst->set_opcode(SH3ADD);
      //   inst->set_operand(1, rhsOp); inst->set_operand(2, rhsOp);
      // }
    }
    return false;
  };

  bool modified = false;
  for (auto inst : block.insts()) {
    if (inst->opcode() == LoadImm12) {
      addImm(inst->operand(1).imm(), inst->operand(0));
    } else if (inst->opcode() == LoadImm32 || inst->opcode() == LoadImm64) {
      const auto val = inst->operand(1).imm();
      if (reuseImm(val, inst)) {
        modified = true;
      }
      addImm(val, inst->operand(0));
    }
  }
  return modified;
}

static bool foldStoreZero(MIRFunction& func, MIRBlock& block, const CodeGenContext& ctx) {
  return false;
}

/*
 * @brief: simplifyOpWithZero function
 * @note:
 */
static bool simplifyOpWithZero(MIRFunction& func, const CodeGenContext&) {
  bool modified = false;
  for (auto& block : func.blocks()) {
    for (auto inst : block->insts()) {
      const auto isZero = [&](uint32_t idx) {
        auto& op = inst->operand(idx);
        return op.isReg() && op.reg() == RISCV::X0;
      };
      const auto getZero = [&](MIROperand& op) {
        return MIROperand::asISAReg(RISCV::X0, op.type());
      };
      const auto resetToZero = [&]{
        // auto tmpInst = MIRInst{ MoveGPR };
        modified = true;
      };
      const auto resetToCopy = [&](uint32_t idx) {
        // 
        modified = true;
      };
      const auto resetToSignExtend = [&](uint32_t idx) {
        // 
        modified = true;
      };
      const auto resetToLoadImm = [&](uint32_t idx) {
        auto tmpInst = MIRInst{ LoadImm12 };
        tmpInst.set_operand(0, inst->operand(0)); tmpInst.set_operand(1, inst->operand(idx));
        *inst = tmpInst;
        modified = true;
      };
      const auto resetToImm = [&](intmax_t imm) {
        assert(imm);
        auto tmpInst = MIRInst{ LoadImm12 };
        tmpInst.set_operand(0, inst->operand(0)); tmpInst.set_operand(1, MIROperand::asImm(imm, inst->operand(0).type()));
        modified = true;
      };
    
      switch (inst->opcode()) {
        case MUL:
        case MULW: {
          if (isZero(1) || isZero(2)) resetToZero();
          break;
        }
        case ADDI:
        case ADDIW: {
          if (isZero(1)) resetToLoadImm(2);
          break;
        }
        case SUBW: {
          if (isZero(2)) resetToSignExtend(1);
          break;
        }
        case REMW: {
          if (isZero(1)) resetToZero();
          break;
        }
        case SLLI:
        case SLLIW:
        case SRLI:
        case SRLIW:
        case SRAI:
        case SRAIW: {
          if (isZero(1)) resetToZero();
          break;
        }
        // case SH1ADD:
        // case SH2ADD:
        // case SH3ADD: {
        //   if (isZero(1)) resetToCopy(2);
        //   break;
        // }
        case ADD: {
          if (isZero(1)) resetToCopy(2);
          else if (isZero(2)) resetToCopy(1);
          break;
        }
        case ADDW: {
          if (isZero(1)) resetToSignExtend(2);
          else if (isZero(2)) resetToSignExtend(1);
          break;
        }
        case ANDI: {
          if (isZero(1)) resetToZero();
          break;
        }
        case SLTI: {
          if (isZero(1)) {
            if (0 < inst->operand(2).imm()) resetToImm(1);
            else resetToZero();
            break;
          }
        }
        case OR: {
          if (isZero(1)) resetToCopy(2);
          else if (isZero(2)) resetToCopy(1);
          break;
        }
        // case SEXT_W:
        // case ADD_UW: {
        //   if (isZero(1) && isZero(2)) resetToZero();
        //   else if (isZero(1)) resetToCopy(2);
        //   else if (isZero(2)) {
        //     // 
        //     modified = true;
        //   }
        //   break;
        // }
        default: break;
      }
    }
  }
  return modified;
}

static bool relaxWInst(MIRFunction& func, const CodeGenContext& ctx) {
  if (!ctx.flags.inSSAForm) return false;
  // treat inputs as signed 32-bit integers
  const auto isWUser = [](uint32_t opcode) {
    switch (opcode) {
      case ADDIW:
      // case ADD_UW:
      case ADDW:
      case SUBW:
      case MULW:
      case DIVW:
      case DIVUW:
      case REMW:
      case REMUW:
      // case SEXT_W:
      case SLLIW:
      case SRAIW:
      case SRLIW:
      case SLLW:
      case SRAW:
      case SRLW:
      case SW:
      case SH:
      case SB:
      case FCVT_S_W: return true;
      default: return false;
    }
  };
  // instructions should be relaxed
  const auto isInterestWInst = [](uint32_t opcode) {
    switch (opcode) {
      case ADDIW:
      // case SEXT_W:
      case SLLIW:
      case SRAIW: return true;
      default: return false;
    }
  };
  // need signed 32-bit inputs
  const auto needWProvider = [](uint32_t opcode) {
    switch (opcode) {
      case ADDIW:
      case ADDW:
      case SUBW:
      case SLLIW:
      case SRAIW:
      case SRLIW:
      case SLLW:
      case SRAW:
      case SRLW: return true;
      default: return false;
    }
  };

  std::unordered_map<MIRInst*, std::vector<MIRInst*>> users;
  for (auto& block : func.blocks()) {
    for (auto inst : block->insts()) {
      if (isInterestWInst(inst->opcode())) {
        // users[]
      }
    }
  }
}

bool RISCVScheduleModel_sifive_u74::peepholeOpt(MIRFunction& func, CodeGenContext& context) {
  bool modified = false;
  if (context.flags.preRA) {
    for (auto& block : func.blocks()) {
      if (!context.flags.inSSAForm) {
        
      }
    }
  }
  if (context.flags.postSA) {
    for (auto& block : func.blocks()) {

    }
  }
  modified |= branch2jump(func, context);
  modified |= removeDeadBranch(func, context);
  modified |= simplifyOpWithZero(func, context);
  modified |= relaxWInst(func, context);
  // modified |= removeSExtW(func, context);
  // modified |= 
  return modified;
}
bool RISCVScheduleModel_sifive_u74::isExpensiveInst(MIRInst* inst, CodeGenContext& context) {
  switch (inst->opcode()) {
    default: return true;
  }
}
}  // namespace mir::RISCV

//! Dont Change This Line!"