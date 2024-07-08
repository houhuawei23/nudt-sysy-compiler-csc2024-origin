
// Automatically generated file, do not edit!
#include "autogen/riscv/ScheduleModelDecl.hpp"
#include <iostream>

RISCV_NAMESPACE_BEGIN

class RISCVScheduleModel_sifive_u74 final : public TargetScheduleModel {
  RISCVScheduleClassIntegerArithmetic mScheduleClass_IntegerArithmetic;

  RISCVScheduleClassSlowLoadImm mScheduleClass_SlowLoadImm;

  RISCVScheduleClassBranch mScheduleClass_Branch;

  RISCVScheduleClassLoadStore mScheduleClass_LoadStore;

  RISCVScheduleClassMulti mScheduleClass_Multi;

  RISCVScheduleClassDivRem mScheduleClass_DivRem;

  public:
  ScheduleClass& getInstScheClass(uint32_t opcode) override {
    switch (opcode) {
      case ADDI:
      case SLTI:
      case SLTIU:
      case ANDI:
      case ORI:
      case XORI:
      case SLLI:
      case SRLI:
      case SRAI:
      case LUI:
      case AUIPC:
      case ADD:
      case SLT:
      case SLTU:
      case AND:
      case OR:
      case XOR:
      case SLL:
      case SRL:
      case SUB:
      case SRA:
      case ADDW:
      case SUBW:
      case LoadImm12:
      case MV:
      case InstLoadStackObjectAddr:
      case InstCopy:
      case InstCopyFromReg:
      case InstCopyToReg:
        return mScheduleClass_IntegerArithmetic;

      case InstLoadImm:
      case LoadImm32:
      case LoadImm64:
        return mScheduleClass_SlowLoadImm;

      case JAL:
      case RET:
      case BEQ:
      case BNE:
      case BLT:
      case BLE:
      case BGT:
      case BGE:
      case BLTU:
      case BLEU:
      case BGTU:
      case BGEU:
      case J:
        return mScheduleClass_Branch;

      case LB:
      case LH:
      case LW:
      case LBU:
      case LHU:
      case SB:
      case SH:
      case SW:
      case LD:
      case SD:
      case InstStoreRegToStack:
        return mScheduleClass_LoadStore;

      case MUL:
      case MULH:
      case MULHSU:
      case MULHU:
      case MULW:
        return mScheduleClass_Multi;

      case DIVW:
      case REMW:
        return mScheduleClass_DivRem;

      default:
        std::cerr << "getInstScheClass() failed: op: " << opcode << std::endl;
        assert(false && "Invalid opcode");
    }
  }
  MicroArchInfo& getMicroArchInfo() override;
  bool peepholeOpt(MIRFunction& func, CodeGenContext& context) override;
  bool isExpensiveInst(MIRInst* inst, CodeGenContext& context) override;
};

TargetScheduleModel& getRISCVScheduleModel() {
  static RISCVScheduleModel_sifive_u74 model_sifive_u74;
  return model_sifive_u74;
}

RISCV_NAMESPACE_END