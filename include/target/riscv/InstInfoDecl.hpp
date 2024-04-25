// Automatically generated file, do not edit!

#pragma once
#include "mir/mir.hpp"

#define RISCV_NAMESPACE_BEGIN namespace mir::RISCV {
#define RISCV_NAMESPACE_END }

RISCV_NAMESPACE_BEGIN
enum RISCVInst {
  RISCVInstBegin = ISASpecificBegin,

  ADD,
  SUB,
  XOR,
  OR,
  AND,
  SLL,
  SRL,
  SRA,
  SLT,
  SLTU,
  ADDI,
  XORI,
  ORI,
  ANDI,
  SLTI,
  SLTIU,
  SLLI,
  SRLI,
  SRAI,
  LB,
  LH,
  LW,
  LBU,
  LHU,
  LD,
  SB,
  SH,
  SW,
  SD,
  BEQ,
  BNE,
  BLT,
  BGE,
  BLTU,
  BGEU,
  JAL,
  JALR,
  J,
  RET,
  LUI,
  AUIPC,
  MUL,
  MULH,
  MULHSU,
  MULHU,
  DIV,
  DIVU,
  REM,
  REMU,
  LR,
  SC,
  AMOSWAP,
  AMOADD,
  AMOAND,
  AMOOR,
  AMOXOR,
  LoadImm12,
  LoadImm32,
  LoadImm64,
  MV,

  RISCVInstEnd
};

static TargetInstInfo &getRISCVInstInfo();

RISCV_NAMESPACE_END