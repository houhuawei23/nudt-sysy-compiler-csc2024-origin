// Automatically generated file, do not edit!

#pragma once
#include "mir/mir.hpp"

#define RISCV_NAMESPACE_BEGIN namespace mir::RISCV {
#define RISCV_NAMESPACE_END }

RISCV_NAMESPACE_BEGIN
enum RISCVInst {
  RISCVInstBegin = ISASpecificBegin,

  BEQ,
  BNE,
  BLE,
  BGT,
  BLT,
  BGE,
  BLEU,
  BGTU,
  BLTU,
  BGEU,
  ADD,
  ADDW,
  SUB,
  SUBW,
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
  JAL,
  J,
  RET,
  LUI,
  AUIPC,
  LLA,
  MUL,
  MULW,
  MULH,
  MULHSU,
  MULHU,
  DIV,
  DIVW,
  DIVU,
  REM,
  REMW,
  REMU,
  LR,
  SC,
  AMOSWAP,
  AMOADD,
  AMOAND,
  AMOOR,
  AMOXOR,
  FLW,
  FSW,
  LoadImm12,
  LoadImm32,
  LoadImm64,
  MV,

  RISCVInstEnd
};

TargetInstInfo& getRISCVInstInfo();

RISCV_NAMESPACE_END