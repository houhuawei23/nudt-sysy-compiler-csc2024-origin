// Automatically generated file, do not edit!

#pragma once
#include "mir/mir.hpp"

#define TARGET_NAMESPACE_BEGIN namespace mir::RISCV {
#define TARGET_NAMESPACE_END }

TARGET_NAMESPACE_BEGIN
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
    SB,
    SH,
    SW,
    BEQ,
    BNE,
    BLT,
    BGE,
    BLTU,
    BGEU,
    JAL,
    JALR,
    LUI,
    AUIPC,
    RISCVInstEnd
};

const TargetInstInfo& getRISCVInstInfo();

TARGET_NAMESPACE_END