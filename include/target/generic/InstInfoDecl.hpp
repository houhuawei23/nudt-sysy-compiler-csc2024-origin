// Automatically generated file, do not edit!

#pragma once
#include "mir/mir.hpp"

#define GENERIC_NAMESPACE_BEGIN namespace mir::GENERIC {
#define GENERIC_NAMESPACE_END }

GENERIC_NAMESPACE_BEGIN
enum GENERICInst {
    GENERICInstBegin = ISASpecificBegin,
    Add,
    Sub,
    Mul,
    UDiv,
    URem,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
    SMin,
    SMax,
    FAdd,
    FSub,
    FMul,
    FDiv,
    Neg,
    Abs,
    SExt,
    ZExt,
    Trunc,
    F2U,
    F2S,
    U2F,
    S2F,
    FCast,
    Load,
    Store,
    Jump,
    Branch,
    Copy,
    LoadGlobalAddress,
    LoadImm,
    LoadStackObjAddr,
    GENERICInstEnd
};

TargetInstInfo& getGENERICInstInfo();

GENERIC_NAMESPACE_END