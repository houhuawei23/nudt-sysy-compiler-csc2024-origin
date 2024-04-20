// Automatically generated file, do not edit!

#pragma once
#include "mir/mir.hpp"

#define GENERIC_NAMESPACE_BEGIN namespace mir::GENERIC {
#define GENERIC_NAMESPACE_END }

GENERIC_NAMESPACE_BEGIN
enum GENERICInst {
    GENERICInstBegin = ISASpecificBegin,

    Jump,
    Branch,
    Unreachable, /* not implemented yet */
    Load,
    Store,
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
    SDiv, /* not implemented yet */
    SRem, /* not implemented yet */
    SMin,
    SMax,
    Neg,
    Abs,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FNeg, /* not implemented yet */
    FAbs, /* not implemented yet */
    FFma, /* not implemented yet */
    ICmp, /* not implemented yet */
    FCmp, /* not implemented yet */
    SExt,
    ZExt,
    Trunc,
    F2U,
    F2S,
    U2F,
    S2F,
    FCast,
    Copy,
    Select, /* not implemented yet */
    LoadGlobalAddress,
    LoadImm,
    LoadStackObjectAddr,
    CopyFromReg,      /* not implemented yet */
    CopyToReg,        /* not implemented yet */
    LoadImmToReg,     /* not implemented yet */
    LoadRegFromStack, /* not implemented yet */
    StoreRegToStack,  /* not implemented yet */
    Return,

    GENERICInstEnd
};

TargetInstInfo& getGENERICInstInfo();

GENERIC_NAMESPACE_END