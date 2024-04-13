enum GenericInst {
    GenericInstBegin = ISASpecificBegin,
    Jump,    // Jump label, OperandFlagMetadata
    Branch,  // Branch cond, true_label, false_label; Use, Metadata, Metadata

    Unreachable,  // Unreachable

    Load,   // Load dst, src, imm; Def, Use, Metadata
    Store,  // Store dst, src, imm; Use, Use, Metadata
    Add,    // Add dst, src1, src2; Def, Use, Use
    Sub,    // Sub dst, src1, src2; Def, Use, Use
    Mul,    // Mul dst, src1, src2; Def, Use, Use
    UDiv,   // UDiv dst, src1, src2; Def, Use, Use
    URem,   // URem dst, src1, src2; Def, Use, Use
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,  // AShr dst, src1, src2; Def, Use, Use
    // Signed Div/Rem
    SDiv,  // SDiv dst, src1, src2, ?, ?, ?; Def, Use, Use, Metadata, Metadata,
           // Metadata
    SRem,  // same
    // dst, src1, src2
    SMin,
    SMax,
    // dst, src
    Neg,
    Abs,
    // dst, src1, src2
    FAdd,
    FSub,
    FMul,
    FDiv,
    FNeg,  // dst, src
    FAbs,  // dst, src
    FFma,  // dst, srcc1, srcc2, src3; Def, Use. Use, Use
    // dst, src1, src2, imm
    ICmp,
    FCmp,
    // dst, src
    SExt,
    ZExt,
    Trunc,
    F2U,
    F2S,
    U2F,
    S2F,
    FCast,
    Copy,                 // dst, src
    Select,               // dst, cond, true_val, false_val; Def, Use, Use, Use
    LoadGlobalAddress,    // dst, global_var_name; Def, Metadata
    LoadImm,              // dst, imm; Def, Metadata
    LoadStackObjectAddr,  // dst, stack; Def, Metadata
    CopyFromReg,          // dst, src; Def, Use
    CopyToReg,            // dst, src; Use, Def
    LoadImmToReg,         // dst, imm; Def, Metadata
    LoadRegFromStack,     // dst, stack; Def, Metadata
    StoreRegToStack,      // src, stack; Use, Metadata
    GenericInstEnd
};