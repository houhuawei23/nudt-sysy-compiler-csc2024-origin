- The base integer ISA is named “I” (prefixed by RV32 or RV64 depending on integer registerwidth), and contains integer computational instructions, integer loads, integer stores, and control flow instructions. 
- The standard integer multiplication and division extension is named “M”, andadds instructions to multiply and divide values held in the integer registers. 
- The standard atomic instruction extension, denoted by “A”, adds instructions that atomically read, modify, and write
memory for inter-processor synchronization. 
- The standard single-precision floating-point extension, denoted by “F”, adds floating-point registers, single-precision computational instructions, and single-precision loads and stores. 
- The standard double-precision floating-point extension, denoted by “D”, expands the floating-point registers, and adds double-precision computational instructions, loads, and stores. 
- The standard “C” compressed instruction extension provides narrower 16-bit forms of common instructions.


`lla rd, symbol` x[rd] = &symbol
Load Local Address. **Pseudoinstruction**, RV32I and RV64I.
Loads the address of symbol into x[rd]. 
Expands into `auipc rd, offsetHi` then `addi rd, rd, offsetLo`.

`add rd, rs1, rs2`
    Add. R-type, RV32I and RV64I
`addi rd, rs1, immediate`
    Add Immediate. I-type, RV32I and RV64I.
`addiw rd, rs1, immediate`
    Add Word Immediate. I-type, RV64I only
`addw rd, rs1, rs2`
    Add Word. R-type, RV64I only.
`amoadd.d rd, rs2, (rs1)` 
    x[rd] = AMO64(M[x[rs1]] + x[rs2])
`and rd, rs1, rs2 `
    x[rd] = x[rs1] & x[rs2]
    AND. R-type, RV32I and RV64I.
`andi rd, rs1, immediate `
    x[rd] = x[rs1] & sext(immediate)
    AND Immediate. I-type, RV32I and RV64I.
`auipc rd, immediate `
    x[rd] = pc + sext(immediate[31:12] << 12)
    Add Upper Immediate to PC. U-type, RV32I and RV64I.

`beq rs1, rs2, offset` 000
    if (rs1 == rs2) pc += sext(offset)
    Branch if Equal. B-type, RV32I and RV64I.
`beqz rs1, offset` 
    if (rs1 == 0) pc += sext(offset)
    Branch if Equal to Zero. **Pseudoinstruction**, RV32I and RV64I.
`bge rs1, rs2, offset` 101 
    if (rs1 ≥s rs2) pc += sext(offset)
    Branch if Greater Than or Equal. B-type, RV32I and RV64I.
`bgeu rs1, rs2, offset` 111
    if (rs1 ≥u rs2) pc += sext(offset)
    Branch if Greater Than or Equal, Unsigned. B-type, RV32I and RV64I.
`bgez rs1, offset` 
    if (rs1 ≥s 0) pc += sext(offset)
    Branch if Greater Than or Equal to Zero. **Pseudoinstruction**, RV32I and RV64I.
`bgt rs1, rs2, offset` 
    if (rs1 >s rs2) pc += sext(offset)
    Branch if Greater Than. **Pseudoinstruction**, RV32I and RV64I.
`bgtu rs1, rs2, offset` 
    if (rs1 >u rs2) pc += sext(offset)
    Branch if Greater Than, Unsigned. **Pseudoinstruction**, RV32I and RV64I.
blt rs1, rs2, offset 
    if (rs1 <s rs2) pc += sext(offset)
    Branch if Less Than. B-type, RV32I and RV64I.
`bne rs1, rs2, offset` 001
    if (rs1 6= rs2) pc += sext(offset)
    Branch if Not Equal. B-type, RV32I and RV64I


## Pseudoinstructions

Pseudoinstruction Base Instruction(s) Meaning
- `la rd, symbol` -> `auipc rd, symbol[31:12]`/`addi rd, rd, symbol[11:0]`
  - Load address 
- `l{b|h|w|d} rd`


, symbol auipc rd, symbol[31:12] Load global l{b|h|w|d} rd, symbol[11:0](rd)
s{b|h|w|d} rd, symbol, rt auipc rt, symbol[31:12] Store global s{b|h|w|d} rd, symbol[11:0](rt)
fl{w|d} rd, symbol, rt auipc rt, symbol[31:12] Floating-point load global fl{w|d} rd, symbol[11:0](rt)
fs{w|d} rd, symbol, rt auipc rt, symbol[31:12] Floating-point store global fs{w|d} rd, symbol[11:0](rt)





nanke:

RType: `$Mnemonic:Template $Rd:GPR[Def], $Rs1:GPR[Use], $Rs2:GPR[Use]`
R2Type: `$Mnemonic:Template $Rd:GPR[Def], $Rs1:GPR[Use]`
IType: `$Mnemonic:Template $Rd:GPR[Def], $Rs1:GPR[Use], $Imm:Imm12[Metadata]`
UType: `$Mnemonic:Template $Rd:GPR[Def], $Imm:UImm20[Metadata]`


RType: `$Mnemonic:Template $Rd:GPR[Def], $Rs1:GPR[Use], $Rs2:GPR[Use]`
    RRR: add, slt, sltu, and, or, xor, sll, srl, sub, sra
R2Type: `$Mnemonic:Template $Rd:GPR[Def], $Rs1:GPR[Use]`
IType: `$Mnemonic:Template $Rd:GPR[Def], $Rs1:GPR[Use], $Imm:Imm12[Metadata]`
    RII: addi, slti, sltiu, andi, ori, xori
    ShiftImm: `$Mnemonic:Template $Rd:GPR[Def], $Rs1:GPR[Use], $Imm:UImm6[Metadata]`
        slli, srli, srai
    Load: `$Mnemonic:Template $Rd:GPR[Def], $Imm:Imm12[Metadata]($Rs1:BaseLike[Use]) # $Alignment:Align[Metadata]`
        lb, lh, lw, lbu, lhu

UType: `$Mnemonic:Template $Rd:GPR[Def], $Imm:UImm20[Metadata]`
    RIU: lui, auipc
    JAL: `jal $Tgt:Reloc[Metadata]` 
        Flag: [Call]
    RET: `ret`  -> `jalr x0, x1, 0`
        Flag: [Terminator, Return, NoFallthrough]
    JR: `jr $Tgt:GPR[Use] # $Table:Reloc[Metadata]`
        Flag: [Terminator, IndirectJump, NoFallthrough]
    J: `j $Tgt:Reloc[Metadata]`
        Flag: [Terminator, Branch, NoFallthrough]
Branches: `"$Mnemonic:Template $Rs1:GPR[Use], $Rs2:GPR[Use], $Tgt:Reloc[Metadata] # $Prob:Prob[Metadata]"`
    Flag: [Terminator, Branch]
    beq, bne, blt, ble, bgt, bge, bltu, bgtu, bgeu

Store: `$Mnemonic:Template $Rs2:GPR[Use], $Imm:Imm12[Metadata]($Rs1:BaseLike[Use]) # $Alignment:Align[Metadata]`
    sb, sh, sw
    

