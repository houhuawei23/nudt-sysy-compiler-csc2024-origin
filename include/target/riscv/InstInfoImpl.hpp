// Automatically generated file, do not edit!

#pragma once
#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "target/riscv/InstInfoDecl.hpp"

TARGET_NAMESPACE_BEGIN

class RISCVInstInfoADD final : public InstInfo {
   public:
    RISCVInstInfoADD() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "add" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.ADD"; }
};

class RISCVInstInfoSUB final : public InstInfo {
   public:
    RISCVInstInfoSUB() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "sub" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.SUB"; }
};

class RISCVInstInfoXOR final : public InstInfo {
   public:
    RISCVInstInfoXOR() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "xor" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.XOR"; }
};

class RISCVInstInfoOR final : public InstInfo {
   public:
    RISCVInstInfoOR() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "or" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.OR"; }
};

class RISCVInstInfoAND final : public InstInfo {
   public:
    RISCVInstInfoAND() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "and" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.AND"; }
};

class RISCVInstInfoSLL final : public InstInfo {
   public:
    RISCVInstInfoSLL() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "sll" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.SLL"; }
};

class RISCVInstInfoSRL final : public InstInfo {
   public:
    RISCVInstInfoSRL() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "srl" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.SRL"; }
};

class RISCVInstInfoSRA final : public InstInfo {
   public:
    RISCVInstInfoSRA() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "sra" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.SRA"; }
};

class RISCVInstInfoSLT final : public InstInfo {
   public:
    RISCVInstInfoSLT() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "slt" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.SLT"; }
};

class RISCVInstInfoSLTU final : public InstInfo {
   public:
    RISCVInstInfoSLTU() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "sltu" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.SLTU"; }
};

class RISCVInstInfoADDI final : public InstInfo {
   public:
    RISCVInstInfoADDI() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "addi" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.ADDI"; }
};

class RISCVInstInfoXORI final : public InstInfo {
   public:
    RISCVInstInfoXORI() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "xori" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.XORI"; }
};

class RISCVInstInfoORI final : public InstInfo {
   public:
    RISCVInstInfoORI() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "ori" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.ORI"; }
};

class RISCVInstInfoANDI final : public InstInfo {
   public:
    RISCVInstInfoANDI() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "andi" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.ANDI"; }
};

class RISCVInstInfoSLTI final : public InstInfo {
   public:
    RISCVInstInfoSLTI() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "slti" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.SLTI"; }
};

class RISCVInstInfoSLTIU final : public InstInfo {
   public:
    RISCVInstInfoSLTIU() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "sltiu" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.SLTIU"; }
};

class RISCVInstInfoSLLI final : public InstInfo {
   public:
    RISCVInstInfoSLLI() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "slli" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.SLLI"; }
};

class RISCVInstInfoSRLI final : public InstInfo {
   public:
    RISCVInstInfoSRLI() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "srli" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.SRLI"; }
};

class RISCVInstInfoSRAI final : public InstInfo {
   public:
    RISCVInstInfoSRAI() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "srai" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.SRAI"; }
};

class RISCVInstInfoLB final : public InstInfo {
   public:
    RISCVInstInfoLB() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "lb" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone | InstFlagLoad; }

    std::string_view name() { return "RISCV.LB"; }
};

class RISCVInstInfoLH final : public InstInfo {
   public:
    RISCVInstInfoLH() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "lh" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone | InstFlagLoad; }

    std::string_view name() { return "RISCV.LH"; }
};

class RISCVInstInfoLW final : public InstInfo {
   public:
    RISCVInstInfoLW() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "lw" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone | InstFlagLoad; }

    std::string_view name() { return "RISCV.LW"; }
};

class RISCVInstInfoLBU final : public InstInfo {
   public:
    RISCVInstInfoLBU() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "lbu" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone | InstFlagLoad; }

    std::string_view name() { return "RISCV.LBU"; }
};

class RISCVInstInfoLHU final : public InstInfo {
   public:
    RISCVInstInfoLHU() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "lhu" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone | InstFlagLoad; }

    std::string_view name() { return "RISCV.LHU"; }
};

class RISCVInstInfoSB final : public InstInfo {
   public:
    RISCVInstInfoSB() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "sb" << inst.operand(0) << ", " << inst.operand(1) << "("
            << inst.operand(2) << ")";
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagUse;
            case 1:
                return OperandFlagMetadata;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone | InstFlagStore; }

    std::string_view name() { return "RISCV.SB"; }
};

class RISCVInstInfoSH final : public InstInfo {
   public:
    RISCVInstInfoSH() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "sh" << inst.operand(0) << ", " << inst.operand(1) << "("
            << inst.operand(2) << ")";
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagUse;
            case 1:
                return OperandFlagMetadata;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone | InstFlagStore; }

    std::string_view name() { return "RISCV.SH"; }
};

class RISCVInstInfoSW final : public InstInfo {
   public:
    RISCVInstInfoSW() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "sw" << inst.operand(0) << ", " << inst.operand(1) << "("
            << inst.operand(2) << ")";
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagUse;
            case 1:
                return OperandFlagMetadata;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone | InstFlagStore; }

    std::string_view name() { return "RISCV.SW"; }
};

class RISCVInstInfoBEQ final : public InstInfo {
   public:
    RISCVInstInfoBEQ() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "beq" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator;
    }

    std::string_view name() { return "RISCV.BEQ"; }
};

class RISCVInstInfoBNE final : public InstInfo {
   public:
    RISCVInstInfoBNE() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "bne" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator;
    }

    std::string_view name() { return "RISCV.BNE"; }
};

class RISCVInstInfoBLT final : public InstInfo {
   public:
    RISCVInstInfoBLT() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "blt" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator;
    }

    std::string_view name() { return "RISCV.BLT"; }
};

class RISCVInstInfoBGE final : public InstInfo {
   public:
    RISCVInstInfoBGE() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "bge" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator;
    }

    std::string_view name() { return "RISCV.BGE"; }
};

class RISCVInstInfoBLTU final : public InstInfo {
   public:
    RISCVInstInfoBLTU() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "bltu" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator;
    }

    std::string_view name() { return "RISCV.BLTU"; }
};

class RISCVInstInfoBGEU final : public InstInfo {
   public:
    RISCVInstInfoBGEU() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "bgeu" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator;
    }

    std::string_view name() { return "RISCV.BGEU"; }
};

class RISCVInstInfoJAL final : public InstInfo {
   public:
    RISCVInstInfoJAL() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "jal" << inst.operand(0) << ", " << inst.operand(1);
    }

    uint32_t operand_num() { return 2; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone | InstFlagCall; }

    std::string_view name() { return "RISCV.JAL"; }
};

class RISCVInstInfoJALR final : public InstInfo {
   public:
    RISCVInstInfoJALR() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "jalr" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone | InstFlagCall; }

    std::string_view name() { return "RISCV.JALR"; }
};

class RISCVInstInfoLUI final : public InstInfo {
   public:
    RISCVInstInfoLUI() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "lui" << inst.operand(0) << ", " << inst.operand(1);
    }

    uint32_t operand_num() { return 2; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone | InstFlagLoadConstant; }

    std::string_view name() { return "RISCV.LUI"; }
};

class RISCVInstInfoAUIPC final : public InstInfo {
   public:
    RISCVInstInfoAUIPC() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "auipc" << inst.operand(0) << ", " << inst.operand(1);
    }

    uint32_t operand_num() { return 2; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() {
        return InstFlagNone | InstFlagPCRel | InstFlagLoadConstant;
    }

    std::string_view name() { return "RISCV.AUIPC"; }
};

class RISCVInstInfoMUL final : public InstInfo {
   public:
    RISCVInstInfoMUL() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "mul" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.MUL"; }
};

class RISCVInstInfoMULH final : public InstInfo {
   public:
    RISCVInstInfoMULH() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "mulh" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.MULH"; }
};

class RISCVInstInfoMULHSU final : public InstInfo {
   public:
    RISCVInstInfoMULHSU() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "mulhsu" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.MULHSU"; }
};

class RISCVInstInfoMULHU final : public InstInfo {
   public:
    RISCVInstInfoMULHU() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "mulhu" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.MULHU"; }
};

class RISCVInstInfoDIV final : public InstInfo {
   public:
    RISCVInstInfoDIV() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "div" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.DIV"; }
};

class RISCVInstInfoDIVU final : public InstInfo {
   public:
    RISCVInstInfoDIVU() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "divu" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.DIVU"; }
};

class RISCVInstInfoREM final : public InstInfo {
   public:
    RISCVInstInfoREM() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "rem" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.REM"; }
};

class RISCVInstInfoREMU final : public InstInfo {
   public:
    RISCVInstInfoREMU() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "remu" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.REMU"; }
};

class RISCVInstInfoLR final : public InstInfo {
   public:
    RISCVInstInfoLR() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "lr.w" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.LR"; }
};

class RISCVInstInfoSC final : public InstInfo {
   public:
    RISCVInstInfoSC() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "sc.w" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.SC"; }
};

class RISCVInstInfoAMOSWAP final : public InstInfo {
   public:
    RISCVInstInfoAMOSWAP() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "amoswap.w" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.AMOSWAP"; }
};

class RISCVInstInfoAMOADD final : public InstInfo {
   public:
    RISCVInstInfoAMOADD() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "amoadd.w" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.AMOADD"; }
};

class RISCVInstInfoAMOAND final : public InstInfo {
   public:
    RISCVInstInfoAMOAND() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "amoand.w" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.AMOAND"; }
};

class RISCVInstInfoAMOOR final : public InstInfo {
   public:
    RISCVInstInfoAMOOR() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "amoor.w" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.AMOOR"; }
};

class RISCVInstInfoAMOXOR final : public InstInfo {
   public:
    RISCVInstInfoAMOXOR() = default;

    void print(std::ostream& out, MIRInst& inst, bool comment) {
        out << "amoxor.w" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }

    uint32_t operand_num() { return 3; }

    OperandFlag operand_flag(uint32_t idx) {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() { return InstFlagNone; }

    std::string_view name() { return "RISCV.AMOXOR"; }
};

class RISCVInstInfo final : public TargetInstInfo {
    RISCVInstInfoADD _instinfoADD;
    RISCVInstInfoSUB _instinfoSUB;
    RISCVInstInfoXOR _instinfoXOR;
    RISCVInstInfoOR _instinfoOR;
    RISCVInstInfoAND _instinfoAND;
    RISCVInstInfoSLL _instinfoSLL;
    RISCVInstInfoSRL _instinfoSRL;
    RISCVInstInfoSRA _instinfoSRA;
    RISCVInstInfoSLT _instinfoSLT;
    RISCVInstInfoSLTU _instinfoSLTU;
    RISCVInstInfoADDI _instinfoADDI;
    RISCVInstInfoXORI _instinfoXORI;
    RISCVInstInfoORI _instinfoORI;
    RISCVInstInfoANDI _instinfoANDI;
    RISCVInstInfoSLTI _instinfoSLTI;
    RISCVInstInfoSLTIU _instinfoSLTIU;
    RISCVInstInfoSLLI _instinfoSLLI;
    RISCVInstInfoSRLI _instinfoSRLI;
    RISCVInstInfoSRAI _instinfoSRAI;
    RISCVInstInfoLB _instinfoLB;
    RISCVInstInfoLH _instinfoLH;
    RISCVInstInfoLW _instinfoLW;
    RISCVInstInfoLBU _instinfoLBU;
    RISCVInstInfoLHU _instinfoLHU;
    RISCVInstInfoSB _instinfoSB;
    RISCVInstInfoSH _instinfoSH;
    RISCVInstInfoSW _instinfoSW;
    RISCVInstInfoBEQ _instinfoBEQ;
    RISCVInstInfoBNE _instinfoBNE;
    RISCVInstInfoBLT _instinfoBLT;
    RISCVInstInfoBGE _instinfoBGE;
    RISCVInstInfoBLTU _instinfoBLTU;
    RISCVInstInfoBGEU _instinfoBGEU;
    RISCVInstInfoJAL _instinfoJAL;
    RISCVInstInfoJALR _instinfoJALR;
    RISCVInstInfoLUI _instinfoLUI;
    RISCVInstInfoAUIPC _instinfoAUIPC;
    RISCVInstInfoMUL _instinfoMUL;
    RISCVInstInfoMULH _instinfoMULH;
    RISCVInstInfoMULHSU _instinfoMULHSU;
    RISCVInstInfoMULHU _instinfoMULHU;
    RISCVInstInfoDIV _instinfoDIV;
    RISCVInstInfoDIVU _instinfoDIVU;
    RISCVInstInfoREM _instinfoREM;
    RISCVInstInfoREMU _instinfoREMU;
    RISCVInstInfoLR _instinfoLR;
    RISCVInstInfoSC _instinfoSC;
    RISCVInstInfoAMOSWAP _instinfoAMOSWAP;
    RISCVInstInfoAMOADD _instinfoAMOADD;
    RISCVInstInfoAMOAND _instinfoAMOAND;
    RISCVInstInfoAMOOR _instinfoAMOOR;
    RISCVInstInfoAMOXOR _instinfoAMOXOR;

   public:
    RISCVInstInfo() = default;
    InstInfo& get_instinfo(uint32_t opcode) {
        switch (opcode) {
            case RISCVInst::ADD:
                return _instinfoADD;
            case RISCVInst::SUB:
                return _instinfoSUB;
            case RISCVInst::XOR:
                return _instinfoXOR;
            case RISCVInst::OR:
                return _instinfoOR;
            case RISCVInst::AND:
                return _instinfoAND;
            case RISCVInst::SLL:
                return _instinfoSLL;
            case RISCVInst::SRL:
                return _instinfoSRL;
            case RISCVInst::SRA:
                return _instinfoSRA;
            case RISCVInst::SLT:
                return _instinfoSLT;
            case RISCVInst::SLTU:
                return _instinfoSLTU;
            case RISCVInst::ADDI:
                return _instinfoADDI;
            case RISCVInst::XORI:
                return _instinfoXORI;
            case RISCVInst::ORI:
                return _instinfoORI;
            case RISCVInst::ANDI:
                return _instinfoANDI;
            case RISCVInst::SLTI:
                return _instinfoSLTI;
            case RISCVInst::SLTIU:
                return _instinfoSLTIU;
            case RISCVInst::SLLI:
                return _instinfoSLLI;
            case RISCVInst::SRLI:
                return _instinfoSRLI;
            case RISCVInst::SRAI:
                return _instinfoSRAI;
            case RISCVInst::LB:
                return _instinfoLB;
            case RISCVInst::LH:
                return _instinfoLH;
            case RISCVInst::LW:
                return _instinfoLW;
            case RISCVInst::LBU:
                return _instinfoLBU;
            case RISCVInst::LHU:
                return _instinfoLHU;
            case RISCVInst::SB:
                return _instinfoSB;
            case RISCVInst::SH:
                return _instinfoSH;
            case RISCVInst::SW:
                return _instinfoSW;
            case RISCVInst::BEQ:
                return _instinfoBEQ;
            case RISCVInst::BNE:
                return _instinfoBNE;
            case RISCVInst::BLT:
                return _instinfoBLT;
            case RISCVInst::BGE:
                return _instinfoBGE;
            case RISCVInst::BLTU:
                return _instinfoBLTU;
            case RISCVInst::BGEU:
                return _instinfoBGEU;
            case RISCVInst::JAL:
                return _instinfoJAL;
            case RISCVInst::JALR:
                return _instinfoJALR;
            case RISCVInst::LUI:
                return _instinfoLUI;
            case RISCVInst::AUIPC:
                return _instinfoAUIPC;
            case RISCVInst::MUL:
                return _instinfoMUL;
            case RISCVInst::MULH:
                return _instinfoMULH;
            case RISCVInst::MULHSU:
                return _instinfoMULHSU;
            case RISCVInst::MULHU:
                return _instinfoMULHU;
            case RISCVInst::DIV:
                return _instinfoDIV;
            case RISCVInst::DIVU:
                return _instinfoDIVU;
            case RISCVInst::REM:
                return _instinfoREM;
            case RISCVInst::REMU:
                return _instinfoREMU;
            case RISCVInst::LR:
                return _instinfoLR;
            case RISCVInst::SC:
                return _instinfoSC;
            case RISCVInst::AMOSWAP:
                return _instinfoAMOSWAP;
            case RISCVInst::AMOADD:
                return _instinfoAMOADD;
            case RISCVInst::AMOAND:
                return _instinfoAMOAND;
            case RISCVInst::AMOOR:
                return _instinfoAMOOR;
            case RISCVInst::AMOXOR:
                return _instinfoAMOXOR;
            default:
                assert(false && "Invalid opcode");
                // return TargetInstInfo::get_instinfo(opcode);
        }
    }
};

TargetInstInfo& getRISCVInstInfo() {
    static RISCVInstInfo instance;
    return instance;
}
TARGET_NAMESPACE_END