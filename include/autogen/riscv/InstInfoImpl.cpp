// Automatically generated file, do not edit!

#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "autogen/riscv/InstInfoDecl.hpp"

RISCV_NAMESPACE_BEGIN

class RISCVInstInfoBEQ final : public InstInfo {
   public:
    RISCVInstInfoBEQ() = default;

    uint32_t operand_num() override { return 4; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagUse;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            case 3:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator;
    }

    std::string_view name() override { return "RISCV.BEQ"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "beq"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)} << "# "
            << mir::RISCV::OperandDumper{inst.operand(3)};
    }
};

class RISCVInstInfoBNE final : public InstInfo {
   public:
    RISCVInstInfoBNE() = default;

    uint32_t operand_num() override { return 4; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagUse;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            case 3:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator;
    }

    std::string_view name() override { return "RISCV.BNE"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "bne"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)} << "# "
            << mir::RISCV::OperandDumper{inst.operand(3)};
    }
};

class RISCVInstInfoBLT final : public InstInfo {
   public:
    RISCVInstInfoBLT() = default;

    uint32_t operand_num() override { return 4; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagUse;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            case 3:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator;
    }

    std::string_view name() override { return "RISCV.BLT"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "blt"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)} << "# "
            << mir::RISCV::OperandDumper{inst.operand(3)};
    }
};

class RISCVInstInfoBGE final : public InstInfo {
   public:
    RISCVInstInfoBGE() = default;

    uint32_t operand_num() override { return 4; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagUse;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            case 3:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator;
    }

    std::string_view name() override { return "RISCV.BGE"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "bge"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)} << "# "
            << mir::RISCV::OperandDumper{inst.operand(3)};
    }
};

class RISCVInstInfoBLTU final : public InstInfo {
   public:
    RISCVInstInfoBLTU() = default;

    uint32_t operand_num() override { return 4; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagUse;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            case 3:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator;
    }

    std::string_view name() override { return "RISCV.BLTU"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "bltu"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)} << "# "
            << mir::RISCV::OperandDumper{inst.operand(3)};
    }
};

class RISCVInstInfoBGEU final : public InstInfo {
   public:
    RISCVInstInfoBGEU() = default;

    uint32_t operand_num() override { return 4; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagUse;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            case 3:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator;
    }

    std::string_view name() override { return "RISCV.BGEU"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "bgeu"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)} << "# "
            << mir::RISCV::OperandDumper{inst.operand(3)};
    }
};

class RISCVInstInfoADD final : public InstInfo {
   public:
    RISCVInstInfoADD() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.ADD"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "add"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoADDW final : public InstInfo {
   public:
    RISCVInstInfoADDW() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.ADDW"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "addw"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSUB final : public InstInfo {
   public:
    RISCVInstInfoSUB() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SUB"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "sub"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSUBW final : public InstInfo {
   public:
    RISCVInstInfoSUBW() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SUBW"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "subw"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoXOR final : public InstInfo {
   public:
    RISCVInstInfoXOR() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.XOR"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "xor"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoOR final : public InstInfo {
   public:
    RISCVInstInfoOR() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.OR"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "or"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoAND final : public InstInfo {
   public:
    RISCVInstInfoAND() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.AND"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "and"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSLL final : public InstInfo {
   public:
    RISCVInstInfoSLL() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SLL"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "sll"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSRL final : public InstInfo {
   public:
    RISCVInstInfoSRL() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SRL"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "srl"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSRA final : public InstInfo {
   public:
    RISCVInstInfoSRA() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SRA"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "sra"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSLT final : public InstInfo {
   public:
    RISCVInstInfoSLT() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SLT"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "slt"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSLTU final : public InstInfo {
   public:
    RISCVInstInfoSLTU() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SLTU"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "sltu"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoADDI final : public InstInfo {
   public:
    RISCVInstInfoADDI() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.ADDI"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "addi"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoXORI final : public InstInfo {
   public:
    RISCVInstInfoXORI() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.XORI"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "xori"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoORI final : public InstInfo {
   public:
    RISCVInstInfoORI() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.ORI"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "ori"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoANDI final : public InstInfo {
   public:
    RISCVInstInfoANDI() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.ANDI"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "andi"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSLTI final : public InstInfo {
   public:
    RISCVInstInfoSLTI() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SLTI"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "slti"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSLTIU final : public InstInfo {
   public:
    RISCVInstInfoSLTIU() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SLTIU"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "sltiu"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSLLI final : public InstInfo {
   public:
    RISCVInstInfoSLLI() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SLLI"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "slli"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSRLI final : public InstInfo {
   public:
    RISCVInstInfoSRLI() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SRLI"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "srli"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSRAI final : public InstInfo {
   public:
    RISCVInstInfoSRAI() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SRAI"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "srai"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoLB final : public InstInfo {
   public:
    RISCVInstInfoLB() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone | InstFlagLoad; }

    std::string_view name() override { return "RISCV.LB"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "lb"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoLH final : public InstInfo {
   public:
    RISCVInstInfoLH() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone | InstFlagLoad; }

    std::string_view name() override { return "RISCV.LH"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "lh"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoLW final : public InstInfo {
   public:
    RISCVInstInfoLW() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone | InstFlagLoad; }

    std::string_view name() override { return "RISCV.LW"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "lw"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoLBU final : public InstInfo {
   public:
    RISCVInstInfoLBU() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone | InstFlagLoad; }

    std::string_view name() override { return "RISCV.LBU"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "lbu"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoLHU final : public InstInfo {
   public:
    RISCVInstInfoLHU() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone | InstFlagLoad; }

    std::string_view name() override { return "RISCV.LHU"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "lhu"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoLD final : public InstInfo {
   public:
    RISCVInstInfoLD() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone | InstFlagLoad; }

    std::string_view name() override { return "RISCV.LD"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "ld"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSB final : public InstInfo {
   public:
    RISCVInstInfoSB() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone | InstFlagStore; }

    std::string_view name() override { return "RISCV.SB"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "sb"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << "("
            << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
    }
};

class RISCVInstInfoSH final : public InstInfo {
   public:
    RISCVInstInfoSH() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone | InstFlagStore; }

    std::string_view name() override { return "RISCV.SH"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "sh"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << "("
            << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
    }
};

class RISCVInstInfoSW final : public InstInfo {
   public:
    RISCVInstInfoSW() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone | InstFlagStore; }

    std::string_view name() override { return "RISCV.SW"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "sw"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << "("
            << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
    }
};

class RISCVInstInfoSD final : public InstInfo {
   public:
    RISCVInstInfoSD() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone | InstFlagStore; }

    std::string_view name() override { return "RISCV.SD"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "sd"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << "("
            << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
    }
};

class RISCVInstInfoJAL final : public InstInfo {
   public:
    RISCVInstInfoJAL() = default;

    uint32_t operand_num() override { return 1; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override { return InstFlagNone | InstFlagCall; }

    std::string_view name() override { return "RISCV.JAL"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "jal"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)};
    }
};

class RISCVInstInfoJ final : public InstInfo {
   public:
    RISCVInstInfoJ() = default;

    uint32_t operand_num() override { return 1; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagBranch | InstFlagTerminator |
               InstFlagNoFallThrough;
    }

    std::string_view name() override { return "RISCV.J"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "j"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)};
    }
};

class RISCVInstInfoRET final : public InstInfo {
   public:
    RISCVInstInfoRET() = default;

    uint32_t operand_num() override { return 0; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagReturn | InstFlagTerminator |
               InstFlagNoFallThrough;
    }

    std::string_view name() override { return "RISCV.RET"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "ret";
    }
};

class RISCVInstInfoLUI final : public InstInfo {
   public:
    RISCVInstInfoLUI() = default;

    uint32_t operand_num() override { return 2; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagLoadConstant;
    }

    std::string_view name() override { return "RISCV.LUI"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "lui"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)};
    }
};

class RISCVInstInfoAUIPC final : public InstInfo {
   public:
    RISCVInstInfoAUIPC() = default;

    uint32_t operand_num() override { return 2; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagPCRel | InstFlagLoadConstant;
    }

    std::string_view name() override { return "RISCV.AUIPC"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "auipc"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)};
    }
};

class RISCVInstInfoMUL final : public InstInfo {
   public:
    RISCVInstInfoMUL() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.MUL"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "mul"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoMULH final : public InstInfo {
   public:
    RISCVInstInfoMULH() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.MULH"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "mulh"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoMULHSU final : public InstInfo {
   public:
    RISCVInstInfoMULHSU() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.MULHSU"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "mulhsu"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoMULHU final : public InstInfo {
   public:
    RISCVInstInfoMULHU() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.MULHU"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "mulhu"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoDIV final : public InstInfo {
   public:
    RISCVInstInfoDIV() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.DIV"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "div"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoDIVU final : public InstInfo {
   public:
    RISCVInstInfoDIVU() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.DIVU"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "divu"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoREM final : public InstInfo {
   public:
    RISCVInstInfoREM() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.REM"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "rem"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoREMU final : public InstInfo {
   public:
    RISCVInstInfoREMU() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.REMU"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "remu"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoLR final : public InstInfo {
   public:
    RISCVInstInfoLR() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.LR"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "lr.w"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoSC final : public InstInfo {
   public:
    RISCVInstInfoSC() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.SC"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "sc.w"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoAMOSWAP final : public InstInfo {
   public:
    RISCVInstInfoAMOSWAP() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.AMOSWAP"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "amoswap.w"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoAMOADD final : public InstInfo {
   public:
    RISCVInstInfoAMOADD() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.AMOADD"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "amoadd.w"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoAMOAND final : public InstInfo {
   public:
    RISCVInstInfoAMOAND() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.AMOAND"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "amoand.w"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoAMOOR final : public InstInfo {
   public:
    RISCVInstInfoAMOOR() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.AMOOR"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "amoor.w"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoAMOXOR final : public InstInfo {
   public:
    RISCVInstInfoAMOXOR() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "RISCV.AMOXOR"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "amoxor.w"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(2)};
    }
};

class RISCVInstInfoFLW final : public InstInfo {
   public:
    RISCVInstInfoFLW() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagMetadata;
            case 2:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override { return InstFlagNone | InstFlagLoad; }

    std::string_view name() override { return "RISCV.FLW"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "flw"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << "("
            << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
    }
};

class RISCVInstInfoFSW final : public InstInfo {
   public:
    RISCVInstInfoFSW() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
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

    uint32_t inst_flag() override { return InstFlagNone | InstFlagStore; }

    std::string_view name() override { return "RISCV.FSW"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "fsw"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)} << "("
            << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
    }
};

class RISCVInstInfoLoadImm12 final : public InstInfo {
   public:
    RISCVInstInfoLoadImm12() = default;

    uint32_t operand_num() override { return 2; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagLoadConstant;
    }

    std::string_view name() override { return "RISCV.LoadImm12"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "li"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)};
    }
};

class RISCVInstInfoLoadImm32 final : public InstInfo {
   public:
    RISCVInstInfoLoadImm32() = default;

    uint32_t operand_num() override { return 2; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagLoadConstant;
    }

    std::string_view name() override { return "RISCV.LoadImm32"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "li"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)};
    }
};

class RISCVInstInfoLoadImm64 final : public InstInfo {
   public:
    RISCVInstInfoLoadImm64() = default;

    uint32_t operand_num() override { return 2; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override {
        return InstFlagNone | InstFlagLoadConstant;
    }

    std::string_view name() override { return "RISCV.LoadImm64"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "li"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)};
    }
};

class RISCVInstInfoMV final : public InstInfo {
   public:
    RISCVInstInfoMV() = default;

    uint32_t operand_num() override { return 2; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagDef;
            case 1:
                return OperandFlagUse;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override { return InstFlagNone | InstFlagRegCopy; }

    std::string_view name() override { return "RISCV.MV"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "mv"
            << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
            << mir::RISCV::OperandDumper{inst.operand(1)};
    }
};

class RISCVInstInfo final : public TargetInstInfo {
    RISCVInstInfoBEQ _instinfoBEQ;
    RISCVInstInfoBNE _instinfoBNE;
    RISCVInstInfoBLT _instinfoBLT;
    RISCVInstInfoBGE _instinfoBGE;
    RISCVInstInfoBLTU _instinfoBLTU;
    RISCVInstInfoBGEU _instinfoBGEU;
    RISCVInstInfoADD _instinfoADD;
    RISCVInstInfoADDW _instinfoADDW;
    RISCVInstInfoSUB _instinfoSUB;
    RISCVInstInfoSUBW _instinfoSUBW;
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
    RISCVInstInfoLD _instinfoLD;
    RISCVInstInfoSB _instinfoSB;
    RISCVInstInfoSH _instinfoSH;
    RISCVInstInfoSW _instinfoSW;
    RISCVInstInfoSD _instinfoSD;
    RISCVInstInfoJAL _instinfoJAL;
    RISCVInstInfoJ _instinfoJ;
    RISCVInstInfoRET _instinfoRET;
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
    RISCVInstInfoFLW _instinfoFLW;
    RISCVInstInfoFSW _instinfoFSW;
    RISCVInstInfoLoadImm12 _instinfoLoadImm12;
    RISCVInstInfoLoadImm32 _instinfoLoadImm32;
    RISCVInstInfoLoadImm64 _instinfoLoadImm64;
    RISCVInstInfoMV _instinfoMV;

   public:
    RISCVInstInfo() = default;
    InstInfo& get_instinfo(uint32_t opcode) {
        switch (opcode) {
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
            case RISCVInst::ADD:
                return _instinfoADD;
            case RISCVInst::ADDW:
                return _instinfoADDW;
            case RISCVInst::SUB:
                return _instinfoSUB;
            case RISCVInst::SUBW:
                return _instinfoSUBW;
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
            case RISCVInst::LD:
                return _instinfoLD;
            case RISCVInst::SB:
                return _instinfoSB;
            case RISCVInst::SH:
                return _instinfoSH;
            case RISCVInst::SW:
                return _instinfoSW;
            case RISCVInst::SD:
                return _instinfoSD;
            case RISCVInst::JAL:
                return _instinfoJAL;
            case RISCVInst::J:
                return _instinfoJ;
            case RISCVInst::RET:
                return _instinfoRET;
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
            case RISCVInst::FLW:
                return _instinfoFLW;
            case RISCVInst::FSW:
                return _instinfoFSW;
            case RISCVInst::LoadImm12:
                return _instinfoLoadImm12;
            case RISCVInst::LoadImm32:
                return _instinfoLoadImm32;
            case RISCVInst::LoadImm64:
                return _instinfoLoadImm64;
            case RISCVInst::MV:
                return _instinfoMV;
            default:
                return TargetInstInfo::get_instinfo(opcode);
        }
    }
    bool matchBranch(MIRInst* inst, MIRBlock*& target, double& prob) override {
        auto& instInfo = get_instinfo(inst->opcode());
        if (requireFlag(instInfo.inst_flag(), InstFlagBranch)) {
            if (inst->opcode() < ISASpecificBegin) {
                return TargetInstInfo::matchBranch(inst, target, prob);
            }
            switch (inst->opcode()) {
                case BEQ:
                    target = dynamic_cast<MIRBlock*>(inst->operand(2)->reloc());
                    prob = inst->operand(3)->prob();
                    break;
                case BNE:
                    target = dynamic_cast<MIRBlock*>(inst->operand(2)->reloc());
                    prob = inst->operand(3)->prob();
                    break;
                case BLT:
                    target = dynamic_cast<MIRBlock*>(inst->operand(2)->reloc());
                    prob = inst->operand(3)->prob();
                    break;
                case BGE:
                    target = dynamic_cast<MIRBlock*>(inst->operand(2)->reloc());
                    prob = inst->operand(3)->prob();
                    break;
                case BLTU:
                    target = dynamic_cast<MIRBlock*>(inst->operand(2)->reloc());
                    prob = inst->operand(3)->prob();
                    break;
                case BGEU:
                    target = dynamic_cast<MIRBlock*>(inst->operand(2)->reloc());
                    prob = inst->operand(3)->prob();
                    break;
                case J:
                    target = dynamic_cast<MIRBlock*>(inst->operand(0)->reloc());
                    prob = 1.0;
                    break;
                default:
                    std::cerr << "Error: unknown branch instruction: "
                              << instInfo.name() << std::endl;
            }
            return true;
        }
        return false;
    }
};

TargetInstInfo& getRISCVInstInfo() {
    static RISCVInstInfo instance;
    return instance;
}
RISCV_NAMESPACE_END