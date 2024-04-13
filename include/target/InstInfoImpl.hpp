// Automatically generated file, do not edit!

#pragma once
#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "target/InstInfoDecl.hpp"

TARGET_NAMESPACE_BEGIN

class RISCVInstInfoADD final : public InstInfo {
   public:
    RISCVInstInfoADD() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::ADD";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSUB final : public InstInfo {
   public:
    RISCVInstInfoSUB() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SUB";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoXOR final : public InstInfo {
   public:
    RISCVInstInfoXOR() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::XOR";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoOR final : public InstInfo {
   public:
    RISCVInstInfoOR() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::OR";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoAND final : public InstInfo {
   public:
    RISCVInstInfoAND() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::AND";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSLL final : public InstInfo {
   public:
    RISCVInstInfoSLL() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SLL";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSRL final : public InstInfo {
   public:
    RISCVInstInfoSRL() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SRL";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSRA final : public InstInfo {
   public:
    RISCVInstInfoSRA() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SRA";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSLT final : public InstInfo {
   public:
    RISCVInstInfoSLT() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SLT";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSLTU final : public InstInfo {
   public:
    RISCVInstInfoSLTU() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SLTU";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoADDI final : public InstInfo {
   public:
    RISCVInstInfoADDI() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::ADDI";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoXORI final : public InstInfo {
   public:
    RISCVInstInfoXORI() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::XORI";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoORI final : public InstInfo {
   public:
    RISCVInstInfoORI() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::ORI";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoANDI final : public InstInfo {
   public:
    RISCVInstInfoANDI() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::ANDI";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSLTI final : public InstInfo {
   public:
    RISCVInstInfoSLTI() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SLTI";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSLTIU final : public InstInfo {
   public:
    RISCVInstInfoSLTIU() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SLTIU";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSLLI final : public InstInfo {
   public:
    RISCVInstInfoSLLI() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SLLI";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSRLI final : public InstInfo {
   public:
    RISCVInstInfoSRLI() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SRLI";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSRAI final : public InstInfo {
   public:
    RISCVInstInfoSRAI() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SRAI";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoLB final : public InstInfo {
   public:
    RISCVInstInfoLB() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::LB";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoLH final : public InstInfo {
   public:
    RISCVInstInfoLH() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::LH";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoLW final : public InstInfo {
   public:
    RISCVInstInfoLW() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::LW";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoLBU final : public InstInfo {
   public:
    RISCVInstInfoLBU() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::LBU";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoLHU final : public InstInfo {
   public:
    RISCVInstInfoLHU() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::LHU";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSB final : public InstInfo {
   public:
    RISCVInstInfoSB() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SB";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSH final : public InstInfo {
   public:
    RISCVInstInfoSH() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SH";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoSW final : public InstInfo {
   public:
    RISCVInstInfoSW() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::SW";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoBEQ final : public InstInfo {
   public:
    RISCVInstInfoBEQ() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::BEQ";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoBNE final : public InstInfo {
   public:
    RISCVInstInfoBNE() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::BNE";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoBLT final : public InstInfo {
   public:
    RISCVInstInfoBLT() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::BLT";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoBGE final : public InstInfo {
   public:
    RISCVInstInfoBGE() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::BGE";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoBLTU final : public InstInfo {
   public:
    RISCVInstInfoBLTU() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::BLTU";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoBGEU final : public InstInfo {
   public:
    RISCVInstInfoBGEU() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::BGEU";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoJAL final : public InstInfo {
   public:
    RISCVInstInfoJAL() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::JAL";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoJALR final : public InstInfo {
   public:
    RISCVInstInfoJALR() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::JALR";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoLUI final : public InstInfo {
   public:
    RISCVInstInfoLUI() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::LUI";
    }
    uint32_t operand_num() const { return 3; }
};

class RISCVInstInfoAUIPC final : public InstInfo {
   public:
    RISCVInstInfoAUIPC() = default;
    void print(std::ostream& out, const MIRInst& inst, bool comment) const {
        out << "RISCVInst::AUIPC";
    }
    uint32_t operand_num() const { return 3; }
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

            default:
                return TargetInstInfo::get_instinfo(opcode);
        }
    }
};

const TargetInstInfo& getRISCVInstInfo() {
    static RISCVInstInfo instance;
    return instance;
}
TARGET_NAMESPACE_END