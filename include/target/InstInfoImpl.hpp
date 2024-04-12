// Automatically generated file, do not edit!

#pragma once
#include "target/InstInfoDecl.hpp

TARGET_NAMESPACE_BEGIN

class RISCVInstInfoADD final : public InstInfo {
public:
}

class RISCVInstInfoSUB final : public InstInfo {
public:
}

class RISCVInstInfoXOR final : public InstInfo {
public:
}

class RISCVInstInfoOR final : public InstInfo {
public:
}

class RISCVInstInfoAND final : public InstInfo {
public:
}

class RISCVInstInfoSLL final : public InstInfo {
public:
}

class RISCVInstInfoSRL final : public InstInfo {
public:
}

class RISCVInstInfoSRA final : public InstInfo {
public:
}

class RISCVInstInfoSLT final : public InstInfo {
public:
}

class RISCVInstInfoSLTU final : public InstInfo {
public:
}

class RISCVInstInfoADDI final : public InstInfo {
public:
}

class RISCVInstInfoXORI final : public InstInfo {
public:
}

class RISCVInstInfoORI final : public InstInfo {
public:
}

class RISCVInstInfoANDI final : public InstInfo {
public:
}

class RISCVInstInfoSLTI final : public InstInfo {
public:
}

class RISCVInstInfoSLTIU final : public InstInfo {
public:
}

class RISCVInstInfoSLLI final : public InstInfo {
public:
}

class RISCVInstInfoSRLI final : public InstInfo {
public:
}

class RISCVInstInfoSRAI final : public InstInfo {
public:
}

class RISCVInstInfoLB final : public InstInfo {
public:
}

class RISCVInstInfoLH final : public InstInfo {
public:
}

class RISCVInstInfoLW final : public InstInfo {
public:
}

class RISCVInstInfoLBU final : public InstInfo {
public:
}

class RISCVInstInfoLHU final : public InstInfo {
public:
}

class RISCVInstInfoSB final : public InstInfo {
public:
}

class RISCVInstInfoSH final : public InstInfo {
public:
}

class RISCVInstInfoSW final : public InstInfo {
public:
}

class RISCVInstInfoBEQ final : public InstInfo {
public:
}

class RISCVInstInfoBNE final : public InstInfo {
public:
}

class RISCVInstInfoBLT final : public InstInfo {
public:
}

class RISCVInstInfoBGE final : public InstInfo {
public:
}

class RISCVInstInfoBLTU final : public InstInfo {
public:
}

class RISCVInstInfoBGEU final : public InstInfo {
public:
}

class RISCVInstInfoJAL final : public InstInfo {
public:
}

class RISCVInstInfoJALR final : public InstInfo {
public:
}

class RISCVInstInfoLUI final : public InstInfo {
public:
}

class RISCVInstInfoAUIPC final : public InstInfo {
public:
}

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
  InstInfo &get_instinfo(uint32_t opcode) {
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
}

TARGET_NAMESPACE_END