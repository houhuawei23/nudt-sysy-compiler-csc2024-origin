// Automatically generated file, do not edit!

#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "autogen/riscv/InstInfoDecl.hpp"

RISCV_NAMESPACE_BEGIN

class RISCVInstInfoBEQ final : public InstInfo {
public:
  RISCVInstInfoBEQ() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagBranch | InstFlagTerminator;
  }

  std::string_view name() const override { return "RISCV.BEQ"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "beq"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << " # "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoBNE final : public InstInfo {
public:
  RISCVInstInfoBNE() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagBranch | InstFlagTerminator;
  }

  std::string_view name() const override { return "RISCV.BNE"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "bne"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << " # "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoBLE final : public InstInfo {
public:
  RISCVInstInfoBLE() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagBranch | InstFlagTerminator;
  }

  std::string_view name() const override { return "RISCV.BLE"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "ble"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << " # "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoBGT final : public InstInfo {
public:
  RISCVInstInfoBGT() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagBranch | InstFlagTerminator;
  }

  std::string_view name() const override { return "RISCV.BGT"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "bgt"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << " # "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoBLT final : public InstInfo {
public:
  RISCVInstInfoBLT() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagBranch | InstFlagTerminator;
  }

  std::string_view name() const override { return "RISCV.BLT"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "blt"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << " # "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoBGE final : public InstInfo {
public:
  RISCVInstInfoBGE() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagBranch | InstFlagTerminator;
  }

  std::string_view name() const override { return "RISCV.BGE"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "bge"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << " # "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoBLEU final : public InstInfo {
public:
  RISCVInstInfoBLEU() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagBranch | InstFlagTerminator;
  }

  std::string_view name() const override { return "RISCV.BLEU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "bleu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << " # "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoBGTU final : public InstInfo {
public:
  RISCVInstInfoBGTU() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagBranch | InstFlagTerminator;
  }

  std::string_view name() const override { return "RISCV.BGTU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "bgtu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << " # "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoBLTU final : public InstInfo {
public:
  RISCVInstInfoBLTU() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagBranch | InstFlagTerminator;
  }

  std::string_view name() const override { return "RISCV.BLTU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "bltu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << " # "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoBGEU final : public InstInfo {
public:
  RISCVInstInfoBGEU() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagBranch | InstFlagTerminator;
  }

  std::string_view name() const override { return "RISCV.BGEU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "bgeu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << " # "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoFADD_S final : public InstInfo {
public:
  RISCVInstInfoFADD_S() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FADD_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fadd.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoFSUB_S final : public InstInfo {
public:
  RISCVInstInfoFSUB_S() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FSUB_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fsub.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoFMUL_S final : public InstInfo {
public:
  RISCVInstInfoFMUL_S() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FMUL_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fmul.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoFDIV_S final : public InstInfo {
public:
  RISCVInstInfoFDIV_S() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FDIV_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fdiv.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoFMIN_S final : public InstInfo {
public:
  RISCVInstInfoFMIN_S() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FMIN_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fmin.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoFMAX_S final : public InstInfo {
public:
  RISCVInstInfoFMAX_S() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FMAX_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fmax.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoFMADD_S final : public InstInfo {
public:
  RISCVInstInfoFMADD_S() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      case 2:
        return OperandFlagUse;
      case 3:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FMADD_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fmadd.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoFMSUB_S final : public InstInfo {
public:
  RISCVInstInfoFMSUB_S() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      case 2:
        return OperandFlagUse;
      case 3:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FMSUB_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fmsub.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoFNMADD_S final : public InstInfo {
public:
  RISCVInstInfoFNMADD_S() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      case 2:
        return OperandFlagUse;
      case 3:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FNMADD_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fnmadd.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoFNMSUB_S final : public InstInfo {
public:
  RISCVInstInfoFNMSUB_S() = default;

  uint32_t operand_num() const override { return 4; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      case 2:
        return OperandFlagUse;
      case 3:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FNMSUB_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fnmsub.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(3)};
  }
};

class RISCVInstInfoFNEG_S final : public InstInfo {
public:
  RISCVInstInfoFNEG_S() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FNEG_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fneg.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoFABS_S final : public InstInfo {
public:
  RISCVInstInfoFABS_S() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FABS_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fabs.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoFMV_S final : public InstInfo {
public:
  RISCVInstInfoFMV_S() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FMV_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fmv.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoFSQRT_S final : public InstInfo {
public:
  RISCVInstInfoFSQRT_S() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FSQRT_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fsqrt.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoFSGNJ_S final : public InstInfo {
public:
  RISCVInstInfoFSGNJ_S() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FSGNJ_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fsgnj.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoFCLASS_S final : public InstInfo {
public:
  RISCVInstInfoFCLASS_S() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FCLASS_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fclass.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoFMV_X_W final : public InstInfo {
public:
  RISCVInstInfoFMV_X_W() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FMV_X_W"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fmv.x.w"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoFCVT_WU_S final : public InstInfo {
public:
  RISCVInstInfoFCVT_WU_S() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FCVT_WU_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fcvt.wu.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoFMV_W_X final : public InstInfo {
public:
  RISCVInstInfoFMV_W_X() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FMV_W_X"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fmv.w.x"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoFCVT_S_W final : public InstInfo {
public:
  RISCVInstInfoFCVT_S_W() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FCVT_S_W"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fcvt.s.w"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoFCVT_S_WU final : public InstInfo {
public:
  RISCVInstInfoFCVT_S_WU() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FCVT_S_WU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fcvt.s.wu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoFEQ_S final : public InstInfo {
public:
  RISCVInstInfoFEQ_S() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FEQ_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "feq.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoFLT_S final : public InstInfo {
public:
  RISCVInstInfoFLT_S() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FLT_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "flt.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoFLE_S final : public InstInfo {
public:
  RISCVInstInfoFLE_S() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FLE_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fle.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoFCVT_W_S final : public InstInfo {
public:
  RISCVInstInfoFCVT_W_S() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.FCVT_W_S"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fcvt.w.s"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", rtz";
  }
};

class RISCVInstInfoADD final : public InstInfo {
public:
  RISCVInstInfoADD() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.ADD"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "add"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoADDW final : public InstInfo {
public:
  RISCVInstInfoADDW() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.ADDW"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "addw"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSUB final : public InstInfo {
public:
  RISCVInstInfoSUB() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SUB"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "sub"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSUBW final : public InstInfo {
public:
  RISCVInstInfoSUBW() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SUBW"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "subw"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoXOR final : public InstInfo {
public:
  RISCVInstInfoXOR() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.XOR"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "xor"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoOR final : public InstInfo {
public:
  RISCVInstInfoOR() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.OR"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "or"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoAND final : public InstInfo {
public:
  RISCVInstInfoAND() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.AND"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "and"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSLL final : public InstInfo {
public:
  RISCVInstInfoSLL() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SLL"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "sll"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSRL final : public InstInfo {
public:
  RISCVInstInfoSRL() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SRL"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "srl"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSRA final : public InstInfo {
public:
  RISCVInstInfoSRA() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SRA"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "sra"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSLT final : public InstInfo {
public:
  RISCVInstInfoSLT() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SLT"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "slt"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSLTU final : public InstInfo {
public:
  RISCVInstInfoSLTU() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SLTU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "sltu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoADDI final : public InstInfo {
public:
  RISCVInstInfoADDI() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.ADDI"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "addi"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoXORI final : public InstInfo {
public:
  RISCVInstInfoXORI() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.XORI"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "xori"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoORI final : public InstInfo {
public:
  RISCVInstInfoORI() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.ORI"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "ori"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoANDI final : public InstInfo {
public:
  RISCVInstInfoANDI() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.ANDI"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "andi"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSLTI final : public InstInfo {
public:
  RISCVInstInfoSLTI() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SLTI"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "slti"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSLTIU final : public InstInfo {
public:
  RISCVInstInfoSLTIU() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SLTIU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "sltiu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSLLI final : public InstInfo {
public:
  RISCVInstInfoSLLI() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SLLI"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "slli"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSRLI final : public InstInfo {
public:
  RISCVInstInfoSRLI() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SRLI"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "srli"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSRAI final : public InstInfo {
public:
  RISCVInstInfoSRAI() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SRAI"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "srai"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoLB final : public InstInfo {
public:
  RISCVInstInfoLB() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagLoad; }

  std::string_view name() const override { return "RISCV.LB"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "lb"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << "("
        << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
  }
};

class RISCVInstInfoLH final : public InstInfo {
public:
  RISCVInstInfoLH() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagLoad; }

  std::string_view name() const override { return "RISCV.LH"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "lh"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << "("
        << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
  }
};

class RISCVInstInfoLW final : public InstInfo {
public:
  RISCVInstInfoLW() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagLoad; }

  std::string_view name() const override { return "RISCV.LW"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "lw"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << "("
        << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
  }
};

class RISCVInstInfoLBU final : public InstInfo {
public:
  RISCVInstInfoLBU() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagLoad; }

  std::string_view name() const override { return "RISCV.LBU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "lbu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << "("
        << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
  }
};

class RISCVInstInfoLHU final : public InstInfo {
public:
  RISCVInstInfoLHU() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagLoad; }

  std::string_view name() const override { return "RISCV.LHU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "lhu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << "("
        << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
  }
};

class RISCVInstInfoLD final : public InstInfo {
public:
  RISCVInstInfoLD() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagLoad; }

  std::string_view name() const override { return "RISCV.LD"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "ld"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << "("
        << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
  }
};

class RISCVInstInfoSB final : public InstInfo {
public:
  RISCVInstInfoSB() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagStore; }

  std::string_view name() const override { return "RISCV.SB"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "sb"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << "("
        << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
  }
};

class RISCVInstInfoSH final : public InstInfo {
public:
  RISCVInstInfoSH() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagStore; }

  std::string_view name() const override { return "RISCV.SH"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "sh"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << "("
        << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
  }
};

class RISCVInstInfoSW final : public InstInfo {
public:
  RISCVInstInfoSW() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagStore; }

  std::string_view name() const override { return "RISCV.SW"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "sw"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << "("
        << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
  }
};

class RISCVInstInfoSD final : public InstInfo {
public:
  RISCVInstInfoSD() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagStore; }

  std::string_view name() const override { return "RISCV.SD"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "sd"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << "("
        << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
  }
};

class RISCVInstInfoJAL final : public InstInfo {
public:
  RISCVInstInfoJAL() = default;

  uint32_t operand_num() const override { return 1; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagMetadata;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagCall; }

  std::string_view name() const override { return "RISCV.JAL"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "jal"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)};
  }
};

class RISCVInstInfoJ final : public InstInfo {
public:
  RISCVInstInfoJ() = default;

  uint32_t operand_num() const override { return 1; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagMetadata;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagBranch | InstFlagTerminator |
           InstFlagNoFallThrough;
  }

  std::string_view name() const override { return "RISCV.J"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "j"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)};
  }
};

class RISCVInstInfoRET final : public InstInfo {
public:
  RISCVInstInfoRET() = default;

  uint32_t operand_num() const override { return 0; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagReturn | InstFlagTerminator |
           InstFlagNoFallThrough;
  }

  std::string_view name() const override { return "RISCV.RET"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "ret";
  }
};

class RISCVInstInfoLUI final : public InstInfo {
public:
  RISCVInstInfoLUI() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagMetadata;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagLoadConstant;
  }

  std::string_view name() const override { return "RISCV.LUI"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "lui"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoAUIPC final : public InstInfo {
public:
  RISCVInstInfoAUIPC() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagMetadata;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagPCRel | InstFlagLoadConstant;
  }

  std::string_view name() const override { return "RISCV.AUIPC"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "auipc"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoLLA final : public InstInfo {
public:
  RISCVInstInfoLLA() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagMetadata;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.LLA"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "lla"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoMUL final : public InstInfo {
public:
  RISCVInstInfoMUL() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.MUL"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "mul"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoMULW final : public InstInfo {
public:
  RISCVInstInfoMULW() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.MULW"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "mulw"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoMULH final : public InstInfo {
public:
  RISCVInstInfoMULH() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.MULH"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "mulh"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoMULHSU final : public InstInfo {
public:
  RISCVInstInfoMULHSU() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.MULHSU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "mulhsu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoMULHU final : public InstInfo {
public:
  RISCVInstInfoMULHU() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.MULHU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "mulhu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoDIV final : public InstInfo {
public:
  RISCVInstInfoDIV() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.DIV"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "div"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoDIVW final : public InstInfo {
public:
  RISCVInstInfoDIVW() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.DIVW"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "divw"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoDIVU final : public InstInfo {
public:
  RISCVInstInfoDIVU() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.DIVU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "divu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoREM final : public InstInfo {
public:
  RISCVInstInfoREM() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.REM"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "rem"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoREMW final : public InstInfo {
public:
  RISCVInstInfoREMW() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.REMW"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "remw"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoREMU final : public InstInfo {
public:
  RISCVInstInfoREMU() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.REMU"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "remu"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoLR final : public InstInfo {
public:
  RISCVInstInfoLR() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.LR"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "lr.w"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoSC final : public InstInfo {
public:
  RISCVInstInfoSC() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.SC"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "sc.w"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoAMOSWAP final : public InstInfo {
public:
  RISCVInstInfoAMOSWAP() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.AMOSWAP"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "amoswap.w"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoAMOADD final : public InstInfo {
public:
  RISCVInstInfoAMOADD() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.AMOADD"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "amoadd.w"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoAMOAND final : public InstInfo {
public:
  RISCVInstInfoAMOAND() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.AMOAND"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "amoand.w"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoAMOOR final : public InstInfo {
public:
  RISCVInstInfoAMOOR() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.AMOOR"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "amoor.w"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoAMOXOR final : public InstInfo {
public:
  RISCVInstInfoAMOXOR() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone; }

  std::string_view name() const override { return "RISCV.AMOXOR"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "amoxor.w"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(2)};
  }
};

class RISCVInstInfoFLW final : public InstInfo {
public:
  RISCVInstInfoFLW() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagLoad; }

  std::string_view name() const override { return "RISCV.FLW"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "flw"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << "("
        << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
  }
};

class RISCVInstInfoFSW final : public InstInfo {
public:
  RISCVInstInfoFSW() = default;

  uint32_t operand_num() const override { return 3; }

  OperandFlag operand_flag(uint32_t idx) const override {
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

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagStore; }

  std::string_view name() const override { return "RISCV.FSW"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "fsw"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)} << "("
        << mir::RISCV::OperandDumper{inst.operand(2)} << ")";
  }
};

class RISCVInstInfoLoadImm12 final : public InstInfo {
public:
  RISCVInstInfoLoadImm12() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagMetadata;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagLoadConstant;
  }

  std::string_view name() const override { return "RISCV.LoadImm12"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "li"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoLoadImm32 final : public InstInfo {
public:
  RISCVInstInfoLoadImm32() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagMetadata;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagLoadConstant;
  }

  std::string_view name() const override { return "RISCV.LoadImm32"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "li"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoLoadImm64 final : public InstInfo {
public:
  RISCVInstInfoLoadImm64() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagMetadata;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override {
    return InstFlagNone | InstFlagLoadConstant;
  }

  std::string_view name() const override { return "RISCV.LoadImm64"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "li"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfoMV final : public InstInfo {
public:
  RISCVInstInfoMV() = default;

  uint32_t operand_num() const override { return 2; }

  OperandFlag operand_flag(uint32_t idx) const override {
    switch (idx) {
      case 0:
        return OperandFlagDef;
      case 1:
        return OperandFlagUse;
      default:
        assert(false && "Invalid operand index");
    }
  }

  uint32_t inst_flag() const override { return InstFlagNone | InstFlagRegCopy; }

  std::string_view name() const override { return "RISCV.MV"; }

  void print(std::ostream& out, MIRInst& inst, bool comment) const override {
    out << "mv"
        << " " << mir::RISCV::OperandDumper{inst.operand(0)} << ", "
        << mir::RISCV::OperandDumper{inst.operand(1)};
  }
};

class RISCVInstInfo final : public TargetInstInfo {
  RISCVInstInfoBEQ _instinfoBEQ;
  RISCVInstInfoBNE _instinfoBNE;
  RISCVInstInfoBLE _instinfoBLE;
  RISCVInstInfoBGT _instinfoBGT;
  RISCVInstInfoBLT _instinfoBLT;
  RISCVInstInfoBGE _instinfoBGE;
  RISCVInstInfoBLEU _instinfoBLEU;
  RISCVInstInfoBGTU _instinfoBGTU;
  RISCVInstInfoBLTU _instinfoBLTU;
  RISCVInstInfoBGEU _instinfoBGEU;
  RISCVInstInfoFADD_S _instinfoFADD_S;
  RISCVInstInfoFSUB_S _instinfoFSUB_S;
  RISCVInstInfoFMUL_S _instinfoFMUL_S;
  RISCVInstInfoFDIV_S _instinfoFDIV_S;
  RISCVInstInfoFMIN_S _instinfoFMIN_S;
  RISCVInstInfoFMAX_S _instinfoFMAX_S;
  RISCVInstInfoFMADD_S _instinfoFMADD_S;
  RISCVInstInfoFMSUB_S _instinfoFMSUB_S;
  RISCVInstInfoFNMADD_S _instinfoFNMADD_S;
  RISCVInstInfoFNMSUB_S _instinfoFNMSUB_S;
  RISCVInstInfoFNEG_S _instinfoFNEG_S;
  RISCVInstInfoFABS_S _instinfoFABS_S;
  RISCVInstInfoFMV_S _instinfoFMV_S;
  RISCVInstInfoFSQRT_S _instinfoFSQRT_S;
  RISCVInstInfoFSGNJ_S _instinfoFSGNJ_S;
  RISCVInstInfoFCLASS_S _instinfoFCLASS_S;
  RISCVInstInfoFMV_X_W _instinfoFMV_X_W;
  RISCVInstInfoFCVT_WU_S _instinfoFCVT_WU_S;
  RISCVInstInfoFMV_W_X _instinfoFMV_W_X;
  RISCVInstInfoFCVT_S_W _instinfoFCVT_S_W;
  RISCVInstInfoFCVT_S_WU _instinfoFCVT_S_WU;
  RISCVInstInfoFEQ_S _instinfoFEQ_S;
  RISCVInstInfoFLT_S _instinfoFLT_S;
  RISCVInstInfoFLE_S _instinfoFLE_S;
  RISCVInstInfoFCVT_W_S _instinfoFCVT_W_S;
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
  RISCVInstInfoLLA _instinfoLLA;
  RISCVInstInfoMUL _instinfoMUL;
  RISCVInstInfoMULW _instinfoMULW;
  RISCVInstInfoMULH _instinfoMULH;
  RISCVInstInfoMULHSU _instinfoMULHSU;
  RISCVInstInfoMULHU _instinfoMULHU;
  RISCVInstInfoDIV _instinfoDIV;
  RISCVInstInfoDIVW _instinfoDIVW;
  RISCVInstInfoDIVU _instinfoDIVU;
  RISCVInstInfoREM _instinfoREM;
  RISCVInstInfoREMW _instinfoREMW;
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
  const InstInfo& get_instinfo(uint32_t opcode) const override {
    switch (opcode) {
      case RISCVInst::BEQ:
        return _instinfoBEQ;
      case RISCVInst::BNE:
        return _instinfoBNE;
      case RISCVInst::BLE:
        return _instinfoBLE;
      case RISCVInst::BGT:
        return _instinfoBGT;
      case RISCVInst::BLT:
        return _instinfoBLT;
      case RISCVInst::BGE:
        return _instinfoBGE;
      case RISCVInst::BLEU:
        return _instinfoBLEU;
      case RISCVInst::BGTU:
        return _instinfoBGTU;
      case RISCVInst::BLTU:
        return _instinfoBLTU;
      case RISCVInst::BGEU:
        return _instinfoBGEU;
      case RISCVInst::FADD_S:
        return _instinfoFADD_S;
      case RISCVInst::FSUB_S:
        return _instinfoFSUB_S;
      case RISCVInst::FMUL_S:
        return _instinfoFMUL_S;
      case RISCVInst::FDIV_S:
        return _instinfoFDIV_S;
      case RISCVInst::FMIN_S:
        return _instinfoFMIN_S;
      case RISCVInst::FMAX_S:
        return _instinfoFMAX_S;
      case RISCVInst::FMADD_S:
        return _instinfoFMADD_S;
      case RISCVInst::FMSUB_S:
        return _instinfoFMSUB_S;
      case RISCVInst::FNMADD_S:
        return _instinfoFNMADD_S;
      case RISCVInst::FNMSUB_S:
        return _instinfoFNMSUB_S;
      case RISCVInst::FNEG_S:
        return _instinfoFNEG_S;
      case RISCVInst::FABS_S:
        return _instinfoFABS_S;
      case RISCVInst::FMV_S:
        return _instinfoFMV_S;
      case RISCVInst::FSQRT_S:
        return _instinfoFSQRT_S;
      case RISCVInst::FSGNJ_S:
        return _instinfoFSGNJ_S;
      case RISCVInst::FCLASS_S:
        return _instinfoFCLASS_S;
      case RISCVInst::FMV_X_W:
        return _instinfoFMV_X_W;
      case RISCVInst::FCVT_WU_S:
        return _instinfoFCVT_WU_S;
      case RISCVInst::FMV_W_X:
        return _instinfoFMV_W_X;
      case RISCVInst::FCVT_S_W:
        return _instinfoFCVT_S_W;
      case RISCVInst::FCVT_S_WU:
        return _instinfoFCVT_S_WU;
      case RISCVInst::FEQ_S:
        return _instinfoFEQ_S;
      case RISCVInst::FLT_S:
        return _instinfoFLT_S;
      case RISCVInst::FLE_S:
        return _instinfoFLE_S;
      case RISCVInst::FCVT_W_S:
        return _instinfoFCVT_W_S;
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
      case RISCVInst::LLA:
        return _instinfoLLA;
      case RISCVInst::MUL:
        return _instinfoMUL;
      case RISCVInst::MULW:
        return _instinfoMULW;
      case RISCVInst::MULH:
        return _instinfoMULH;
      case RISCVInst::MULHSU:
        return _instinfoMULHSU;
      case RISCVInst::MULHU:
        return _instinfoMULHU;
      case RISCVInst::DIV:
        return _instinfoDIV;
      case RISCVInst::DIVW:
        return _instinfoDIVW;
      case RISCVInst::DIVU:
        return _instinfoDIVU;
      case RISCVInst::REM:
        return _instinfoREM;
      case RISCVInst::REMW:
        return _instinfoREMW;
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
  bool matchBranch(MIRInst* inst,
                   MIRBlock*& target,
                   double& prob) const override {
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
        case BLE:
          target = dynamic_cast<MIRBlock*>(inst->operand(2)->reloc());
          prob = inst->operand(3)->prob();
          break;
        case BGT:
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
        case BLEU:
          target = dynamic_cast<MIRBlock*>(inst->operand(2)->reloc());
          prob = inst->operand(3)->prob();
          break;
        case BGTU:
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
          std::cerr << "Error: unknown branch instruction: " << instInfo.name()
                    << std::endl;
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