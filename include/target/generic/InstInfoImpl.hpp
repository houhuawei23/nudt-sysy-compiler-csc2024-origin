// Automatically generated file, do not edit!

#pragma once
#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "target/generic/InstInfoDecl.hpp"

GENERIC_NAMESPACE_BEGIN

class GENERICInstInfoJump final : public InstInfo {
   public:
    GENERICInstInfoJump() = default;

    uint32_t operand_num() override { return 1; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.Jump"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Jump"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)};
    }
};

class GENERICInstInfoBranch final : public InstInfo {
   public:
    GENERICInstInfoBranch() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagUse;
            case 1:
                return OperandFlagMetadata;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.Branch"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Branch"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoLoad final : public InstInfo {
   public:
    GENERICInstInfoLoad() = default;

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

    std::string_view name() override { return "GENERIC.Load"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Load"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoStore final : public InstInfo {
   public:
    GENERICInstInfoStore() = default;

    uint32_t operand_num() override { return 3; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            case 0:
                return OperandFlagUse;
            case 1:
                return OperandFlagUse;
            case 2:
                return OperandFlagMetadata;
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.Store"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Store"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoAdd final : public InstInfo {
   public:
    GENERICInstInfoAdd() = default;

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

    std::string_view name() override { return "GENERIC.Add"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Add"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoSub final : public InstInfo {
   public:
    GENERICInstInfoSub() = default;

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

    std::string_view name() override { return "GENERIC.Sub"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Sub"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoMul final : public InstInfo {
   public:
    GENERICInstInfoMul() = default;

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

    std::string_view name() override { return "GENERIC.Mul"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Mul"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoUDiv final : public InstInfo {
   public:
    GENERICInstInfoUDiv() = default;

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

    std::string_view name() override { return "GENERIC.UDiv"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "UDiv"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoURem final : public InstInfo {
   public:
    GENERICInstInfoURem() = default;

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

    std::string_view name() override { return "GENERIC.URem"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "URem"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoAnd final : public InstInfo {
   public:
    GENERICInstInfoAnd() = default;

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

    std::string_view name() override { return "GENERIC.And"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "And"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoOr final : public InstInfo {
   public:
    GENERICInstInfoOr() = default;

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

    std::string_view name() override { return "GENERIC.Or"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Or"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoXor final : public InstInfo {
   public:
    GENERICInstInfoXor() = default;

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

    std::string_view name() override { return "GENERIC.Xor"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Xor"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoShl final : public InstInfo {
   public:
    GENERICInstInfoShl() = default;

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

    std::string_view name() override { return "GENERIC.Shl"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Shl"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoLShr final : public InstInfo {
   public:
    GENERICInstInfoLShr() = default;

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

    std::string_view name() override { return "GENERIC.LShr"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "LShr"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoAShr final : public InstInfo {
   public:
    GENERICInstInfoAShr() = default;

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

    std::string_view name() override { return "GENERIC.AShr"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "AShr"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoSMin final : public InstInfo {
   public:
    GENERICInstInfoSMin() = default;

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

    std::string_view name() override { return "GENERIC.SMin"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "SMin"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoSMax final : public InstInfo {
   public:
    GENERICInstInfoSMax() = default;

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

    std::string_view name() override { return "GENERIC.SMax"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "SMax"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoNeg final : public InstInfo {
   public:
    GENERICInstInfoNeg() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.Neg"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Neg"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoAbs final : public InstInfo {
   public:
    GENERICInstInfoAbs() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.Abs"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Abs"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoFAdd final : public InstInfo {
   public:
    GENERICInstInfoFAdd() = default;

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

    std::string_view name() override { return "GENERIC.FAdd"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "FAdd"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoFSub final : public InstInfo {
   public:
    GENERICInstInfoFSub() = default;

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

    std::string_view name() override { return "GENERIC.FSub"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "FSub"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoFMul final : public InstInfo {
   public:
    GENERICInstInfoFMul() = default;

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

    std::string_view name() override { return "GENERIC.FMul"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "FMul"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoFDiv final : public InstInfo {
   public:
    GENERICInstInfoFDiv() = default;

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

    std::string_view name() override { return "GENERIC.FDiv"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "FDiv"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(2)};
    }
};

class GENERICInstInfoSExt final : public InstInfo {
   public:
    GENERICInstInfoSExt() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.SExt"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "SExt"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoZExt final : public InstInfo {
   public:
    GENERICInstInfoZExt() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.ZExt"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "ZExt"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoTrunc final : public InstInfo {
   public:
    GENERICInstInfoTrunc() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.Trunc"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Trunc"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoF2U final : public InstInfo {
   public:
    GENERICInstInfoF2U() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.F2U"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "F2U"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoF2S final : public InstInfo {
   public:
    GENERICInstInfoF2S() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.F2S"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "F2S"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoU2F final : public InstInfo {
   public:
    GENERICInstInfoU2F() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.U2F"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "U2F"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoS2F final : public InstInfo {
   public:
    GENERICInstInfoS2F() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.S2F"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "S2F"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoFCast final : public InstInfo {
   public:
    GENERICInstInfoFCast() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.FCast"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "FCast"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoCopy final : public InstInfo {
   public:
    GENERICInstInfoCopy() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.Copy"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Copy"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoLoadGlobalAddress final : public InstInfo {
   public:
    GENERICInstInfoLoadGlobalAddress() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.LoadGlobalAddress"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "LoadGlobalAddress"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoLoadImm final : public InstInfo {
   public:
    GENERICInstInfoLoadImm() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.LoadImm"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "LoadImm"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoLoadStackObjectAddr final : public InstInfo {
   public:
    GENERICInstInfoLoadStackObjectAddr() = default;

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

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.LoadStackObjectAddr"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "LoadStackObjectAddr"
            << " " << mir::GENERIC::OperandDumper{inst.operand(0)} << ", "
            << mir::GENERIC::OperandDumper{inst.operand(1)};
    }
};

class GENERICInstInfoReturn final : public InstInfo {
   public:
    GENERICInstInfoReturn() = default;

    uint32_t operand_num() override { return 0; }

    OperandFlag operand_flag(uint32_t idx) override {
        switch (idx) {
            default:
                assert(false && "Invalid operand index");
        }
    }

    uint32_t inst_flag() override { return InstFlagNone; }

    std::string_view name() override { return "GENERIC.Return"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Return";
    }
};

class GENERICInstInfo final : public TargetInstInfo {
    GENERICInstInfoJump _instinfoJump;
    GENERICInstInfoBranch _instinfoBranch;
    GENERICInstInfoLoad _instinfoLoad;
    GENERICInstInfoStore _instinfoStore;
    GENERICInstInfoAdd _instinfoAdd;
    GENERICInstInfoSub _instinfoSub;
    GENERICInstInfoMul _instinfoMul;
    GENERICInstInfoUDiv _instinfoUDiv;
    GENERICInstInfoURem _instinfoURem;
    GENERICInstInfoAnd _instinfoAnd;
    GENERICInstInfoOr _instinfoOr;
    GENERICInstInfoXor _instinfoXor;
    GENERICInstInfoShl _instinfoShl;
    GENERICInstInfoLShr _instinfoLShr;
    GENERICInstInfoAShr _instinfoAShr;
    GENERICInstInfoSMin _instinfoSMin;
    GENERICInstInfoSMax _instinfoSMax;
    GENERICInstInfoNeg _instinfoNeg;
    GENERICInstInfoAbs _instinfoAbs;
    GENERICInstInfoFAdd _instinfoFAdd;
    GENERICInstInfoFSub _instinfoFSub;
    GENERICInstInfoFMul _instinfoFMul;
    GENERICInstInfoFDiv _instinfoFDiv;
    GENERICInstInfoSExt _instinfoSExt;
    GENERICInstInfoZExt _instinfoZExt;
    GENERICInstInfoTrunc _instinfoTrunc;
    GENERICInstInfoF2U _instinfoF2U;
    GENERICInstInfoF2S _instinfoF2S;
    GENERICInstInfoU2F _instinfoU2F;
    GENERICInstInfoS2F _instinfoS2F;
    GENERICInstInfoFCast _instinfoFCast;
    GENERICInstInfoCopy _instinfoCopy;
    GENERICInstInfoLoadGlobalAddress _instinfoLoadGlobalAddress;
    GENERICInstInfoLoadImm _instinfoLoadImm;
    GENERICInstInfoLoadStackObjectAddr _instinfoLoadStackObjectAddr;
    GENERICInstInfoReturn _instinfoReturn;

   public:
    GENERICInstInfo() = default;
    InstInfo& get_instinfo(uint32_t opcode) {
        switch (opcode) {
            case GENERICInst::Jump:
                return _instinfoJump;
            case GENERICInst::Branch:
                return _instinfoBranch;
            case GENERICInst::Unreachable:
                break; /* not supported */
            case GENERICInst::Load:
                return _instinfoLoad;
            case GENERICInst::Store:
                return _instinfoStore;
            case GENERICInst::Add:
                return _instinfoAdd;
            case GENERICInst::Sub:
                return _instinfoSub;
            case GENERICInst::Mul:
                return _instinfoMul;
            case GENERICInst::UDiv:
                return _instinfoUDiv;
            case GENERICInst::URem:
                return _instinfoURem;
            case GENERICInst::And:
                return _instinfoAnd;
            case GENERICInst::Or:
                return _instinfoOr;
            case GENERICInst::Xor:
                return _instinfoXor;
            case GENERICInst::Shl:
                return _instinfoShl;
            case GENERICInst::LShr:
                return _instinfoLShr;
            case GENERICInst::AShr:
                return _instinfoAShr;
            case GENERICInst::SDiv:
                break; /* not supported */
            case GENERICInst::SRem:
                break; /* not supported */
            case GENERICInst::SMin:
                return _instinfoSMin;
            case GENERICInst::SMax:
                return _instinfoSMax;
            case GENERICInst::Neg:
                return _instinfoNeg;
            case GENERICInst::Abs:
                return _instinfoAbs;
            case GENERICInst::FAdd:
                return _instinfoFAdd;
            case GENERICInst::FSub:
                return _instinfoFSub;
            case GENERICInst::FMul:
                return _instinfoFMul;
            case GENERICInst::FDiv:
                return _instinfoFDiv;
            case GENERICInst::FNeg:
                break; /* not supported */
            case GENERICInst::FAbs:
                break; /* not supported */
            case GENERICInst::FFma:
                break; /* not supported */
            case GENERICInst::ICmp:
                break; /* not supported */
            case GENERICInst::FCmp:
                break; /* not supported */
            case GENERICInst::SExt:
                return _instinfoSExt;
            case GENERICInst::ZExt:
                return _instinfoZExt;
            case GENERICInst::Trunc:
                return _instinfoTrunc;
            case GENERICInst::F2U:
                return _instinfoF2U;
            case GENERICInst::F2S:
                return _instinfoF2S;
            case GENERICInst::U2F:
                return _instinfoU2F;
            case GENERICInst::S2F:
                return _instinfoS2F;
            case GENERICInst::FCast:
                return _instinfoFCast;
            case GENERICInst::Copy:
                return _instinfoCopy;
            case GENERICInst::Select:
                break; /* not supported */
            case GENERICInst::LoadGlobalAddress:
                return _instinfoLoadGlobalAddress;
            case GENERICInst::LoadImm:
                return _instinfoLoadImm;
            case GENERICInst::LoadStackObjectAddr:
                return _instinfoLoadStackObjectAddr;
            case GENERICInst::CopyFromReg:
                break; /* not supported */
            case GENERICInst::CopyToReg:
                break; /* not supported */
            case GENERICInst::LoadImmToReg:
                break; /* not supported */
            case GENERICInst::LoadRegFromStack:
                break; /* not supported */
            case GENERICInst::StoreRegToStack:
                break; /* not supported */
            case GENERICInst::Return:
                return _instinfoReturn;
            default:
                return TargetInstInfo::get_instinfo(opcode);
        }
    }
};

TargetInstInfo& getGENERICInstInfo() {
    static GENERICInstInfo instance;
    return instance;
}
GENERIC_NAMESPACE_END