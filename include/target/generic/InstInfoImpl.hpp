// Automatically generated file, do not edit!

#pragma once
#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "target/generic/InstInfoDecl.hpp"

GENERIC_NAMESPACE_BEGIN

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

    std::string_view name() { return "GENERIC.Add"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Add" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.Sub"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Sub" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.Mul"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Mul" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.UDiv"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "UDiv" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.URem"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "URem" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.And"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "And" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.Or"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Or" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.Xor"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Xor" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.Shl"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Shl" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.LShr"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "LShr" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.AShr"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "AShr" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.SMin"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "SMin" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.SMax"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "SMax" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.FAdd"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "FAdd" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.FSub"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "FSub" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.FMul"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "FMul" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.FDiv"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "FDiv" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.Neg"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Neg" << inst.operand(0) << ", " << inst.operand(1);
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

    std::string_view name() { return "GENERIC.Abs"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Abs" << inst.operand(0) << ", " << inst.operand(1);
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

    std::string_view name() { return "GENERIC.SExt"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "SExt" << inst.operand(0) << ", " << inst.operand(1);
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

    std::string_view name() { return "GENERIC.ZExt"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "ZExt" << inst.operand(0) << ", " << inst.operand(1);
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

    std::string_view name() { return "GENERIC.Trunc"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Trunc" << inst.operand(0) << ", " << inst.operand(1);
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

    std::string_view name() { return "GENERIC.F2U"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "F2U" << inst.operand(0) << ", " << inst.operand(1);
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

    std::string_view name() { return "GENERIC.F2S"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "F2S" << inst.operand(0) << ", " << inst.operand(1);
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

    std::string_view name() { return "GENERIC.U2F"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "U2F" << inst.operand(0) << ", " << inst.operand(1);
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

    std::string_view name() { return "GENERIC.S2F"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "S2F" << inst.operand(0) << ", " << inst.operand(1);
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

    std::string_view name() { return "GENERIC.FCast"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "FCast" << inst.operand(0) << ", " << inst.operand(1);
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

    std::string_view name() { return "GENERIC.Load"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Load" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }
};

class GENERICInstInfoStore final : public InstInfo {
   public:
    GENERICInstInfoStore() = default;

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

    std::string_view name() { return "GENERIC.Store"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Store" << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
    }
};

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

    std::string_view name() { return "GENERIC.Jump"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Jump " << inst.operand(0);
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

    std::string_view name() { return "GENERIC.Branch"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Branch " << inst.operand(0) << ", " << inst.operand(1) << ", "
            << inst.operand(2);
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

    std::string_view name() { return "GENERIC.Copy"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "Copy " << inst.operand(0) << ", " << inst.operand(1);
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

    std::string_view name() { return "GENERIC.LoadGlobalAddress"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "LoadGlobalAddress " << inst.operand(0) << ", "
            << inst.operand(1);
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

    std::string_view name() { return "GENERIC.LoadImm"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "LoadImm " << inst.operand(0) << ", " << inst.operand(1);
    }
};

class GENERICInstInfoLoadStackObjAddr final : public InstInfo {
   public:
    GENERICInstInfoLoadStackObjAddr() = default;

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

    std::string_view name() { return "GENERIC.LoadStackObjAddr"; }

    void print(std::ostream& out, MIRInst& inst, bool comment) override {
        out << "LoadStackObjAddr " << inst.operand(0) << ", "
            << inst.operand(1);
    }
};

class GENERICInstInfo final : public TargetInstInfo {
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
    GENERICInstInfoFAdd _instinfoFAdd;
    GENERICInstInfoFSub _instinfoFSub;
    GENERICInstInfoFMul _instinfoFMul;
    GENERICInstInfoFDiv _instinfoFDiv;
    GENERICInstInfoNeg _instinfoNeg;
    GENERICInstInfoAbs _instinfoAbs;
    GENERICInstInfoSExt _instinfoSExt;
    GENERICInstInfoZExt _instinfoZExt;
    GENERICInstInfoTrunc _instinfoTrunc;
    GENERICInstInfoF2U _instinfoF2U;
    GENERICInstInfoF2S _instinfoF2S;
    GENERICInstInfoU2F _instinfoU2F;
    GENERICInstInfoS2F _instinfoS2F;
    GENERICInstInfoFCast _instinfoFCast;
    GENERICInstInfoLoad _instinfoLoad;
    GENERICInstInfoStore _instinfoStore;
    GENERICInstInfoJump _instinfoJump;
    GENERICInstInfoBranch _instinfoBranch;
    GENERICInstInfoCopy _instinfoCopy;
    GENERICInstInfoLoadGlobalAddress _instinfoLoadGlobalAddress;
    GENERICInstInfoLoadImm _instinfoLoadImm;
    GENERICInstInfoLoadStackObjAddr _instinfoLoadStackObjAddr;

   public:
    GENERICInstInfo() = default;
    InstInfo& get_instinfo(uint32_t opcode) {
        switch (opcode) {
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
            case GENERICInst::SMin:
                return _instinfoSMin;
            case GENERICInst::SMax:
                return _instinfoSMax;
            case GENERICInst::FAdd:
                return _instinfoFAdd;
            case GENERICInst::FSub:
                return _instinfoFSub;
            case GENERICInst::FMul:
                return _instinfoFMul;
            case GENERICInst::FDiv:
                return _instinfoFDiv;
            case GENERICInst::Neg:
                return _instinfoNeg;
            case GENERICInst::Abs:
                return _instinfoAbs;
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
            case GENERICInst::Load:
                return _instinfoLoad;
            case GENERICInst::Store:
                return _instinfoStore;
            case GENERICInst::Jump:
                return _instinfoJump;
            case GENERICInst::Branch:
                return _instinfoBranch;
            case GENERICInst::Copy:
                return _instinfoCopy;
            case GENERICInst::LoadGlobalAddress:
                return _instinfoLoadGlobalAddress;
            case GENERICInst::LoadImm:
                return _instinfoLoadImm;
            case GENERICInst::LoadStackObjAddr:
                return _instinfoLoadStackObjAddr;
            default:
                assert(false && "Invalid opcode");
                // return TargetInstInfo::get_instinfo(opcode);
        }
    }
};

TargetInstInfo& getGENERICInstInfo() {
    static GENERICInstInfo instance;
    return instance;
}
GENERIC_NAMESPACE_END