#include "autogen/riscv/InstInfoDecl.hpp"

RISCV_NAMESPACE_BEGIN

enum RISCVPipeline : uint32_t { RISCVIDivPipeline, RISCVFPDivPipeline };
enum RISCVIssueMask : uint32_t {
    RISCVPipelineA = 1 << 0,                           // 0b01
    RISCVPipelineB = 1 << 1,                           // 0b10
    RISCVPipelineAB = RISCVPipelineA | RISCVPipelineB  // 0b11
};

template <uint32_t ValidPipeline, bool Early, bool Late>
class RISCVScheduleClassIntegerArithmeticGeneric final : public ScheduleClass {
    static_assert(ValidPipeline != 0 && (Early || Late));

public:
    bool schedule(ScheduleState& state,
                  const MIRInst& inst,
                  const InstInfo& instInfo) const override {
        if (not state.isAvailable(ValidPipeline))
            return false;
        // Address Genration Unit (AG)
        bool availableInAG = true;

        /*
        if operand latency is 0:
           availableInAG = availableInAG & true;
        else if operand latency is 1:
           availableInAG = false;
        else latency > 2:
           return false;
        */
        for (uint32_t idx = 0; idx < instInfo.operand_num(); idx++) {
            if (instInfo.operand_flag(idx) & OperandFlagUse) {
                const auto latency = state.queryRegisterLatency(inst, idx);
                switch (latency) {
                    case 0:
                        break;
                    case 1:
                        availableInAG = false;
                        break;
                    default:
                        return false;
                        break;
                }
            }
        }
        // all operands laytency is 0 or 1, issue instruction

        if constexpr (Early) {
            if (availableInAG) {
                // all operands are ready (latency == 0)
                if constexpr (ValidPipeline == RISCVPipelineAB) {
                    auto availablePipeline = state.isAvailable(RISCVPipelineA)
                                                 ? RISCVPipelineA
                                                 : RISCVPipelineB;
                    state.setIssued(availablePipeline);
                } else {
                    state.setIssued(ValidPipeline);
                }
                // dst operand reg will be ready after next cycle
                state.makeRegisterReady(inst, 0, 1);
                return true;
            }
        }
        if constexpr (Late) {
            if constexpr (ValidPipeline == RISCVPipelineAB) {
                auto availablePipeline = state.isAvailable(RISCVPipelineA)
                                             ? RISCVPipelineA
                                             : RISCVPipelineB;
                state.setIssued(availablePipeline);
            } else {
                state.setIssued(ValidPipeline);
            }
            // dst operand reg will be ready after 3 cycles
            state.makeRegisterReady(inst, 0, 3);
            return true;
        }

        return false;
    }
};
/* Normal Integer Arithmetic, can issued in A/B, early and late scheduling */
using RISCVScheduleClassIntegerArithmetic =
    RISCVScheduleClassIntegerArithmeticGeneric<RISCVPipelineAB, true, true>;
/* LateB Integer Arithmetic, can issued in B, late scheduling */
using RISCVScheduleClassIntegerArithmeticLateB =
    RISCVScheduleClassIntegerArithmeticGeneric<RISCVPipelineB, false, true>;
using RISCVScheduleClassIntegerArithmeticEarlyB =
    RISCVScheduleClassIntegerArithmeticGeneric<RISCVPipelineB, true, false>;
using RISCVScheduleClassIntegerArithmeticLateAB =
    RISCVScheduleClassIntegerArithmeticGeneric<RISCVPipelineAB, false, true>;
using RISCVScheduleClassIntegerArithmeticEarlyLateB =
    RISCVScheduleClassIntegerArithmeticGeneric<RISCVPipelineB, true, true>;

class RISCVScheduleClassSlowLoadImm final : public ScheduleClass {
public:
    bool schedule(ScheduleState& state,
                  const MIRInst& inst,
                  const InstInfo& instInfo) const override {
        return false;
    }
};

class RISCVScheduleClassBranch final : public ScheduleClass {
public:
    bool schedule(ScheduleState& state,
                  const MIRInst& inst,
                  const InstInfo& instInfo) const override {
        if (not state.isAvailable(RISCVPipelineB))
            return false;
        for (uint32_t idx = 0; idx < instInfo.operand_num(); ++idx) {
            if (instInfo.operand_flag(idx) & OperandFlagUse) {
                if (state.queryRegisterLatency(inst, idx) > 2)
                    return false;
            }
        }

        state.setIssued(RISCVPipelineB);
        return true;
    }
};
class RISCVScheduleClassLoadStore final : public ScheduleClass {
public:
    bool schedule(ScheduleState& state,
                  const MIRInst& inst,
                  const InstInfo& instInfo) const override {
        if (not state.isAvailable(RISCVPipelineA))
            return false;
        // require effective address ready in AG stage
        for (uint32_t idx = 0; idx < instInfo.operand_num(); ++idx) {
            if (instInfo.operand_flag(idx) & OperandFlagUse) {
                if (state.queryRegisterLatency(inst, idx) > 0)
                    return false;
            }
        }
        /* def operand reg will be ready after 3 cycles */
        if (instInfo.operand_flag(0) & OperandFlagDef) {
            state.makeRegisterReady(inst, 0, 3);
        }
        state.setIssued(RISCVPipelineA);
        return true;
    }
};

class RISCVScheduleClassMulti final : public ScheduleClass {
public:
    bool schedule(ScheduleState& state,
                  const MIRInst& inst,
                  const InstInfo& instInfo) const override {
        if (not state.isAvailable(RISCVPipelineB))
            return false;

        for (uint32_t idx = 0; idx < instInfo.operand_num(); ++idx) {
            if (instInfo.operand_flag(idx) & OperandFlagUse) {
                if (state.queryRegisterLatency(inst, idx) > 0)
                    return false;
            }
        }
        /* def operand reg will be ready after 3 cycles */
        if (instInfo.operand_flag(0) & OperandFlagDef) {
            state.makeRegisterReady(inst, 0, 3);
        }
        state.setIssued(RISCVPipelineB);
        return true;
    }
};

class RISCVScheduleClassDivRem final : public ScheduleClass {
public:
    bool schedule(ScheduleState& state,
                  const MIRInst& inst,
                  const InstInfo& instInfo) const override {
        return false;
    }
};
class RISCVScheduleClassSDivRemW final : public ScheduleClass {
public:
    bool schedule(ScheduleState& state,
                  const MIRInst& inst,
                  const InstInfo& instInfo) const override {
        return false;
    }
};

RISCV_NAMESPACE_END