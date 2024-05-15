#include "autogen/riscv/InstInfoDecl.hpp"

RISCV_NAMESPACE_BEGIN

enum RISCVPipeline : uint32_t { RISCVIDivPipeline, RISCVFPDivPipeline };
enum RISCVIssueMask : uint32_t {
    RISCVPipelineA = 1 << 0,
    RISCVPipelineB = 1 << 1,
    RISCVPipelineAB = RISCVPipelineA | RISCVPipelineB
};

template <uint32_t ValidPipeline, bool Early, bool Late>
class RISCVScheduleClassIntegerArithmeticGeneric final : public ScheduleClass {
    static_assert(ValidPipeline != 0 && (Early || Late));

public:
    bool schedule(ScheduleState& state,
                  const MIRInst& inst,
                  const InstInfo& instInfo) const override {
        return false;
    }
};

using RISCVScheduleClassIntegerArithmetic =
    RISCVScheduleClassIntegerArithmeticGeneric<RISCVPipelineAB, true, true>;
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

class RISCVScheduleClassLoadStore final : public ScheduleClass {
public:
    bool schedule(ScheduleState& state,
                  const MIRInst& inst,
                  const InstInfo& instInfo) const override {
        return false;
    }
};

class RISCVScheduleClassMulti final : public ScheduleClass {
public:
    bool schedule(ScheduleState& state,
                  const MIRInst& inst,
                  const InstInfo& instInfo) const override {
        return false;
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