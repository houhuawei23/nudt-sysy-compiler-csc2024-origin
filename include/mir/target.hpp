// clang-format off
#pragma once
#include "mir/mir.hpp"
#include "mir/datalayout.hpp"
#include "mir/instinfo.hpp"
#include "mir/iselinfo.hpp"
#include "mir/frameinfo.hpp"
// clang-format on
namespace mir {
class Target {
    // getDataLayout
    // getScheduleModel
    // getISelInfo
    // getFrameInfo
    // getRegisterInfo
    // emitAssembly
   public:
    virtual ~Target() = default;
    virtual const DataLayout& get_datalayout() const = 0;
    // virtual const TargetScheduleModel& get_schedule_model() const = 0;
    virtual const TargetInstInfo& get_inst_info() const = 0;
    virtual const TargetISelInfo& get_isel_info() const = 0;
    // virtual const TargetFrameInfo& get_frame_info() const = 0;
    // virtual const TargetRegisterInfo* get_register_info() const = 0;
};

struct MIRFlags final {
    bool endsWithTerminator = true;
    bool inSSAForm = true;
    bool preRA = true;
    bool postSA = false;
    bool dontForward = false;
    bool postLegal = false;
};

class CodeGenContext final {
    const Target& target;
    // const TargetScheduleModel& scheduleModel;
    const DataLayout& dataLayout;
    const TargetInstInfo& instInfo;
    const TargetISelInfo& iselInfo;
    const TargetFrameInfo& frameInfo;
    // const TargetRegisterInfo* registerInfo;
    MIRFlags flags;
    uint32_t idx = 0;
    uint32_t next_id() { return ++idx; }
};

class RISCVTarget : public Target {
    // RISCVDataLayout mDataLayout;
    // RISCVFrameInfo mFrameInfo;
    // RISCVRegisterInfo mRegisterInfo;
};
}  // namespace mir
