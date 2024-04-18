// clang-format off
#pragma once
#include "mir/mir.hpp"
#include "mir/datalayout.hpp"
#include "mir/iselinfo.hpp"
#include "mir/instinfo.hpp"
#include "mir/frameinfo.hpp"
#include "mir/registerinfo.hpp"
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
    virtual DataLayout& get_datalayout() = 0;
    // virtual  TargetScheduleModel& get_schedule_model()  = 0;
    virtual TargetInstInfo& get_target_inst_info() = 0;
    virtual TargetISelInfo& get_target_isel_info() = 0;
    virtual TargetFrameInfo& get_target_frame_info() = 0;
    virtual TargetRegisterInfo& get_register_info() = 0;

    virtual void emit_assembly(std::ostream& out, MIRModule& module) = 0;
};

struct MIRFlags final {
    bool endsWithTerminator = true;
    bool inSSAForm = true;
    bool preRA = true;
    bool postSA = false;
    bool dontForward = false;
    bool postLegal = false;
};

struct CodeGenContext final {
    Target& target;
    //  TargetScheduleModel& scheduleModel;
    DataLayout& dataLayout;
    TargetInstInfo& instInfo;
    TargetFrameInfo& frameInfo;
    MIRFlags flags;
    TargetISelInfo* iselInfo;
    TargetRegisterInfo* registerInfo;

    uint32_t idx = 0;
    uint32_t next_id() { return ++idx; }
};

// using TargetBuilder = std::pair<std::string_view, std::function<Targe*()> >;
// class TargetRegistry {
//     std::vector<TargetBuilder> _targets;
// public:
//     void add_target( TargetBuilder& target_builder);
//     Target* select_target()
// };

}  // namespace mir
