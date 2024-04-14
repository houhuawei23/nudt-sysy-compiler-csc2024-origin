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
    virtual DataLayout& get_datalayout() = 0;
    // virtual  TargetScheduleModel& get_schedule_model()  = 0;
    virtual TargetInstInfo& get_inst_info() = 0;
    // virtual  TargetISelInfo& get_isel_info()  = 0;
    virtual TargetFrameInfo& get_frame_info() = 0;
    // virtual  TargetRegisterInfo* get_register_info()  = 0;
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
    //  TargetISelInfo& iselInfo;
    TargetFrameInfo& frameInfo;
    //  TargetRegisterInfo* registerInfo;
    MIRFlags flags;
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
