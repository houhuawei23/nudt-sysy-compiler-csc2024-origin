#pragma once
#include "mir/mir.hpp"
#include <stdint.h>
#include <unordered_map>
namespace mir {

class ScheduleClass {
   public:
    virtual ~ScheduleClass() = default;
    virtual bool schedule(class ScheduleState& state,
                          MIRInst& inst,
                          class InstInfo& instInfo) = 0;
};
struct MicroArchInfo {
    bool enablePostRAScheduling;
    // Front-end
    bool hasRegRenaming;
    bool hasMacroFusion;
    uint32_t issueWidth;
    // Back-end
    bool outOfOrder;
    // Memory system
    bool hardwarePrefetch;
    uint32_t maxDataStreams;
    uint32_t maxStrideByBytes;
};

class TargetScheduleModel {
   public:
    virtual ~TargetScheduleModel() = default;
    virtual ScheduleClass& getInstScheClass(uint32_t opcode) = 0;
    virtual MicroArchInfo& getMicroArchInfo() = 0;
    virtual bool peepholeOpt(MIRFunction& func, CodeGenContext& context) {
        return false;
    }
    virtual bool isExpensiveInst(MIRInst* inst, CodeGenContext& context) {
        return false;
    }
};

class ScheduleState {
    uint32_t mCycleCount;

    std::unordered_map<uint32_t, uint32_t> mNextPipelineAvailable;

    std::unordered_map<uint32_t, uint32_t> mRegisterAvailableTime;

    // idx -> register
    const std::unordered_map<const MIRInst*, std::unordered_map<uint32_t, uint32_t>>&
        mRegRenameMap;
    uint32_t mIssuedFlag;

   public:
    ScheduleState(
        const std::unordered_map<const MIRInst*, std::unordered_map<uint32_t, uint32_t>>&
            regRenameMap)
        : mCycleCount(0), mRegRenameMap(regRenameMap), mIssuedFlag(0) {}
    // query
    uint32_t queryRegisterLatency(MIRInst& inst, uint32_t idx);
    bool isPipelineReady(uint32_t pipelineId);
    bool isAvailable(uint32_t mask);
    // issue
    void setIssued(uint32_t mask);
    void resetPipeline(uint32_t pipelineId, uint32_t duration);
    void makeRegisterReady(MIRInst& inst, uint32_t idx, uint32_t latency);

    uint32_t nextCycle() {
        mCycleCount++;
        mIssuedFlag = 0;
        return mCycleCount;
    }
};
}  // namespace mir
