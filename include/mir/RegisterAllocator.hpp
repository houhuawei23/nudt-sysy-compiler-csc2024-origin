#pragma once
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include "mir/MIR.hpp"
#include "mir/target.hpp"
#include "mir/LiveInterval.hpp"

namespace mir {
/*
 * @brief: IPRAInfo Class
 * @details:
 *      存储每个函数中所使用到的Caller Saved寄存器
 */
using IPRAInfo = std::unordered_set<RegNum>;
class Target;
class IPRAUsageCache final {
  std::unordered_map<std::string, IPRAInfo> _cache;

public:  // utils function
  void add(const CodeGenContext& ctx, MIRFunction& mfunc);
  void add(std::string symbol, IPRAInfo info);
  const IPRAInfo* query(std::string calleeFunc) const;

public:  // Just for Debug
  void dump(std::ostream& out, std::string calleeFunc) const;
};

void fastAllocator(MIRFunction& mfunc, CodeGenContext& ctx, IPRAUsageCache& infoIPRA);
void fastAllocatorBeta(MIRFunction& mfunc, CodeGenContext& ctx, IPRAUsageCache& infoIPRA);

void graphColoringAllocateBeta(MIRFunction& mfunc, CodeGenContext& ctx, IPRAUsageCache& infoIPRA);
void GraphColoringAllocate(MIRFunction& mfunc, CodeGenContext& ctx, IPRAUsageCache& infoIPRA);

void linearAllocator(MIRFunction& mfunc, CodeGenContext& ctx);

void mixedRegisterAllocate(MIRFunction& mfunc, CodeGenContext& ctx, IPRAUsageCache& infoIPRA);

using RegWeightMap = std::unordered_map<RegNum, double>;
struct RegNumComparator final {
  const RegWeightMap* weights;

  bool operator()(RegNum lhs, RegNum rhs) const { return weights->at(lhs) > weights->at(rhs); }
};

/*
 * @brief: Interference Graph (干涉图)
 * @note:
 *      成员变量:
 *          1. std::unordered_map<RegNum, std::unordered_set<RegNum>> _adj
 *               干涉图 (存储每个节点的邻近节点)
 *          2. std::unordered_map<RegNum, uint32_t> _degree
 *               存储干涉图中虚拟寄存器节点的度数
 *          3. Queue _queue
 *               优先队列, 存储可分配物理寄存器的虚拟寄存器节点
 */
class InterferenceGraph final {
  std::unordered_map<RegNum, std::unordered_set<RegNum>> mAdj;
  std::unordered_map<RegNum, uint32_t> mDegree;
  using Queue = std::priority_queue<RegNum, std::vector<RegNum>, RegNumComparator>;
  Queue mQueue;

public: /* Get Function */
  auto& adj(RegNum u) {
    assert(isVirtualReg(u));
    return mAdj[u];
  }

public: /* About Degree Function */
  void create(RegNum u) {
    if (!mDegree.count(u)) mDegree[u] = 0U;
  }
  auto empty() const { return mDegree.empty(); }
  auto size() const { return mDegree.size(); }

public: /* Util Function */
  /* 功能: 加边 */
  void add_edge(RegNum lhs, RegNum rhs);
  /* 功能: 为图着色寄存器分配做准备 */
  void prepare_for_assign(const RegWeightMap& weights, uint32_t k);
  /* 功能: 选择虚拟寄存器来为其分配物理寄存器 */
  RegNum pick_to_assign(uint32_t k);
  /* 功能: 选择虚拟寄存器来将其spill到栈内存中 */
  RegNum pick_to_spill(const std::unordered_set<RegNum>& blockList,
                       const RegWeightMap& weights,
                       uint32_t k) const;
  /* 功能: 统计干涉图中虚拟寄存器的个数 */
  std::vector<RegNum> collect_nodes() const;

public: /* just for debug */
  void dump(std::ostream& out) const;
};

struct GraphColoringAllocateContext final {
  IPRAUsageCache& infoIPRA;
  std::unordered_map<uint32_t, uint32_t> regMap;
  uint32_t allocationClass;

  std::unordered_map<RegNum, MIROperand> inStackArguments;  // reg to in stack argument inst
  std::unordered_map<RegNum, MIRInst*> constants;           // reg to constant inst

  bool fixHazard;
  size_t regCount;
  std::unordered_set<uint32_t> allocableISARegs;
  std::unordered_set<uint32_t> blockList;

  bool collectInStackArgumentsRegisters(MIRFunction& mfunc, CodeGenContext& ctx);
  bool collectConstantsRegisters(MIRFunction& mfunc, CodeGenContext& ctx);

  std::unordered_set<RegNum> collectVirtualRegs(MIRFunction& mfunc, CodeGenContext& ctx);

  bool isAllocatableType(OperandType type, CodeGenContext& ctx) {
    return (type <= OperandType::Float32) &&
           (ctx.registerInfo->getAllocationClass(type) == allocationClass);
  }
  bool isLockedOrUnderRenamedType(OperandType type) { return (type <= OperandType::Float32); };

  std::unordered_map<RegNum, std::set<InstNum>> defUseTime;
  void colorDefUse(RegNum src, RegNum dst) {
    assert(isVirtualReg(src) && isISAReg(dst));
    if (!fixHazard || !defUseTime.count(src)) return;
    auto& dstInfo = defUseTime[dst];
    auto& srcInfo = defUseTime[src];
    dstInfo.insert(srcInfo.begin(), srcInfo.end());
  };

  std::unordered_map<RegNum, RegWeightMap> copyHint;
  void updateCopyHint(RegNum dst, RegNum src, double weight) {
    if (isVirtualReg(dst)) { copyHint[dst][src] += weight; }
  };

  void buildGraph(MIRFunction& mfunc,
                  CodeGenContext& ctx,
                  LiveVariablesInfo& liveInterval,
                  InterferenceGraph& graph,
                  std::unordered_set<RegNum>& vregSet,
                  class BlockTripCountResult& blockFreq);

  RegWeightMap computeRegWeight(MIRFunction& mfunc,
                                CodeGenContext& ctx,
                                std::vector<uint32_t>& vregs,
                                BlockTripCountResult& blockFreq,
                                LiveVariablesInfo& liveInterval,
                                std::vector<std::pair<InstNum, double>>& freq);

  //
  // std::stack<uint32_t> assignStack;
  bool assignRegisters(MIRFunction& mfunc,
                       CodeGenContext& ctx,
                       InterferenceGraph& graph,
                       RegWeightMap& weights,
                       std::stack<uint32_t>& assignStack);
  //
  bool allocateRegisters(MIRFunction& mfunc,
                         CodeGenContext& ctx,
                         std::vector<uint32_t>& vregs,
                         std::stack<uint32_t>& assignStack,
                         InterferenceGraph& graph,
                         std::vector<std::pair<InstNum, double>>& freq);
};
};  // namespace mir