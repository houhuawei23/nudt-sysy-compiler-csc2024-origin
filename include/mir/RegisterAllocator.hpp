#pragma once
#include <unordered_set>
#include <unordered_map>
#include "mir/mir.hpp"

namespace mir {
using IPRAInfo = std::unordered_set<MIROperand*, MIROperandHasher>;
class Target;

class IPRAUsageCache final {
    std::unordered_map<MIRRelocable*, IPRAInfo> _cache;

public:
    void add(const CodeGenContext& ctx, MIRRelocable* symbol, MIRFunction& mfunc);
    void add(MIRRelocable* symbol, IPRAInfo info);
    const IPRAInfo* query(MIRRelocable* calleeFunc) const;
};
};