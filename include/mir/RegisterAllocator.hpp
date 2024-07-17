#pragma once
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include "mir/mir.hpp"
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
};