#pragma once
#include <unordered_map>

#include "mir/mir.hpp"
#include "mir/target.hpp"

namespace mir {
using RegAllocFunction = void (*)(MIRFunction& mfunc, CodeGenContext& ctx);
class RegAllocator final {
    std::unordered_map<std::string, RegAllocFunction> _methods;

public:
    void addMethod(std::string name, RegAllocFunction func) { _methods.emplace(name, func); }

public:
    static RegAllocator& get();
};
}  // namespace mir
