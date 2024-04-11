#pragma once
#include "mir/mir.hpp"

namespace mir {
class InstInfo {
   public:
    InstInfo() = default;
    virtual ~InstInfo() = default;
    virtual void print(std::ostream& out,
                       const MIRInst& inst,
                       bool printComment) const = 0;
    virtual uint32_t operand_num() const = 0;

};

class TargetInstInfo {
public:
    TargetInstInfo() = default;
    virtual ~TargetInstInfo() = default;
    virtual const InstInfo& get_instinfo(uint32_t opcode) const;

    // virtual bool match_branch(const MIRInst* inst, MIRBlock* target)

};
}  // namespace mir
