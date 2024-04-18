#pragma once
#include "mir/mir.hpp"
namespace mir
{

class TargetRegisterInfo {
public:
    virtual ~TargetRegisterInfo() = default;
    virtual uint32_t get_alloca_class_cnt() = 0; // ?
    virtual uint32_t get_alloca_class(OperandType type) = 0;
    virtual bool is_legal_isa_reg_operand(MIROperand& op) = 0;
    // ..
    virtual bool is_zero_reg() = 0;

};
} // namespace mir
