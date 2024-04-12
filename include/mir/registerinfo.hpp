#pragma once
#include "mir/mir.hpp"
namespace mir
{

class TargetRegisterInfo {
public:
    virtual ~TargetRegisterInfo() = default;
    virtual uint32_t get_alloca_class_cnt(); // ?
    virtual uint32_t get_alloca_class(OperandType type);
    virtual bool is_legal_isa_reg_operand(MIROperand& op);
    // ..
    virtual bool is_zero_reg();

};
} // namespace mir
