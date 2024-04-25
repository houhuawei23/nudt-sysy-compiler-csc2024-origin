#pragma once
#include "mir/mir.hpp"
namespace mir
{
/*
 * @brief: TargetRegisterInfo Class (抽象基类)
 */
class TargetRegisterInfo {
    public:
        virtual ~TargetRegisterInfo() = default;

    public:  // get function
        virtual uint32_t get_alloca_class_cnt() = 0;
        virtual uint32_t get_alloca_class(OperandType type) = 0;
        virtual MIROperand* get_return_address_register() = 0;
        virtual MIROperand* get_stack_pointer_register() = 0;
        
    public:  // check function    
        virtual bool is_legal_isa_reg_operand(MIROperand& op) = 0;
        virtual bool is_zero_reg(const uint32_t x) const = 0;
    virtual OperandType getCanonicalizedRegisterType(OperandType type) = 0;
};
} // namespace mir
