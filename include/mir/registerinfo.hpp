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

    public:
        virtual uint32_t get_alloca_class_cnt() = 0;
        virtual uint32_t get_alloca_class(OperandType type) = 0;
        virtual bool is_legal_isa_reg_operand(MIROperand& op) = 0;
        virtual bool is_zero_reg() = 0;
};
} // namespace mir
