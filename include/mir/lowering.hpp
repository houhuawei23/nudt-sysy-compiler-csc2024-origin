#pragma once

#include "mir/mir.hpp"
#include "mir/target.hpp"

/*
lowering context
*/
namespace mir {
class LoweringContext {
   public:
    Target& _target;
    MIRModule& _mir_module;
    MIRFunction* _mir_func = nullptr;
    MIRBlock* _mir_block = nullptr;
    // map
    std::unordered_map<ir::Value*, MIROperand*> _val_map;
    std::unordered_map<ir::Function*, MIRFunction*> func_map;
    std::unordered_map<ir::GlobalVariable*, MIRGlobalObject*> gvar_map;

    uint32_t _idx = 0;
    // pointer type for target platform
    OperandType _ptr_type = OperandType::Int64;

    // code gen context
    CodeGenContext* _code_gen_ctx = nullptr;

   public:
    LoweringContext(MIRModule& mir_module, Target& target)
        : _mir_module(mir_module), _target(target) {}

    void set_code_gen_ctx(CodeGenContext* code_gen_ctx) {
        _code_gen_ctx = code_gen_ctx;
    }

    uint32_t next_id() { return ++_idx; }

    MIRModule* get_mir_module();

    // void set_mir_module(MIRModule& mir_module) { _mir_module = mir_module; }
    void set_mir_func(MIRFunction* mir_func) { _mir_func = mir_func; }
    void set_mir_block(MIRBlock* mir_block) { _mir_block = mir_block; }

    MIROperand* new_vreg(ir::Type* type) {
        auto optype = get_optype(type);
        return MIROperand::as_vreg(next_id(), optype);
    }
    MIROperand* new_vreg(OperandType type) {
        // auto optype = get_optype(type);
        return MIROperand::as_vreg(next_id(), type);
    }
    void emit_inst(MIRInst* inst) { _mir_block->add_inst(inst); }

    // void emit_copy(MIROperand* dst, MIROperand* src) {
    //     uint32_t mir_opcode;
    //     if (dst->is_reg() && isISAReg(dst->reg())) {
    //         // dst is a isa reg
    //         if (src->is_imm()) {
    //             return InstLoadImmToReg;
    //         }
    //         return InstCopyToReg;
    //     }
    // }
    void add_valmap(ir::Value* ir_val, MIROperand* mir_operand) {
        if (_val_map.count(ir_val)) {
            assert(false && "value already mapped");
        }
        _val_map.emplace(ir_val, mir_operand);
    }

    MIROperand* map2operand(ir::Value* ir_val) {
        auto iter = _val_map.find(ir_val);
        if (iter != _val_map.end()) {
            return iter->second;
        }
        // gen the operand
        if (dyn_cast<ir::GlobalVariable>(ir_val)) {
            auto ptr = new_vreg(_ptr_type);
        }
        // constant
    }

    static OperandType get_optype(ir::Type* type) {
        if (type->is_int()) {
            switch (type->btype()) {
                case ir::INT1:
                    return OperandType::Bool;
                case ir::INT32:
                    return OperandType::Int32;
                default:
                    assert(false && "unsupported int type");
            }
        }

        if (type->is_float()) {
        }
    }
};

std::unique_ptr<MIRModule> create_mir_module(ir::Module& ir_module, Target& target);

}  // namespace mir
