#pragma once

#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/iselinfo.hpp"

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

    std::unordered_map<ir::BasicBlock*, MIRBlock*> _block_map;

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

    // void set_mir_module(MIRModule& mir_module) { _mir_module = mir_module; }

   public:  // set function
    void set_mir_func(MIRFunction* mir_func) { _mir_func = mir_func; }
    void set_mir_block(MIRBlock* mir_block) { _mir_block = mir_block; }

   public:  // get function
    MIRModule& get_mir_module() { return _mir_module; };
    MIRBlock* get_mir_block() const { return _mir_block; }
    OperandType get_ptr_type() { return _ptr_type; }

    MIROperand* new_vreg(ir::Type* type) {
        auto optype = get_optype(type);
        return MIROperand::as_vreg(next_id(), optype);
    }
    MIROperand* new_vreg(OperandType type) {
        // auto optype = get_optype(type);
        return MIROperand::as_vreg(next_id(), type);
    }
    void emit_inst(MIRInst* inst) { _mir_block->add_inst(inst); }

    // emit copy generic inst
    void emit_copy(MIROperand* dst, MIROperand* src) {
        uint32_t mir_opcode;
        auto inst = new MIRInst(select_copy_opcode(dst, src));
        inst->set_operand(0, dst);
        inst->set_operand(1, src);
        emit_inst(inst);
    }

    // ir_val -> mir_operand
    void add_valmap(ir::Value* ir_val, MIROperand* mir_operand) {
        if (_val_map.count(ir_val)) {
            assert(false && "value already mapped");
        }
        _val_map.emplace(ir_val, mir_operand);
    }

    MIROperand* map2operand(ir::Value* ir_val) {
        auto iter = _val_map.find(ir_val);
        // ptr create by alloca inst has already been
        // stored as stackobj, can find in _val_map
        // load ret value also has been stored
        if (iter != _val_map.end()) {
            return iter->second;
        }
        // gen the operand
        if (auto gvar = dyn_cast<ir::GlobalVariable>(ir_val)) {
            auto ptr = new_vreg(_ptr_type);
            // must find in gvar_map
            auto inst = new MIRInst(InstLoadGlobalAddress);
            auto g_reloc = gvar_map.at(gvar)->_reloc.get();  // MIRRelocable*
            auto operand = MIROperand::as_reloc(g_reloc);
            inst->set_operand(0, ptr);
            inst->set_operand(1, operand);
            emit_inst(inst);
            return ptr;
        }
        // constant
        // NOTE: constant cant not be cached in _val_map
        if (not ir::isa<ir::Constant>(ir_val)) {
            std::cerr << "error: must be constant" << std::endl;
            assert(false);
        }
        auto const_val = dyn_cast<ir::Constant>(ir_val);
        if (ir_val->type()->is_int()) {
            auto imm = MIROperand::as_imm(const_val->i32(), OperandType::Int32);
            return imm;
        }
        std::cerr << "TODO: map2operand not implemented for type" << std::endl;
        assert(false);
        return nullptr;
    }

    MIRBlock* map2block(ir::BasicBlock* ir_block) {
        auto iter = _block_map.find(ir_block);
        if (iter != _block_map.end()) {
            return iter->second;
        }
        assert(false && "block not found");
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
        assert(false && "unsupported type");
        return OperandType::Special;
    }
};

std::unique_ptr<MIRModule> create_mir_module(ir::Module& ir_module,
                                             Target& target);

}  // namespace mir
