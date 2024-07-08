#pragma once
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/iselinfo.hpp"

namespace mir {
/* LoweringContext */
class LoweringContext {
public:
    Target& _target;
    MIRModule& _mir_module;
    MIRFunction* _mir_func = nullptr;
    MIRBlock* _mir_block = nullptr;

    std::unordered_map<ir::Value*, MIROperand*> _val_map;                /* local variable mappings */
    std::unordered_map<ir::Function*, MIRFunction*> func_map;            /* global */
    std::unordered_map<ir::GlobalVariable*, MIRGlobalObject*> gvar_map;  /* global */
    std::unordered_map<ir::BasicBlock*, MIRBlock*> _block_map;           /* local in function */

    /* Pointer Type for Target Platform */
    OperandType _ptr_type = OperandType::Int64;

    /* Code Generate Context */
    CodeGenContext* _code_gen_ctx = nullptr;
    MIRFunction* memsetFunc;
public:
    LoweringContext(MIRModule& mir_module, Target& target) : _mir_module(mir_module), _target(target) {
        // memsetFunc = new MIRFunction("memset", &_mir_module);
        _mir_module.functions().push_back(std::make_unique<MIRFunction>("_memset", &mir_module));
        memsetFunc = _mir_module.functions().back().get();
    }
public:  // set function
    void set_code_gen_ctx(CodeGenContext* code_gen_ctx) { _code_gen_ctx = code_gen_ctx; }
    void set_mir_func(MIRFunction* mir_func) { _mir_func = mir_func; }
    void set_mir_block(MIRBlock* mir_block) { _mir_block = mir_block; }
public:  // get function
    MIRModule& get_mir_module() { return _mir_module; };
    MIRBlock* get_mir_block() const { return _mir_block; }
    OperandType get_ptr_type() { return _ptr_type; }
public:  // gen function
    MIROperand* new_vreg(ir::Type* type) {
        auto optype = get_optype(type);
        return MIROperand::as_vreg(_code_gen_ctx->next_id(), optype);
    }
    MIROperand* new_vreg(OperandType type) {
        return MIROperand::as_vreg(_code_gen_ctx->next_id(), type);
    }
public:  // emit function
    void emit_inst(MIRInst* inst) { _mir_block->insts().emplace_back(inst); }
    void emit_copy(MIROperand* dst, MIROperand* src) {
        auto inst = new MIRInst(select_copy_opcode(dst, src));
        inst->set_operand(0, dst);
        inst->set_operand(1, src);
        emit_inst(inst);
    }
public:  // ir_val -> mir_operand
    void add_valmap(ir::Value* ir_val, MIROperand* mir_operand) {
        if (_val_map.count(ir_val)) assert(false && "value already mapped");
        _val_map.emplace(ir_val, mir_operand);
    }
    MIROperand* map2operand(ir::Value* ir_val) {
        /* 1. Local Value */
        auto iter = _val_map.find(ir_val);
        if (iter != _val_map.end()) return iter->second;

        /* 2. Global Value */
        if (auto gvar = dyn_cast<ir::GlobalVariable>(ir_val)) {
            auto ptr = new_vreg(_ptr_type);
            auto inst = new MIRInst(InstLoadGlobalAddress);
            auto g_reloc = gvar_map.at(gvar)->reloc.get();  // MIRRelocable*
            auto operand = MIROperand::as_reloc(g_reloc);
            inst->set_operand(0, ptr); inst->set_operand(1, operand);
            emit_inst(inst);
            return ptr;
        }

        /* 3. Constant */
        if (not ir::isa<ir::Constant>(ir_val)) {
            std::cerr << "error: " << ir_val->name() << " must be constant" << std::endl;
            assert(false);
        }

        auto const_val = dyn_cast<ir::Constant>(ir_val);
        if (ir_val->type()->isInt()) {
            auto imm = MIROperand::as_imm(const_val->i32(), OperandType::Int32);
            return imm;
        }

        assert(false);
        return nullptr;
    }
    MIRBlock* map2block(ir::BasicBlock* ir_block) { return _block_map.at(ir_block); }
public:  // utils function
    static OperandType get_optype(ir::Type* type) {
        if (type->isInt()) {
            switch (type->btype()) {
                case ir::INT1: return OperandType::Bool;
                case ir::INT32: return OperandType::Int32;
                default: assert(false && "unsupported int type");
            }
        } else if (type->isFloatPoint()) {
            switch (type->btype()) {
                case ir::FLOAT: return OperandType::Float32;
                default: assert(false && "unsupported float type");
            }
        } else if (type->isPointer()) {
            return OperandType::Int64;
        }
        return OperandType::Special;
    }
};
std::unique_ptr<MIRModule> create_mir_module(ir::Module& ir_module, Target& target);
}  // namespace mir