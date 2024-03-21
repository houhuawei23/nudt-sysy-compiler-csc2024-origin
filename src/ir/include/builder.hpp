/**
 * @file builder.hpp
 *
 */
#pragma once

#include <any>
#include "infrast.hpp"
#include "instructions.hpp"
namespace ir {

/**
 * @brief IR Builder for Module.
 *
 */
class IRBuilder {
   private:
    BasicBlock* _block = nullptr;  // current basic block for insert instruction
    inst_iterator _pos;  // insert pos for cur block
    block_ptr_stack _headers, _exits;
    int _if_cnt, _while_cnt, _rhs_cnt, _func_cnt, _var_cnt;

    // true/false br targets stack for if-else Short-circuit evaluation
    // the top the deeper nest
    block_ptr_stack _true_targets, _false_targets;
    int _bb_cnt;

   public:
    IRBuilder() {
        _if_cnt = 0;
        _while_cnt = 0;
        _rhs_cnt = 0;
        _func_cnt = 0;
        _var_cnt = 0;
        _bb_cnt = 0;
    }

    //! get
    // int bb_cnt() {
    //     return _bb_cnt++;
    // }
    std::string get_bbname() { return "bb" + std::to_string(_bb_cnt++); }
    BasicBlock* block() const { return _block; }
    inst_iterator position() const { return _pos; }

    BasicBlock* header() {
        if (not _headers.empty())
            return _headers.top();
        else
            return nullptr;
    }
    BasicBlock* exit() {
        if (not _exits.empty())
            return _exits.top();
        else
            return nullptr;
    }

    //! manage attributes
    void set_pos(BasicBlock* block, inst_iterator pos) {
        _block = block;
        _pos = pos;  // _pos 与 ->end() 绑定?
    }

    void push_header(BasicBlock* block) { _headers.push(block); }
    void push_exit(BasicBlock* block) { _exits.push(block); }
    void push_loop(BasicBlock* header_block, BasicBlock* exit_block) {
        push_header(header_block);
        push_exit(exit_block);
    }

    void pop_loop() {
        _headers.pop();
        _exits.pop();
    }

    void if_inc() { _if_cnt++; }
    void while_inc() { _while_cnt++; }
    void rhs_inc() { _rhs_cnt++; }
    void func_inc() { _func_cnt++; }

    int if_cnt() const { return _if_cnt; }
    int while_cnt() const { return _while_cnt; }
    int rhs_cnt() const { return _rhs_cnt; }
    int func_cnt() const { return _func_cnt; }

    void push_true_target(BasicBlock* block) { _true_targets.push(block); }
    void push_false_target(BasicBlock* block) { _false_targets.push(block); }
    void push_tf(BasicBlock* true_block, BasicBlock* false_block) {
        _true_targets.push(true_block);
        _false_targets.push(false_block);
    }

    // current stmt or exp 's true/false_target
    BasicBlock* true_target() { return _true_targets.top(); }
    BasicBlock* false_target() { return _false_targets.top(); }

    void pop_tf() {
        _true_targets.pop();
        _false_targets.pop();
    }

    //! create
    AllocaInst* create_alloca(Type* ret_type,
                              const_value_ptr_vector& dims={},
                              const_str_ref name="", 
                              bool is_const=false) {
        auto inst = new AllocaInst(ret_type, _block, dims, name, is_const);
        block()->emplace_back_inst(inst);
        return inst;
    }

    StoreInst* create_store(Value* value,
                            Value* pointer,
                            const_str_ref name = "") {
        auto inst = new StoreInst(value, pointer, _block, name);
        block()->emplace_back_inst(inst);
        return inst;
    }

    ReturnInst* create_return(Value* value = nullptr, const_str_ref name = "") {
        auto inst = new ReturnInst(value, _block);
        block()->emplace_back_inst(inst);
        return inst;
    }

    LoadInst* create_load(Value* ptr,
                          const_value_ptr_vector& indices = {},
                          const_str_ref name = "") {
        auto inst = LoadInst::gen(ptr, _block, indices, name);
        block()->emplace_back_inst(inst);
        return inst;
    }

    UnaryInst* create_unary(Value::ValueId kind,
                            Type* type,
                            Value* value,
                            const_str_ref name = "") {
        auto inst = new UnaryInst(kind, type, value, _block, name);
        block()->emplace_back_inst(inst);
        return inst;
    }
    UnaryInst* create_sitof(Type* type, Value* value, const_str_ref name = "") {
        return create_unary(Value::ValueId::vITOF, type, value, name);
    }
    UnaryInst* create_ftosi(Type* type, Value* value, const_str_ref name = "") {
        return create_unary(Value::ValueId::vFTOI, type, value, name);
    }
    UnaryInst* create_fneg(Type* type, Value* value, const_str_ref name = "") {
        return create_unary(Value::ValueId::vFNEG, type, value, name);
    }

    BinaryInst* create_binary(Value::ValueId kind,
                              Type* type,
                              Value* lvalue,
                              Value* rvalue,
                              const_str_ref name = "") {
        auto inst = new BinaryInst(kind, type, lvalue, rvalue, _block, name);
        block()->emplace_back_inst(inst);
        return inst;
    }
    BinaryInst* create_add(Type* type,
                           Value* lvalue,
                           Value* rvalue,
                           const_str_ref name = "") {
        return type->is_i32() ? create_binary(Value::ValueId::vADD, type,
                                              lvalue, rvalue, name)
                              : create_binary(Value::ValueId::vFADD, type,
                                              lvalue, rvalue, name);
    }
    BinaryInst* create_sub(Type* type,
                           Value* lvalue,
                           Value* rvalue,
                           const_str_ref name = "") {
        return type->is_i32() ? create_binary(Value::ValueId::vSUB, type,
                                              lvalue, rvalue, name)
                              : create_binary(Value::ValueId::vFSUB, type,
                                              lvalue, rvalue, name);
    }
    BinaryInst* create_mul(Type* type,
                           Value* lvalue,
                           Value* rvalue,
                           const_str_ref name = "") {
        return type->is_i32() ? create_binary(Value::ValueId::vMUL, type,
                                              lvalue, rvalue, name)
                              : create_binary(Value::ValueId::vFMUL, type,
                                              lvalue, rvalue, name);
    }
    BinaryInst* create_div(Type* type,
                           Value* lvalue,
                           Value* rvalue,
                           const_str_ref name = "") {
        return type->is_i32() ? create_binary(Value::ValueId::vSDIV, type,
                                              lvalue, rvalue, name)
                              : create_binary(Value::ValueId::vFDIV, type,
                                              lvalue, rvalue, name);
    }
    BinaryInst* create_rem(Type* type,
                           Value* lvalue,
                           Value* rvalue,
                           const_str_ref name = "") {
        return type->is_i32() ? create_binary(Value::ValueId::vSREM, type,
                                              lvalue, rvalue, name)
                              : create_binary(Value::ValueId::vFREM, type,
                                              lvalue, rvalue, name);
    }

    CallInst* create_call(Function* func,
                          const_value_ptr_vector& args,
                          const_str_ref name = "") {
        //! TODO
        auto call = new CallInst(func, args, _block, name);
        block()->emplace_back_inst(call);
        return call;
    }
    BranchInst* create_br(Value* cond,
                          BasicBlock* true_block,
                          BasicBlock* false_block) {
        //! TODO
        // assert(false && "not implemented");
        auto inst = new BranchInst(cond, true_block, false_block, _block);
        // _block->insts().emplace(_pos, inst);
        block()->emplace_back_inst(inst);  // _pos++
        return inst;
    }

    BranchInst* create_br(BasicBlock* dest) {
        //! TODO
        // assert(false && "not implemented");
        auto inst = new BranchInst(dest, _block);
        // _block->insts().emplace(_pos, inst);
        block()->emplace_back_inst(inst);  // _pos++
        return inst;
    }
    //! ICMP inst family
    // (itype, lhs, rhs, parent, name)
    ICmpInst* create_icmp(Value::ValueId itype,
                          Value* lhs,
                          Value* rhs,
                          const_str_ref name = "") {
        //! TODO
        // assert(false && "not implemented");
        auto inst = new ICmpInst(itype, lhs, rhs, _block, name);
        // _block->insts().emplace(_pos, inst);
        block()->emplace_back_inst(inst);  // _pos++
        return inst;
    }
    ICmpInst* create_ieq(Value* lhs, Value* rhs, const_str_ref name = ""

    ) {
        //! TODO
        // assert(false && "not implemented");
        return create_icmp(Value::vIEQ, lhs, rhs, name);
    }
    // icmp ne i32 4, 5
    ICmpInst* create_ine(Value* lhs, Value* rhs, const_str_ref name = "") {
        //! TODO
        // assert(false && "not implemented");
        return create_icmp(Value::vINE, lhs, rhs, name);
    }
    ICmpInst* create_isgt(Value* lhs, Value* rhs, const_str_ref name = "") {
        //! TODO
        // assert(false && "not implemented");
        return create_icmp(Value::vISGT, lhs, rhs, name);
    }
    ICmpInst* create_isge(Value* lhs, Value* rhs, const_str_ref name = "") {
        //! TODO
        // assert(false && "not implemented");
        return create_icmp(Value::vISGE, lhs, rhs, name);
    }
    ICmpInst* create_islt(Value* lhs, Value* rhs, const_str_ref name = "") {
        //! TODO
        // assert(false && "not implemented");
        return create_icmp(Value::vISLT, lhs, rhs, name);
    }
    ICmpInst* create_isle(Value* lhs, Value* rhs, const_str_ref name = "") {
        //! TODO
        // assert(false && "not implemented");
        return create_icmp(Value::vISLE, lhs, rhs, name);
    }
    //! FCMP inst family
    FCmpInst* create_fcmp(Value::ValueId itype,
                          Value* lhs,
                          Value* rhs,
                          const_str_ref name = "") {
        //! TODO: base fcmp
        // assert(false && "not implemented");
        auto inst = new FCmpInst(itype, lhs, rhs, _block, name);
        // _block->insts().emplace(_pos, inst);
        block()->emplace_back_inst(inst);  // _pos++
        return inst;
    }
    //! <result> = fcmp oeq float 4.0, 5.0
    //! yields: result=false
    FCmpInst* create_foeq(Value* lhs, Value* rhs, const_str_ref name = "") {
        //! TODO
        // assert(false && "not implemented");
        return create_fcmp(Value::vFOEQ, lhs, rhs, name);
    }
    // <result> = fcmp one float 4.0, 5.0
    // yields: result=true
    FCmpInst* create_fone(Value* lhs, Value* rhs, const_str_ref name = "") {
        //! TODO
        // assert(false && "not implemented");
        return create_fcmp(Value::vFONE, lhs, rhs, name);
    }
    FCmpInst* create_fogt(Value* lhs, Value* rhs, const_str_ref name = "") {
        //! TODO
        // assert(false && "not implemented");
        return create_fcmp(Value::vFOGT, lhs, rhs, name);
    }
    FCmpInst* create_foge(Value* lhs, Value* rhs, const_str_ref name = "") {
        //! TODO
        // assert(false && "not implemented");
        return create_fcmp(Value::vFOGE, lhs, rhs, name);
    }
    FCmpInst* create_folt(Value* lhs, Value* rhs, const_str_ref name = "") {
        //! TODO
        // assert(false && "not implemented");
        return create_fcmp(Value::vFOLT, lhs, rhs, name);
    }
    FCmpInst* create_fole(Value* lhs, Value* rhs, const_str_ref name = "") {
        //! TODO
        // assert(false && "not implemented");
        return create_fcmp(Value::vFOLE, lhs, rhs, name);
    }

    //! Create GetElementPtr Instruction
    GetElementPtrInst* create_getelementptr(Type* base_type,
                                            Value* value,
                                            Value* idx,
                                            int current_dimension = 1,
                                            const_value_ptr_vector& dims = {},
                                            const_str_ref name = "",
                                            int id = 1) {
        auto inst = new GetElementPtrInst(base_type, value, _block, idx, dims,
                                          current_dimension, name, id);
        block()->emplace_back_inst(inst);
        return inst;
    }

    void reset() {
        _var_cnt = 0;
        _bb_cnt = 0;
    }

    std::string getvarname() {
        // temporary realization
        std::string res = std::to_string(_var_cnt);
        _var_cnt++;

        return "%" + res;
        // TODO!
        // all counting of local variables should be with funcScope
    }
};

}  // namespace ir
