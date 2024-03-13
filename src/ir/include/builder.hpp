/**
 * @file builder.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-03-03
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <stack>
#include "infrast.hpp"
#include "instructions.hpp"

namespace ir {

/**
 * @brief IR Builder for Module.
 * @class IRBuilder
 *
 */
class IRBuilder {
    using block_stack = std::stack<BasicBlock*>;
    using const_vector_Value_ptr = const std::vector<Value*>;
    using const_str_ref = const std::string&;

   private:
    BasicBlock* _block;
    inst_iterator _position;
    block_stack _headers, _exits;
    int _if_cnt, _while_cnt, _rhs_cnt, _func_cnt, _var_cnt;
    block_stack _true_targets, _false_targets;

   public:
    IRBuilder() {
        _if_cnt = 0;
        _while_cnt = 0;
        _rhs_cnt = 0;
        _func_cnt = 0;
        _var_cnt=0;
    }

    //! get
    BasicBlock* block() const { return _block; }
    inst_iterator position() const { return _position; }

    BasicBlock* header() { return _headers.top(); }
    BasicBlock* exit() { return _exits.top(); }

    //! manage attributes
    void set_position(BasicBlock* block, inst_iterator pos) {
        _block = block;
        _position = pos;
    }

    void push_header(BasicBlock* block) { _headers.push(block); }
    void push_exit(BasicBlock* block) { _exits.push(block); }

    void pop_loop() {
        //! Why?
        _headers.pop();
        _exits.pop();
        assert(false && "not understand!");
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

    BasicBlock* true_target() { return _true_targets.top(); }
    BasicBlock* false_target() { return _false_targets.top(); }
    
    //! create
    AllocaInst* create_alloca(Type* ret_type,
                              const_vector_Value_ptr& dims = {},
                              const_str_ref name = "",
                              const bool is_const = false) {
        auto inst = new AllocaInst(ret_type, _block, dims, name, is_const);
        // // assert(inst);
        _block->insts().emplace(_position, inst);
        return inst;
        // return nullptr;
    }

    StoreInst* create_store(Value* value,
                            Value* pointer,
                            const_vector_Value_ptr& dims = {},
                            const_str_ref name = "") {
        auto inst = new StoreInst(value, pointer, _block, dims, name);
        _block->insts().emplace(_position, inst);
        return inst;
    }

    ReturnInst* create_return(Value* value = nullptr,
                              //   const_str_ref name = "",
                              BasicBlock* parent = nullptr) {
        auto inst = new ReturnInst(value, parent);
        _block->insts().emplace(_position, inst);
        return inst;
    }
    LoadInst* create_load(Value* ptr,
                          const_value_ptr_vector& indices = {},
                          const_str& name = "") {
        auto inst = new LoadInst(ptr,_block,indices,name);
        _block->insts().emplace(_position, inst);
        return inst;
    }
    UnaryInst* create_unary() {
        //! TODO
        assert(false && "not implemented");
    }
    BinaryInst* create_binary() {
        //! TODO
        assert(false && "not implemented");
    }
    CallInst* create_call() {
        //! TODO
        assert(false && "not implemented");
    }
    BranchInst* create_branch() {
        //! TODO
        assert(false && "not implemented");
    }

    std::string getvarname(){
        // temporary realization
        _var_cnt++;
        std::string res=std::to_string(_var_cnt);
        
        return "%"+res;
        //TODO!
        //all counting of local variables should be with funcScope
    }
};

}  // namespace ir
