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

#include "infrast.hpp"
#include "instructions.hpp"

// #include <utility>
#include <stack>
namespace ir {

/**
 * @brief IR Builder for Module.
 * @class IRBuilder
 *
 */
class IRBuilder {
    using block_stack = std::stack<BasicBlock *>;

  private:
    BasicBlock *_block;
    inst_iterator _position;
    // block_stack headers, exits;
    int if_cnt, while_cnt, rhs_cnt, func_cnt;
    // block_stack truetargets, falsetargets;

  public:
    IRBuilder() {
        if_cnt = 0;
        while_cnt = 0;
        rhs_cnt = 0;
        func_cnt = 0;
    }

    void func_add() { func_cnt++; }

    void set_position(BasicBlock *bb, inst_iterator pos) {
        _block = bb;
        _position = pos;
    }
    int func() { return 44; }
    AllocaInst *create_alloca_inst(Type *ret_type,
                                   const std::vector<Value *> &dims = {},
                                   const std::string &name = "",
                                   const bool is_const = false) {
        auto inst = new AllocaInst(ret_type, _block, dims, name, is_const);
        // // assert(inst);
        _block->get_insts().emplace(_position, inst);
        return inst;
        // return nullptr;
    }

    StoreInst *create_store_inst(Value *value, Value *pointer,
                                 const std::vector<Value *> &indices = {},
                                 const std::string &name = "") {
        auto inst = new StoreInst(value, pointer, _block, indices, name);
        _block->get_insts().emplace(_position, inst);
        return inst;
    }
};

} // namespace ir
