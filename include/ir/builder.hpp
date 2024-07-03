/*
 * @file builder.hpp
 *
 */
#pragma once

#include <any>
#include "ir/infrast.hpp"
#include "ir/instructions.hpp"
namespace ir {

/*
 * @brief: IR Builder for Module.
 *
 */
class IRBuilder {
   private:
    BasicBlock* _block = nullptr;  // current basic block for insert instruction
    inst_iterator _pos;            // insert pos for cur block
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

    void reset() {
        _var_cnt = 0;
        _bb_cnt = 0;
    }

    //! get
    std::string get_bbname() { return "bb" + std::to_string(_bb_cnt++); }
    uint32_t get_bbidx() { return _bb_cnt++; }

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
    auto true_target() const { return _true_targets.top(); }
    auto false_target() const { return _false_targets.top(); }

    void pop_tf() {
        _true_targets.pop();
        _false_targets.pop();
    }

    //! Create Alloca Instruction
    Value* create_alloca(Type* base_type,
                              bool is_const = false,
                              std::vector<int> dims = {},
                              const_str_ref comment = "",
                              int capacity = 1);
    Value* create_load(Value* ptr);

    //! Create GetElementPtr Instruction
    Value* create_getelementptr(Type* base_type,
                                            Value* value,
                                            Value* idx,
                                            std::vector<int> dims = {},
                                            std::vector<int> cur_dims = {});

    Value* cast_to_i1(Value* val);
    // using pair?
    // defalut promote to i32
    Value* type_promote(Value* val,
                        Type* target_tpye,
                        Type* base_type = Type::i32_type());

    Value* create_cmp(CmpOp op, Value* lhs, Value* rhs);

    Value* create_binary_beta(BinaryOp op, Value* lhs, Value* rhs);

    Value* create_unary_beta(ValueId vid, Value* val, Type* ty = nullptr);

    template <typename T, typename... Args>
    auto makeInst(Args&&... args) {
        // auto inst = make<T>(std::forward<Args>(args)...);
        // auto block = mCurrentBlock;
        // inst->insertBefore(block, mInsertPoint);
        auto inst = new T(std::forward<Args>(args)...);
        block()->emplace_back_inst(inst);
        return inst;
    }
};

}  // namespace ir
