#pragma once

#include "ir/infrast.hpp"
#include "ir/module.hpp"
#include "ir/type.hpp"
#include "support/utils.hpp"
#include "ir/value.hpp"

namespace ir {

// for BasicBlock sort
inline bool compareBB(const BasicBlock* a1, const BasicBlock* a2) {
    // return a1->priority < a2->priority;
    if (a1->name().size() > 1 && a2->name().size() > 1)
        return std::stoi(a1->name().substr(2)) <
               std::stoi(a2->name().substr(2));
    else {
        assert(false && "compareBB error");
    }
}

class Function : public User {
    friend class Module;
    //* Inherited Data Attribute
    // Value: _type = FUNCTION, _name, _uses
    // User:  _operands;
   protected:
    Module* _parent = nullptr;  // parent Module

    block_ptr_list _blocks;       // blocks of the function
    block_ptr_list _exit_blocks;  // exit blocks
    arg_ptr_vector _args;         // formal args

    //* function has concrete local var for return value,
    //* addressed by _ret_value_ptr
    Value* _ret_value_ptr = nullptr;  // return value
    BasicBlock* _entry = nullptr;     // entry block
    BasicBlock* _exit = nullptr;      // exit block

    int var_cnt = 0;   // for local variables count
    int _arg_cnt = 0;  // formal arguments count

    bool _is_defined = false;

   public:
    Function(Type* func_type, const_str_ref name = "", Module* parent = nullptr)
        : User(func_type, vFUNCTION, name), _parent(parent) {
        _is_defined = false;
        _arg_cnt = 0;
        _ret_value_ptr = nullptr;
    }

    //* get
    int getvarcnt() { return var_cnt++; }

    Module* parent() const { return _parent; }

    //* return
    Type* ret_type() const {
        FunctionType* ftype = dyn_cast<FunctionType>(type());
        return ftype->ret_type();
    }

    Value* ret_value_ptr() const { return _ret_value_ptr; }

    void set_ret_value_ptr(Value* value) {
        assert(_ret_value_ptr == nullptr && "new_ret_value can not call 2th");
        _ret_value_ptr = value;
    }

    //* Block
    block_ptr_list& blocks() { return _blocks; }

    block_ptr_list& exit_blocks() { return _exit_blocks; }

    auto entry() const { return _entry; }

    auto exit() const { return _exit; }

    BasicBlock* new_block();

    BasicBlock* new_entry(const_str_ref name = "") {
        assert(_entry == nullptr);
        _entry = new BasicBlock(name, this);
        _blocks.emplace_back(_entry);
        return _entry;
    }

    BasicBlock* new_exit(const_str_ref name = "") {
        assert(_exit == nullptr);
        _exit = new BasicBlock(name, this);
        _blocks.emplace_back(_exit);
        return _exit;
    }

    //* Arguments
    arg_ptr_vector& args() { return _args; }

    type_ptr_vector& arg_types() {
        return dyn_cast<FunctionType>(type())->arg_types();
    }

    Argument* arg_i(int idx) {
        assert(idx < _arg_cnt && "idx out of args vector");
        return _args[idx];
    }

    Argument* new_arg(Type* btype, const_str_ref name="") {
        auto arg = new Argument(btype, _arg_cnt, this, name);
        _arg_cnt++;
        _args.emplace_back(arg);
        return arg;
    }

    //* print blocks in ascending order
    void sort_blocks() { _blocks.sort(compareBB); }

    // isa<>
    static bool classof(const Value* v) { return v->scid() == vFUNCTION; }
    
    // ir print
    void print(std::ostream& os) override;
};
}  // namespace ir