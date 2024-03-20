#pragma once

#include "infrast.hpp"
#include "module.hpp"
#include "type.hpp"
#include "utils.hpp"
#include "value.hpp"

namespace ir {

inline bool compareBB(const BasicBlock* a1, const BasicBlock* a2) {
    // return a1->priority < a2->priority;
    if (a1->name().size() > 1 && a2->name().size() > 1)
        return std::stoi(a1->name().substr(2)) <
               std::stoi(a2->name().substr(2));
    else {
        // std::
        assert(false && "compareBB error");
    }
}

// Value: _type, _name, _uses
class Function : public User {
    friend class Module;
    // _type = FUNCTION
   protected:
    Module* _parent = nullptr;  // parent Module

    // BasicBlock* _entry;            // entry block
    block_ptr_list _blocks;       // blocks of the function
    block_ptr_list _exit_blocks;  // exit blocks
    arg_ptr_vector _args;         // formal args

    int _arg_cnt = 0;
    bool _is_defined = false;

    Value* _ret_value_ptr = nullptr;  // return value
    BasicBlock* _entry = nullptr;
    BasicBlock* _exit = nullptr;

   public:
    Function(Type* func_type, const_str_ref name = "", Module* parent = nullptr)
        : User(func_type, vFUNCTION, name), _parent(parent) {
        _is_defined = false;
        _arg_cnt = 0;
        _ret_value_ptr = nullptr;
        // entry = new BasicBlock(this);
    }
    //! new for entry, exit, ret_value_ptr
    Value* ret_value_ptr() const { return _ret_value_ptr; }
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
    auto entry() const { return _entry; }
    auto exit() const { return _exit; }
    void set_ret_value_ptr(Value* value) {
        assert(_ret_value_ptr == nullptr && "new_ret_value can not call 2th");
        _ret_value_ptr = value;
        // return _ret_value_ptr;
    }

    //* get
    Module* parent() const { return _parent; }
    block_ptr_list& blocks() { return _blocks; }
    block_ptr_list& exit_blocks() { return _exit_blocks; }

    // BasicBlock* entry() const { return _entry; }

    // BasicBlock* create_entry(const_str_ref name = "") {
    //     _entry = new BasicBlock(name, this);
    //     return _entry;
    // }

    arg_ptr_vector& args() { return _args; }

    Argument* arg_i(int idx) {
        assert(idx < _arg_cnt && "idx out of args vector");
        return _args[idx];
    }

    Type* ret_type() const {
        FunctionType* ftype = dyn_cast<FunctionType>(type());
        return ftype->ret_type();
    }

    //* manage
    type_ptr_vector& arg_types() {
        return dyn_cast<FunctionType>(type())->arg_types();
    }

    BasicBlock* new_block();

    Argument* new_arg(Type* btype, const_str_ref name) {
        auto arg = new Argument(btype, _arg_cnt, this, name);
        _arg_cnt++;
        _args.emplace_back(arg);
        return arg;
    }

    void sort_blocks() { _blocks.sort(compareBB); }

    static bool classof(const Value* v) { return v->scid() == vFUNCTION; }

    void print(std::ostream& os) const override;
};
}  // namespace ir