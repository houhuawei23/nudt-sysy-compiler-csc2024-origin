#pragma once

#include "infrast.hpp"
#include "module.hpp"
#include "type.hpp"
#include "utils.hpp"
#include "value.hpp"

namespace ir {

// Value: _type, _name, _uses
class Function : public User {
    friend class Module;
    // _type = FUNCTION
   protected:
    Module* _parent;  // parent Module

    block_ptr_list _blocks;       // blocks of the function
    block_ptr_list _exit_blocks;  // exit blocks
    arg_ptr_list _args;           // formal args

   public:
    Function(Type* func_type, const_str_ref name = "", Module* parent = nullptr)
        : User(func_type, vFUNCTION, name), _parent(parent) {}

    //* get
    Module* parent() const { return _parent; }
    block_ptr_list& blocks() { return _blocks; }
    block_ptr_list& exit_blocks() { return _exit_blocks; }
    arg_ptr_list& args() { return _args; }

    Type* ret_type() const {
        // this->type() return Type*
        // need cast to FunctionType* to call ret_type()
        FunctionType* ftype = dyn_cast<FunctionType>(this->type());
        return ftype->ret_type();
    }

    //* manage
    type_ptr_vector& arg_types() const {
        return dyn_cast<FunctionType>(type())->arg_types();
        //   return dynamic_cast<FunctionType*>(this->type())->param_type();
    }
    BasicBlock* new_block();

    void sort_blocks() { _blocks.sort(compareBB); }
    static bool classof(const Value* v) { return v->scid() == vFUNCTION; }

    void print(std::ostream& os) const override;
};
}  // namespace ir