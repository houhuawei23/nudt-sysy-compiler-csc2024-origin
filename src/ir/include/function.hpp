#pragma once
#include "infrast.hpp"
#include "module.hpp"
#include "type.hpp"
#include "utils.hpp"
#include "value.hpp"
namespace ir {

using arg_list = std::list<Argument*>;      // vector -> list
using block_list = std::list<BasicBlock*>;  // vector -> list

// Value: _type, _name, _uses
class Function : public Value {
    friend class Module;
    // _type = FUNCTION
   protected:
    Module* _parent;  // parent Module

    block_list _blocks;       // blocks of the function
    block_list _exit_blocks;  // exit blocks
    arg_list _args;

   public:
    Function(Type* func_type,
             const std::string& name = "",
             Module* parent = nullptr)
        : Value(func_type, vFUNCTION, name), _parent(parent) {}

    Type* ret_type() const {
        // this->type() return Type*
        // need cast to FunctionType* to call ret_type()
        FunctionType* ftype = dyn_cast<FunctionType>(this->type());

        return ftype->ret_type();
    }

    // arg_list param_type()const {
    //   return dynamic_cast<FunctionType
    //   *>(this->type())->param_type();
    // }
    BasicBlock* add_block();
    block_list& blocks() { return _blocks; }

    static bool classof(const Value* v) { return v->scid() == vFUNCTION; }

    void print(std::ostream& os) const override;
};
}  // namespace ir