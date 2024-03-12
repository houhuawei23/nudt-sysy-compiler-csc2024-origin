#pragma once
#include "type.hpp"
#include "infrast.hpp"
#include "module.hpp"
#include "value.hpp"

namespace ir {
// using inst_list = std::list<std::unique_ptr<Instruction>>; // list
// using iterator = inst_list::iterator;
// using reverse_iterator = inst_list::reverse_iterator;

using arg_list = std::list<std::unique_ptr<Argument>>;     // vector -> list
using block_list = std::list<std::unique_ptr<BasicBlock>>; // vector -> list

// Value: _type, _name, _uses
class Function : public Value {
    friend class Module;
    // _type = FUNCTION
  protected:
    Module *_parent; // parent Module

    block_list _blocks;      // blocks of the function
    block_list _exit_blocks; // exit blocks
    arg_list _arguments;

  public:
    Function(Type *func_type, const std::string &name = "",
             Module *parent = nullptr)
        : Value(func_type, name), _parent(parent) {}

    Type *ret_type() const {
        // this->type() return Type*
        // need cast to FunctionType* to call ret_type()
        // FunctionType *ftype = dynamic_cast<FunctionType *>(this->type());
        FunctionType *ftype = this->type()->as<FunctionType>();
        return ftype->ret_type();
    }

    // arg_list param_type()const {
    //   return dynamic_cast<FunctionType
    //   *>(this->type())->param_type();
    // }
    BasicBlock *add_block(const std::string &name);
    block_list &blocks() { return _blocks; }

    void print(std::ostream &os) const override;
};
} // namespace ir