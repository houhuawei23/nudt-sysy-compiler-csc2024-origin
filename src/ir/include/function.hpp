#pragma once

#include "value.hpp"
#include "infrast.hpp"
#include "module.hpp"


namespace ir {
// using inst_list = std::list<std::unique_ptr<Instruction>>; // list
// using iterator = inst_list::iterator;
// using reverse_iterator = inst_list::reverse_iterator;

using arg_list = std::list<std::unique_ptr<Argument>>;     // vector -> list
using block_list = std::list<std::unique_ptr<BasicBlock>>; // vector -> list

// Value: _type, _name, _uses
class Function : public Value {
  friend class Module;
  protected:
    Module *_parent; // parent Module

    block_list _blocks;      // blocks of the function
    block_list _exit_blocks; // exit blocks
    arg_list _arguments;

  public:
    Function(Module *parent, Type *type, const std::string &name)
        : Value(type, name), _parent(parent) {}

    Type *get_ret_type() const {
        // this->get_type() return Type*
        // need cast to FunctionType* to call get_ret_type()
        return dynamic_cast<FunctionType *>(this->get_type())->get_ret_type();
    }

    BasicBlock *add_bblock(const std::string &name);
};
} // namespace ir