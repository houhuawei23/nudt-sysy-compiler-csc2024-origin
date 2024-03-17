#pragma once
#include <algorithm>  // for block list sort
#include <queue>      // for block list priority queue
#include "infrast.hpp"
#include "module.hpp"
#include "type.hpp"
#include "utils.hpp"
#include "value.hpp"
namespace ir {

using arg_list = std::list<Argument*>;      // vector -> list
using block_list = std::list<BasicBlock*>;  // vector -> list

// struct
// // 比较函数，按照A对象的priority属性从小到大排序

// Value: _type, _name, _uses
class Function : public User {
    friend class Module;
    // _type = FUNCTION
   protected:
    Module* _parent;  // parent Module

    block_list _blocks;       // blocks of the function
    block_list _exit_blocks;  // exit blocks
    arg_list _args;
    // std::priority_queue<BasicBlock*, std::vector<BasicBlock*>,
   public:
    Function(Type* func_type,
             const std::string& name = "",
             Module* parent = nullptr)
        : User(func_type, vFUNCTION, name), _parent(parent) {}

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
    BasicBlock* new_block();
    block_list& blocks() { return _blocks; }

    void sort_blocks() {
        _blocks.sort(compareBB);
        // std::sort(_blocks.begin(), _blocks.end());
        // _blocks.sort(
        //     [](const BasicBlock* a1, const BasicBlock* a2) { return a1 < a2; });
    }
    static bool classof(const Value* v) { return v->scid() == vFUNCTION; }

    void print(std::ostream& os) const override;
};
}  // namespace ir