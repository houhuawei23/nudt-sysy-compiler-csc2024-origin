#pragma once

#include "ir/infrast.hpp"
#include "ir/module.hpp"
#include "ir/type.hpp"
#include "ir/value.hpp"
#include "support/utils.hpp"

namespace ir {

class Loop {
protected:
    std::set<BasicBlock*> _blocks;
    BasicBlock* _header;
    Function* _parent;
    std::set<BasicBlock*> _exits;

public:
    Loop(BasicBlock* header, Function* parent) {
        _header = header;
        _parent = parent;
    }
    BasicBlock* header() { return _header; }
    Function* parent() { return _parent; }
    std::set<BasicBlock*>& blocks() { return _blocks; }
};

class Function : public User {
  friend class Module;

 protected:
  Module* mModule = nullptr;  // parent Module

  block_ptr_list mBlocks;     // blocks of the function
  arg_ptr_vector mArguments;  // formal args
  // std::list<Loop*> mLoops;
  // std::map<ir::BasicBlock*, ir::Loop*> _headToLoop;

  //* function has concrete local var for return value,
  //* addressed by mRetValueAddr
  Value* mRetValueAddr = nullptr;  // return value
  BasicBlock* mEntry = nullptr;    // entry block
  BasicBlock* mExit = nullptr;     // exit block
  size_t mVarCnt = 0;              // for local variables count
  size_t argCnt = 0;               // formal arguments count

  // // for call graph
  // std::set<ir::Function*> mCallees;
  // bool _is_called;
  // bool _is_inline;
  // bool _is_lib;

 public:
  Function(Type* TypeFunction,
           const_str_ref name = "",
           Module* parent = nullptr)
      : User(TypeFunction, vFUNCTION, name), mModule(parent) {
    argCnt = 0;
    mRetValueAddr = nullptr;
  }

  //* get
  auto module() const { return mModule; }

  // return value
  auto retValPtr() const { return mRetValueAddr; }
  auto retType() const { return mType->as<FunctionType>()->retType(); }

  void setRetValueAddr(Value* value) {
    assert(mRetValueAddr == nullptr && "new_ret_value can not call 2th");
    mRetValueAddr = value;
  }

  //* Block
  auto& blocks() const { return mBlocks; }
  auto& blocks() { return mBlocks; }

  auto entry() const { return mEntry; }
  void setEntry(ir::BasicBlock* bb) { mEntry = bb; }

  auto exit() const { return mExit; }
  void setExit(ir::BasicBlock* bb) { mExit = bb; }

  BasicBlock* newBlock();
  BasicBlock* newEntry(const_str_ref name = "");
  BasicBlock* newExit(const_str_ref name = "");

  void delBlock(BasicBlock* bb);
  void forceDelBlock(BasicBlock* bb);

  //* Arguments
  auto& args() const { return mArguments; }
  auto& argTypes() const { return mType->as<FunctionType>()->argTypes(); }

  auto arg_i(size_t idx) {
    assert(idx < argCnt && "idx out of args vector");
    return mArguments[idx];
  }

  auto new_arg(Type* btype, const_str_ref name = "") {
    auto arg = new Argument(btype, argCnt, this, name);
    argCnt++;
    mArguments.emplace_back(arg);
    return arg;
  }

  auto varInc() { return mVarCnt++; }
  void setVarCnt(size_t x) { mVarCnt = x; }

  // auto& Loops() { return mLoops; }
  // auto& headToLoop() { return _headToLoop; }

  // auto& callees() const { return mCallees; }
  // auto& callees() { return mCallees; }
  // // get and set for callgraph
  // bool get_is_inline() { return _is_inline; }
  // void set_is_inline(bool b) { _is_inline = b; }
  // bool get_is_called() { return _is_called; }
  // void set_is_called(bool b) { _is_called = b; }
  // bool get_is_lib() { return _is_lib; }
  // void set_is_lib(bool b) { _is_lib = b; }

 public:
  static bool classof(const Value* v) { return v->valueId() == vFUNCTION; }
  ir::Function* copy_func();

  void rename();
  void print(std::ostream& os) const override;
};
}  // namespace ir