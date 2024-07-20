#pragma once

#include "ir/infrast.hpp"
#include "ir/module.hpp"
#include "ir/type.hpp"
#include "ir/value.hpp"
#include "support/utils.hpp"
#include "support/arena.hpp"

namespace ir {

class Loop;

class Loop {
protected:
    std::set<BasicBlock*> _blocks;
    BasicBlock* _header;
    Function* _parent;
    std::set<BasicBlock*> _exits;
    std::set<BasicBlock*> _latchs;
    std::set<Loop*> _subLoops;
    Loop* _parentLoop;

public:
    Loop(BasicBlock* header, Function* parent) {
        _header = header;
        _parent = parent;
    }
    BasicBlock* header() const{ return _header; }
    Function* parent() const{ return _parent; }
    std::set<BasicBlock*>& blocks() { return _blocks; }
    std::set<BasicBlock*>& exits() { return _exits; }
    std::set<BasicBlock*>& latchs() {return _latchs; }
    std::set<Loop*>& subLoops() {return _subLoops; }
    Loop* parent(){return _parentLoop;}
    void setParent(Loop* lp){_parentLoop=lp;}
    bool contains(BasicBlock* block) const{ return _blocks.find(block) != _blocks.end(); }
    BasicBlock* getlooppPredecessor() const{
      BasicBlock* predecessor = nullptr;
      BasicBlock* Header = header();
      for (auto* pred : Header->pre_blocks()){
        if(!contains(pred)){
          if (predecessor && (predecessor != pred)){
            return nullptr;//多个前驱
          }
          predecessor = pred;
        }
      }
      return predecessor;//返回唯一的predecessor
    }
    BasicBlock* getLoopPreheader() const{
      BasicBlock* preheader = getlooppPredecessor();
      if (!preheader)
        return nullptr;
      if (preheader->next_blocks().size() != 1)
        return nullptr;
      return preheader;
    }
    BasicBlock* getLoopLatch() const{
      BasicBlock* latch = nullptr;
      BasicBlock* Header = header();
      for (auto* pred : Header->pre_blocks()){
        if (contains(pred)){
          if (latch)
            return nullptr;
          latch = pred;
        }
      }
      return latch;//返回唯一的latch
    }
    bool hasDedicatedExits() const{
      for (auto exitbb : _exits){
        for (auto pred : exitbb->pre_blocks()){
          if (!contains(pred))
            return false;
        }
      }
      return true;
    }
    bool isLoopSimplifyForm() const{
      return getLoopPreheader() && getLoopLatch() && hasDedicatedExits();
    }


};

class Function : public User {
  friend class Module;

 protected:
  Module* mModule = nullptr;  // parent Module

  block_ptr_list mBlocks;     // blocks of the function
  arg_ptr_vector mArguments;  // formal args

  Value* mRetValueAddr = nullptr;  // return value
  BasicBlock* mEntry = nullptr;    // entry block
  BasicBlock* mExit = nullptr;     // exit block
  size_t mVarCnt = 0;              // for local variables count
  size_t argCnt = 0;               // formal arguments count

 public:
  Function(Type* TypeFunction, const_str_ref name="",
           Module* parent=nullptr)
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
    auto arg = utils::make<Argument>(btype, argCnt, this, name);
    argCnt++;
    mArguments.emplace_back(arg);
    return arg;
  }

  auto varInc() { return mVarCnt++; }
  void setVarCnt(size_t x) { mVarCnt = x; }

  bool isOnlyDeclare(){return mBlocks.empty();}


 public:
  static bool classof(const Value* v) { return v->valueId() == vFUNCTION; }
  ir::Function* copy_func();

  void rename();
  void print(std::ostream& os) const override;
  bool verify(std::ostream& os) const;
};
}  // namespace ir