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
  BasicBlock* mBlock = nullptr;  // current basic block for insert instruction
  inst_iterator mInsertPos;      // insert pos for cur block
  block_ptr_stack _headers, _exits;
  size_t ifNum, whileNum, rhsNum, funcNum, varNum;

  // true/false br targets stack for if-else Short-circuit evaluation
  // the top the deeper nest
  block_ptr_stack _true_targets, _false_targets;
  size_t blockNum;

 public:
  IRBuilder() {
    ifNum = 0;
    whileNum = 0;
    rhsNum = 0;
    funcNum = 0;
    varNum = 0;
    blockNum = 0;
  }

  void reset() {
    ifNum = 0;
    whileNum = 0;
    rhsNum = 0;
    funcNum = 0;
    varNum = 0;
    blockNum = 0;
  }

  //! get
  auto curBlock() const { return mBlock; }
  auto position() const { return mInsertPos; }

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
    assert(block != nullptr);
    mBlock = block;
    mInsertPos = pos;  // mInsertPos 与 ->end() 绑定?
  }

  void set_pos(BasicBlock* block) {
    assert(block != nullptr and block->insts().empty());
    mBlock = block;
    mInsertPos = block->insts().begin();
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

  void if_inc() { ifNum++; }
  void while_inc() { whileNum++; }
  void rhs_inc() { rhsNum++; }
  void func_inc() { funcNum++; }

  auto if_cnt() const { return ifNum; }
  auto while_cnt() const { return whileNum; }
  auto rhs_cnt() const { return rhsNum; }
  auto func_cnt() const { return funcNum; }

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

  Value* castToBool(Value* val);
  // defalut promote to i32
  Value* promoteType(Value* val,
                     Type* target_tpye,
                     Type* base_type = Type::TypeInt32());

  Value* makeCmp(CmpOp op, Value* lhs, Value* rhs);

  Value* makeBinary(BinaryOp op, Value* lhs, Value* rhs);

  Value* makeUnary(ValueId vid, Value* val, Type* ty = nullptr);

  Value* makeAlloca(Type* base_type,
                    bool is_const = false,
                    const std::vector<size_t>& dims = {},
                    const_str_ref comment = "",
                    size_t capacity = 1);
  Value* makeLoad(Value* ptr);

  Value* makeGetElementPtr(Type* base_type,
                           Value* value,
                           Value* idx,
                           std::vector<size_t> dims = {},
                           std::vector<size_t> cur_dims = {});

  template <typename T, typename... Args>
  auto makeInst(Args&&... args) {
    auto inst = new T(std::forward<Args>(args)...);
    inst->setBlock(mBlock);
    mBlock->emplace_back_inst(inst);
    return inst;
  }

  template <typename T, typename... Args>
  auto makeIdenticalInst(Args&&... args) {
    return new T(std::forward<Args>(args)...);
  }
};

}  // namespace ir
