#pragma once

#include "ir/type.hpp"
#include "ir/utils_ir.hpp"
#include "ir/value.hpp"
#include <algorithm>
namespace ir {
/**
 * @brief Argument represents an incoming formal argument to a Function.
 * 形式参数，因为它是“形式的”，所以不包含实际值，而是表示特定函数的参数的类型、参数编号和属性。
 * 当在所述函数体中使用时，参数当然代表调用该函数的实际参数的值。
 */
class Argument : public Value {
 protected:
  Function* mFunction;
  size_t mIndex;

 public:
  Argument(Type* type,
           size_t index,
           Function* parent = nullptr,
           const_str_ref name = "")
      : Value(type, vARGUMENT, name), mIndex(index), mFunction(parent) {}

  auto function() const { return mFunction; }

  auto index() const { return mIndex; }

  static bool classof(const Value* v) { return v->valueId() == vARGUMENT; }

  void print(std::ostream& os) const override;
};

/**
 * @brief The container for `Instruction` sequence.
 * `BasicBlock` maintains a list of `Instruction`s, with the last one being a
 * terminator (branch or return). Besides, `BasicBlock` stores its arguments and
 * records its predecessor and successor `BasicBlock`s.
 */
class BasicBlock : public Value {
  // mType: TypeLabel()

//   // dom info for anlaysis
//  public:
//   BasicBlock* idom;
//   BasicBlock* sdom;
//   BasicBlock* ipdom;
//   BasicBlock* spdom;
//   std::vector<BasicBlock*> domTree;      // sons in dom Tree
//   std::vector<BasicBlock*> domFrontier;  // dom frontier
//   std::vector<BasicBlock*> pdomTree;
//   std::vector<BasicBlock*> pdomFrontier;
//   // std::set<BasicBlock*> dom;//those bb was dominated by self
//   size_t domLevel;
//   size_t pdomLevel;
  // size_t looplevel;

 protected:
  Function* mFunction;
  inst_list mInsts;

  // for CFG
  block_ptr_list mNextBlocks;
  block_ptr_list mPreBlocks;
  // specially for Phi
  inst_list mPhiInsts;

  size_t mDepth = 0;

  size_t mIdx = 0;

 public:
  BasicBlock(const_str_ref name = "", Function* parent = nullptr)
      : Value(Type::TypeLabel(), vBASIC_BLOCK, name), mFunction(parent){};
  auto idx() const { return mIdx; }
  void set_idx(uint32_t idx) { mIdx = idx; }
  /* must override */
  std::string name() const override { return "bb" + std::to_string(mIdx); }
  auto depth() const { return mDepth; }

  auto empty() const { return mInsts.empty(); }

  //* get Data Attributes
  auto function() const { return mFunction; }

  void set_parent(Function* parent) { mFunction = parent; }

  auto& insts() { return mInsts; }

  auto& phi_insts() { return mPhiInsts; }

  auto& next_blocks() const { return mNextBlocks; }
  auto& pre_blocks() const { return mPreBlocks; }
  auto& next_blocks() { return mNextBlocks; }
  auto& pre_blocks() { return mPreBlocks; }

  void set_depth(size_t d) { mDepth = d; }  // ?
  void emplace_inst(inst_iterator pos, Instruction* i);

  void emplace_first_inst(Instruction* i);
  void emplace_back_inst(Instruction* i);

  void delete_inst(Instruction* inst);

  void force_delete_inst(Instruction* inst);

  void replaceinst(Instruction* old, Value* new_);

  auto terminator() { return mInsts.back(); }
  static void BasicBlockDfs(BasicBlock* bb,
                            std::function<bool(BasicBlock*)> func) {
    if (func(bb))
      return;
    for (auto succ : bb->next_blocks())
      BasicBlockDfs(succ, func);
  }
  // for CFG
  static void block_link(ir::BasicBlock* pre, ir::BasicBlock* next) {
    pre->next_blocks().emplace_back(next);
    next->pre_blocks().emplace_back(pre);
  }

  static void delete_block_link(ir::BasicBlock* pre, ir::BasicBlock* next) {
    pre->next_blocks().remove(next);
    next->pre_blocks().remove(pre);
  }

  // bool dominate(BasicBlock* bb) {
  //   if (this == bb)
  //     return true;
  //   for (auto bbnext : domTree) {
  //     if (bbnext->dominate(bb))
  //       return true;
  //   }
  //   return false;
  // }
  bool isTerminal() const;

  static bool classof(const Value* v) { return v->valueId() == vBASIC_BLOCK; }

  void print(std::ostream& os) const override;
};

/**
 * @brief Base class for all instructions in IR
 *
 */
class Instruction : public User {
  // Instuction 的类型也通过 mValueId
 protected:
  BasicBlock* mBlock;

 public:
  // Construct a new Instruction object
  Instruction(ValueId itype = vINSTRUCTION,
              Type* ret_type = Type::void_type(),
              BasicBlock* pblock = nullptr,
              const_str_ref name = "")
      : User(ret_type, itype, name), mBlock(pblock) {}
  // get
  auto block() const { return mBlock; };

  // set
  void setBlock(BasicBlock* parent) { mBlock = parent; }

  // inst type check
  bool isTerminator();
  bool isUnary();
  bool isBinary();
  bool isBitWise();
  bool isMemory();
  bool isNoName();
  bool isAggressiveAlive();

  // for isa, cast and dyn_cast
  static bool classof(const Value* v) { return v->valueId() >= vINSTRUCTION; }

  void setvarname();  // change varname to pass lli

  void virtual print(std::ostream& os) const = 0;

  virtual Value* getConstantRepl() { return nullptr; };
  Instruction* copy_inst(std::function<Value*(Value*)> getValue);
};

}  // namespace ir