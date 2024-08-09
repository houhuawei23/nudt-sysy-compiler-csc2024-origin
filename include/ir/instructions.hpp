#pragma once
#include "ir/function.hpp"
#include "ir/global.hpp"
#include "ir/infrast.hpp"
#include "ir/value.hpp"
#include "support/utils.hpp"
#include "ir/ConstantValue.hpp"
#include <initializer_list>

namespace ir {

class AllocaInst;
class LoadInst;
class StoreInst;
class GetElementPtrInst;
class BitcastInst;

class ReturnInst;
class BranchInst;

class UnaryInst;
class BinaryInst;

class ICmpInst;
class FCmpInst;
class CallInst;

class PhiInst;

class indVar;
/*
 * @brief: AllocaInst
 */
class AllocaInst : public Instruction {
protected:
  bool mIsConst = false;

public:  // 构造函数
  //! 1. Alloca Scalar
  AllocaInst(Type* base_type,
             BasicBlock* parent = nullptr,
             const_str_ref name = "",
             bool is_const = false)
    : Instruction(vALLOCA, ir::Type::TypePointer(base_type), parent, name), mIsConst(is_const) {}

  //! 2. Alloca Array
  AllocaInst(Type* base_type,
             std::vector<size_t> dims,
             BasicBlock* parent = nullptr,
             const_str_ref name = "",
             bool is_const = false,
             size_t capacity = 1)
    : Instruction(vALLOCA,
                  ir::Type::TypePointer(ir::Type::TypeArray(base_type, dims, capacity)),
                  parent,
                  name),
      mIsConst(is_const) {}

public:  // get function
  Type* baseType() const {
    assert(mType->dynCast<PointerType>() && "type error");
    return mType->as<PointerType>()->baseType();
  }
  auto dimsCnt() const {
    if (baseType()->isArray())
      return dyn_cast<ArrayType>(baseType())->dims_cnt();
    else
      return static_cast<size_t>(0);
  }

public:  // check function
  bool isScalar() const { return !baseType()->isArray(); }
  bool isConst() const { return mIsConst; }

public:
  static bool classof(const Value* v) { return v->valueId() == vALLOCA; }
  void print(std::ostream& os) const override;
  void dumpAsOpernd(std::ostream& os) const override { os << mName; }
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    if (isScalar()) {
      auto inst = utils::make<AllocaInst>(baseType());
      inst->setComment(mComment);
      return inst;
    } else {
      auto basetype = baseType()->dynCast<ArrayType>();
      auto capacity = basetype->size();
      auto inst = utils::make<AllocaInst>(basetype->baseType(), basetype->dims(), nullptr, "",
                                          false, capacity);
      inst->setComment(mComment);
      return inst;
    }
  };
};

class StoreInst : public Instruction {
public:
  StoreInst(Value* value, Value* ptr, BasicBlock* parent = nullptr)
    : Instruction(vSTORE, Type::void_type(), parent) {
    addOperand(value);
    addOperand(ptr);
  }

public:
  Value* value() const { return operand(0); }
  Value* ptr() const { return operand(1); }

public:
  static bool classof(const Value* v) { return v->valueId() == vSTORE; }
  void print(std::ostream& os) const override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    auto inst = utils::make<StoreInst>(getValue(operand(0)), getValue(operand(1)));
    inst->setComment(mComment);
    return inst;
  }
};

/*
 * @brief Load Instruction
 * @details:
 *      <result> = load <ty>, ptr <pointer>
 */
class LoadInst : public Instruction {
public:
  LoadInst(Value* ptr, Type* type, BasicBlock* parent = nullptr)
    : Instruction(vLOAD, type, parent) {
    addOperand(ptr);
  }

  auto ptr() const { return operand(0); }
  static bool classof(const Value* v) { return v->valueId() == vLOAD; }
  void print(std::ostream& os) const override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    auto inst = utils::make<LoadInst>(getValue(operand(0)), mType);
    inst->setComment(mComment);
    return inst;
  }
};

/*
 * @brief Return Instruction
 * @details:
 *      ret <type> <value>
 *      ret void
 */
class ReturnInst : public Instruction {
public:
  ReturnInst(Value* value = nullptr, BasicBlock* parent = nullptr, const_str_ref name = "")
    : Instruction(vRETURN, Type::void_type(), parent, name) {
    if (value) {
      addOperand(value);
    }
  }

public:
  bool hasRetValue() const { return not mOperands.empty(); }
  Value* returnValue() const { return hasRetValue() ? operand(0) : nullptr; }

public:
  static bool classof(const Value* v) { return v->valueId() == vRETURN; }
  void print(std::ostream& os) const override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    return utils::make<ReturnInst>(getValue(returnValue()));
  }
};

/*
 * @brief Unary Instruction
 * @details: 
 *    trunc, zext, sext, fptrunc, fptosi, sitofp,
 *    bitcast, ptrtoint, inttoptr
 */
class UnaryInst : public Instruction {
public:
  UnaryInst(ValueId kind, Type* type, Value* operand,
            BasicBlock* parent=nullptr, const_str_ref name="")
    : Instruction(kind, type, parent, name) {
    addOperand(operand);
  }
public:  // get function
  auto value() const { return operand(0); }
public:  // utils function
  void print(std::ostream& os) const override;
  Value* getConstantRepl(bool recursive = false) override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    return utils::make<UnaryInst>(mValueId, mType, getValue(operand(0)));
  }
  static bool classof(const Value* v) {
    return v->valueId() >= vUNARY_BEGIN && v->valueId() <= vUNARY_END;
  }
};

/*
 * @brief: Binary Instruction
 * @note:
 *    1. exp (MUL | DIV | MODULO) exp
 *    2. exp (ADD | SUB) exp
 */
class BinaryInst : public Instruction {
public:
  BinaryInst(ValueId kind, Type* type, Value* lvalue, Value* rvalue,
             BasicBlock* parent=nullptr, const std::string name="")
    : Instruction(kind, type, parent, name) {
    addOperand(lvalue);
    addOperand(rvalue);
  }
public:  // check function
  static bool classof(const Value* v) {
    return v->valueId() >= vBINARY_BEGIN && v->valueId() <= vBINARY_END;
  }
  bool isCommutative() const {
    return valueId() == vADD || valueId() == vFADD || valueId() == vMUL || valueId() == vFMUL;
  }
public:  // get function
  auto lValue() const { return operand(0); }
  auto rValue() const { return operand(1); }
public:  // utils function
  void print(std::ostream& os) const override;
  Value* getConstantRepl(bool recursive = false) override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    return utils::make<BinaryInst>(mValueId, mType, getValue(operand(0)), getValue(operand(1)));
  }
};
/* CallInst */
class CallInst : public Instruction {
  Function* mCallee = nullptr;
  bool mIsTail = false;
public:
  CallInst(Function* callee, const_value_ptr_vector rargs={},
           BasicBlock* parent=nullptr, const_str_ref name="")
    : Instruction(vCALL, callee->retType(), parent, name), mCallee(callee), mIsTail(false) {
    addOperands(rargs);
  }
  CallInst(Function* callee, std::initializer_list<Value*> rargs={},
           BasicBlock* parent=nullptr, const_str_ref name="")
    : Instruction(vCALL, callee->retType(), parent, name), mCallee(callee), mIsTail(false) {
    addOperands(rargs);
  }
public:  // check function
  bool istail() { return mIsTail; }
  bool isgetarrayorfarray() {
    return (mCallee->name() == "getarray") || (mCallee->name() == "getfarray");
  }
  bool isputarrayorfarray() {
    return (mCallee->name() == "putarray") || (mCallee->name() == "putfarray");
  }
public:  // set function
  void setIsTail(bool b) { mIsTail = b; }
public:  // get function
  Function* callee() const { return mCallee; }
  /* real arguments */
  auto& rargs() const { return mOperands; }
public:  // utils function
  static bool classof(const Value* v) { return v->valueId() == vCALL; }
  void print(std::ostream& os) const override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    std::vector<Value*> args;
    for (auto arg : mOperands) {
      args.push_back(getValue(arg->value()));
    }
    return utils::make<CallInst>(mCallee, args);
  }
  Instruction* clone() const override {
    std::vector<Value*> args;
    for (auto arg : mOperands) {
      args.push_back(arg->value());
    }
    return utils::make<CallInst>(mCallee, args);
  }
};

/*
 * @brief: Conditional or Unconditional Branch Instruction
 * @note:
 *    1. br i1 <cond>, label <iftrue>, label <iffalse>
 *    2. br label <dest>
 */
class BranchInst : public Instruction {
  bool mIsCond = false;
public:
  /* Condition Branch */
  BranchInst(Value* cond, BasicBlock* iftrue, BasicBlock* iffalse,
             BasicBlock* parent=nullptr, const_str_ref name="")
    : Instruction(vBR, Type::void_type(), parent, name), mIsCond(true) {
    addOperand(cond);
    addOperand(iftrue);
    addOperand(iffalse);
  }
  /* UnCondition Branch */
  BranchInst(BasicBlock* dest, BasicBlock* parent=nullptr, const_str_ref name="")
    : Instruction(vBR, Type::void_type(), parent, name), mIsCond(false) {
    addOperand(dest);
  }
public:  // get function
  bool is_cond() const { return mIsCond; }
  auto cond() const {
    assert(mIsCond && "not a conditional branch");
    return operand(0);
  }
  auto iftrue() const {
    assert(mIsCond && "not a conditional branch");
    return operand(1)->as<BasicBlock>();
  }
  auto iffalse() const {
    assert(mIsCond && "not a conditional branch");
    return operand(2)->as<BasicBlock>();
  }
  auto dest() const {
    assert(!mIsCond && "not an unconditional branch");
    return operand(0)->as<BasicBlock>();
  }
public:  // set function
  void replaceDest(ir::BasicBlock* olddest, ir::BasicBlock* newdest);
  void set_iftrue(BasicBlock* bb) {
    assert(mIsCond and "not a conditional branch");
    setOperand(1, bb);
  }
  void set_iffalse(BasicBlock* bb) {
    assert(mIsCond and "not a conditional branch");
    setOperand(2, bb);
  }
  void set_dest(BasicBlock* bb) {
    assert(not mIsCond and "not an unconditional branch");
    setOperand(0, bb);
  }
public:  // utils function
  static bool classof(const Value* v) { return v->valueId() == vBR; }
  void print(std::ostream& os) const override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    if (is_cond()) {
      return utils::make<BranchInst>(getValue(operand(0)), getValue(iftrue())->as<BasicBlock>(),
                                     getValue(iffalse())->as<BasicBlock>());
    } else {
      return utils::make<BranchInst>(getValue(dest())->as<BasicBlock>());
    }
  }
};

/*
 * @brief: ICmpInst
 * @note:
 *    <result> = icmp <cond> <ty> <op1>, <op2>
 */
class ICmpInst : public Instruction {
public:
  ICmpInst(ValueId itype, Value* lhs, Value* rhs,
           BasicBlock* parent=nullptr, const_str_ref name="")
    : Instruction(itype, Type::TypeBool(), parent, name) {
    addOperand(lhs);
    addOperand(rhs);
  }
public:  // get function
  auto lhs() const { return operand(0); }
  auto rhs() const { return operand(1); }
public:  // check function
  bool isReverse(ICmpInst* y);
public:  // set function
  void setCmpOp(ValueId newv) {
    assert(newv >= vICMP_BEGIN and newv <= vICMP_END);
    mValueId = newv;
  }
public:  // utils function
  static bool classof(const Value* v) {
    return v->valueId() >= vICMP_BEGIN && v->valueId() <= vICMP_END;
  }
  void print(std::ostream& os) const override;
  Value* getConstantRepl(bool recursive = false) override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    return utils::make<ICmpInst>(mValueId, getValue(operand(0)), getValue(operand(1)));
  }
};

/* FCmpInst */
class FCmpInst : public Instruction {
public:
  FCmpInst(ValueId itype, Value* lhs, Value* rhs,
           BasicBlock* parent=nullptr, const_str_ref name="")
    : Instruction(itype, Type::TypeBool(), parent, name) {
    addOperand(lhs);
    addOperand(rhs);
  }
public:  // get function
  auto lhs() const { return operand(0); }
  auto rhs() const { return operand(1); }
public:  // check function
  bool isReverse(FCmpInst* y);
public:  // utils function
  static bool classof(const Value* v) {
    return v->valueId() >= vFCMP_BEGIN && v->valueId() <= vFCMP_END;
  }
  void print(std::ostream& os) const override;
  Value* getConstantRepl(bool recursive = false) override;
  // Value* getConstantReplaceRecursive() override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    return utils::make<FCmpInst>(mValueId, getValue(operand(0)), getValue(operand(1)));
  }
};

/*
 * @brief Bitcast Instruction
 * @note: 
 *    <result> = bitcast <ty> <value> to i8*
 */
class BitCastInst : public Instruction {
public:
  BitCastInst(Type* dsttype, Value* value, BasicBlock* parent=nullptr)
    : Instruction(vBITCAST, dsttype, parent) {
    addOperand(value);
  }

public:  // get function
  auto value() const { return operand(0); }
public:  // utils function
  static bool classof(const Value* v) { return v->valueId() == vBITCAST; }
  void print(std::ostream& os) const override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    return utils::make<BitCastInst>(mType, getValue(operand(0)));
  }
};

/*
 * @brief: MemsetInst
 * @details:
 *    memset(i8* <dest>, i8 <val>, i64 <len>, i1 <isvolatile>)
 */
class MemsetInst : public Instruction {
public:
  MemsetInst(Value* dst, Value* val, Value* len, Value* isVolatile, BasicBlock* parent=nullptr)
    : Instruction(vMEMSET, Type::void_type(), parent) {
    addOperand(dst);
    addOperand(val);
    addOperand(len);
    addOperand(isVolatile);
  }
public:  // get function
  auto dst() const { return operand(0); }
  auto val() const { return operand(1); }
  auto len() const { return operand(2); }
  auto isVolatile() const { return operand(3); }
public:  // utils function
  static bool classof(const Value* v) { return v->valueId() == vMEMSET; }
  void print(std::ostream& os) const override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    // return utils::make<MemsetInst>(mType, getValue(operand(0)), mSize);
    return nullptr;
  }
};

/*
 * @brief GetElementPtr Instruction
 * @details:
 *      数组: <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32
 * <idx> 指针: <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
 * @param:
 *      1. mIdx: 数组各个维度的下标索引
 *      2. _id : calculate array address OR pointer address
 */
class GetElementPtrInst : public Instruction {
protected:
  size_t _id = 0;
  std::vector<size_t> _cur_dims = {};

public:
  //! 1. Pointer <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
  GetElementPtrInst(Type* base_type, Value* value, Value* idx, BasicBlock* parent=nullptr)
    : Instruction(vGETELEMENTPTR, ir::Type::TypePointer(base_type), parent) {
    _id = 0;
    addOperand(value);
    addOperand(idx);
  }

  //! 2. 高维 Array <result> = getelementptr <type>, <type>* <ptrval>, i32 0,
  //! i32 <idx>
  GetElementPtrInst(Type* base_type, Value* value, Value* idx,
                    std::vector<size_t> dims,
                    std::vector<size_t> cur_dims,
                    BasicBlock* parent=nullptr)
    : Instruction(vGETELEMENTPTR,
                  ir::Type::TypePointer(ir::Type::TypeArray(base_type, dims)),
                  parent),
      _cur_dims(cur_dims) {
    _id = 1;
    addOperand(value);
    addOperand(idx);
  }

  //! 3. 一维 Array <result> = getelementptr <type>, <type>* <ptrval>, i32 0,
  //! i32 <idx>
  GetElementPtrInst(Type* base_type,
                    Value* value,
                    Value* idx,
                    std::vector<size_t> cur_dims,
                    BasicBlock* parent = nullptr)
    : Instruction(vGETELEMENTPTR, ir::Type::TypePointer(base_type), parent), _cur_dims(cur_dims) {
    _id = 2;
    addOperand(value);
    addOperand(idx);
  }
public:  // get function
  auto value() const { return operand(0); }
  auto index() const { return operand(1); }
  auto getid() const { return _id; }
  Type* baseType() const {
    assert(dyn_cast<PointerType>(type()) && "type error");
    return dyn_cast<PointerType>(type())->baseType();
  }
  auto cur_dims_cnt() const { return _cur_dims.size(); }
  auto cur_dims() const { return _cur_dims; }
public:  // check function
  bool is_arrayInst() const { return _id != 0; }
public:  // utils function
  static bool classof(const Value* v) { return v->valueId() == vGETELEMENTPTR; }
  void print(std::ostream& os) const override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    auto newvalue = getValue(value());
    auto newidx = getValue(index());
    if (getid() == 0) {
      return utils::make<GetElementPtrInst>(baseType(), newvalue, newidx);
    } else if (getid() == 1) {
      auto basetype = baseType()->dynCast<ArrayType>();
      auto dims = basetype->dims();
      auto curdims = cur_dims();
      return utils::make<GetElementPtrInst>(basetype->baseType(), newvalue, newidx, dims, curdims);
    } else {
      return utils::make<GetElementPtrInst>(baseType(), newvalue, newidx, cur_dims());
    }
  }
};

class PhiInst : public Instruction {
protected:
  size_t mSize;
  std::unordered_map<ir::BasicBlock*, ir::Value*> mbbToVal;

public:
  PhiInst(BasicBlock* parent,
          Type* type,
          const std::vector<Value*>& vals = {},
          const std::vector<BasicBlock*>& bbs = {})
    : Instruction(vPHI, type, parent), mSize(vals.size()) {
    assert(vals.size() == bbs.size() and "number of vals and bbs in phi must be equal!");
    for (size_t i = 0; i < mSize; i++) {
      addOperand(vals[i]);
      addOperand(bbs[i]);
      mbbToVal[bbs[i]] = vals[i];
    }
  }
  auto getValue(size_t k) const { return operand(2 * k); }
  auto getBlock(size_t k) const { return operand(2 * k + 1)->dynCast<BasicBlock>(); }
  Value* getvalfromBB(BasicBlock* bb);
  BasicBlock* getbbfromVal(Value* val);

  auto getsize() { return mSize; }
  void addIncoming(Value* val, BasicBlock* bb) {
    if (mbbToVal.count(bb)) {
      assert(false && "Trying to add a duplicated basic block!");
    }
    addOperand(val);
    addOperand(bb);
    // 更新操作数的数量
    mbbToVal[bb] = val;
    mSize++;
  }
  void delValue(Value* val);
  void delBlock(BasicBlock* bb);
  void replaceBlock(BasicBlock* newBB, size_t k);
  void replaceoldtonew(BasicBlock* oldBB, BasicBlock* newBB);
  void refreshMap();

  void print(std::ostream& os) const override;
  Value* getConstantRepl(bool recursive = false) override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override {
    return utils::make<PhiInst>(nullptr, mType);
  }
};

class FunctionPtrInst : public Instruction {
public:
  explicit FunctionPtrInst(Function* func, Type* dstType, BasicBlock* parent = nullptr)
    : Instruction(vFUNCPTR, dstType, parent) {
    addOperand(func);
  }
  Function* getFunc() const { return operand(0)->as<Function>(); }
  static bool classof(const Value* v) { return v->valueId() == vFUNCPTR; }
  void print(std::ostream& os) const override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override { return nullptr; }
};

/*
 * @brief: PtrCastInst
 * @note: 
 */
class PtrCastInst : public Instruction {
public:
  explicit PtrCastInst(Value* src, Type* dstType, BasicBlock* parent = nullptr)
    : Instruction(vPTRCAST, dstType, parent) {
    addOperand(src);
  }
public:  // utils function
  static bool classof(const Value* v) { return v->valueId() == vPTRCAST; }
  void print(std::ostream& os) const override;
  Instruction* copy(std::function<Value*(Value*)> getValue) const override { return nullptr; }
};

class indVar {
private:  // only for constant beginvar and stepvar
  ConstantInteger* mbeginVar;
  Value* mendVar;

  ConstantInteger* mstepVar;

  bool mendIsConst;

  BinaryInst* miterInst;
  Instruction* mcmpInst;
  PhiInst* mphiinst;

public:
  indVar(ConstantInteger* mbegin,
         Value* mend,
         ConstantInteger* mstep,
         BinaryInst* bininst,
         Instruction* cmpinst,
         PhiInst* phiinst)
    : mbeginVar(mbegin),
      mendVar(mend),
      mstepVar(mstep),
      miterInst(bininst),
      mcmpInst(cmpinst),
      mphiinst(phiinst) {
    mendIsConst = mendVar->isa<ConstantInteger>();
  }
  int getBeginI32() {
    if (not mbeginVar->isInt32())
      assert("BeginVar is not i32!");
    return mbeginVar->i32();
  }
  int getStepI32() {
    if (not mstepVar->isInt32())
      assert("StepVar is not i32!");
    return mstepVar->i32();
  }
  bool isEndVarConst() { return mendIsConst; }
  int getEndVarI32() {
    if (mendIsConst) {
      return mendVar->dynCast<ConstantValue>()->i32();
    } else
      assert(false && "endVar is not constant");
  }
  Value* endValue() { return mendVar; }
  BinaryInst* iterInst() { return miterInst; }
  Instruction* cmpInst() { return mcmpInst; }
  PhiInst* phiinst() { return mphiinst; }
  ConstantInteger* getBegin() { return mbeginVar; }
  ConstantInteger* getStep() { return mstepVar; }
  Value* getEnd() { return mendVar; }
  void print(std::ostream& os) const {
    os << "beginVar: ";
    mbeginVar->print(os);

    os << "\n endVar: ";
    mendVar->print(os);

    os << "\n stepVar: ";
    mstepVar->print(os);

    os << "\n iterInst: ";
    miterInst->print(os);

    os << "\n cmpInst: ";
    mcmpInst->print(os);

    os << "\n phiInst: ";
    mphiinst->print(os);

    os << "\n";
  }
};

}  // namespace ir
