#pragma once
#include "ir/function.hpp"
#include "ir/global.hpp"
#include "ir/infrast.hpp"
#include "ir/value.hpp"
#include "support/utils.hpp"

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
      : Instruction(vALLOCA, ir::Type::TypePointer(base_type), parent, name),
        mIsConst(is_const) {}

  //! 2. Alloca Array
  AllocaInst(Type* base_type,
             std::vector<size_t> dims,
             BasicBlock* parent = nullptr,
             const_str_ref name = "",
             bool is_const = false,
             size_t capacity = 1)
      : Instruction(vALLOCA,
                    ir::Type::TypePointer(
                        ir::Type::TypeArray(base_type, dims, capacity)),
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
};

/*
 * @brief Return Instruction
 * @details:
 *      ret <type> <value>
 *      ret void
 */
class ReturnInst : public Instruction {
 public:
  ReturnInst(Value* value = nullptr,
             BasicBlock* parent = nullptr,
             const_str_ref name = "")
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
};

/*
 * @brief Unary Instruction
 * @details:
 *      <result> = sitofp <ty> <value> to <ty2>
 *      <result> = fptosi <ty> <value> to <ty2>
 *
 *      <result> = fneg [fast-math flags]* <ty> <op1>
 */
class UnaryInst : public Instruction {
 public:
  UnaryInst(ValueId kind,
            Type* type,
            Value* operand,
            BasicBlock* parent = nullptr,
            const_str_ref name = "")
      : Instruction(kind, type, parent, name) {
    addOperand(operand);
  }

 public:
  static bool classof(const Value* v) {
    return v->valueId() >= vUNARY_BEGIN && v->valueId() <= vUNARY_END;
  }

 public:
  auto value() const { return operand(0); }

 public:
  void print(std::ostream& os) const override;
  Value* getConstantRepl() override;
};

/*
 * @brief Binary Instruction
 * @details:
 *      1. exp (MUL | DIV | MODULO) exp
 *      2. exp (ADD | SUB) exp
 */
class BinaryInst : public Instruction {
 public:
  BinaryInst(ValueId kind,
             Type* type,
             Value* lvalue,
             Value* rvalue,
             BasicBlock* parent = nullptr,
             const std::string name = "")
      : Instruction(kind, type, parent, name) {
    addOperand(lvalue);
    addOperand(rvalue);
  }

 public:
  static bool classof(const Value* v) {
    return v->valueId() >= vBINARY_BEGIN && v->valueId() <= vBINARY_END;
  }
  bool isCommutative() const {
    return valueId() == vADD || valueId() == vFADD || valueId() == vMUL ||
           valueId() == vFMUL;
  }

 public:
  Value* lValue() const { return operand(0); }
  Value* rValue() const { return operand(1); }

 public:
  void print(std::ostream& os) const override;
  Value* getConstantRepl() override;
};
/* CallInst */
class CallInst : public Instruction {
  Function* mCallee = nullptr;

 public:
  CallInst(Function* callee,
           const_value_ptr_vector rargs = {},
           BasicBlock* parent = nullptr,
           const_str_ref name = "")
      : Instruction(vCALL, callee->retType(), parent, name), mCallee(callee) {
    addOperands(rargs);
  }

  Function* callee() const { return mCallee; }
  /* real arguments */
  auto& rargs() const { return mOperands; }
  static bool classof(const Value* v) { return v->valueId() == vCALL; }
  void print(std::ostream& os) const override;
};

/*
 * @brief: Conditional or Unconditional Branch Instruction
 * @note:
 *      1. br i1 <cond>, label <iftrue>, label <iffalse>
 *      2. br label <dest>
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
  BranchInst(BasicBlock* dest, BasicBlock* parent=nullptr,
             const_str_ref name="")
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
  void replaceDest(ir::BasicBlock* olddest, ir::BasicBlock* newdest);
  void set_iftrue(BasicBlock* bb){
    assert(mIsCond and "not a conditional branch");  
    setOperand(1,bb);
  }
  void set_iffalse(BasicBlock* bb){
    assert(mIsCond and "not a conditional branch");
    setOperand(2,bb);
  }
  void set_dest(BasicBlock* bb){
    assert(not mIsCond and "not an unconditional branch");
    setOperand(0,bb);
  }

 public:
  static bool classof(const Value* v) { return v->valueId() == vBR; }
  void print(std::ostream& os) const override;
};

/*
 * @brief: ICmpInst
 * @note: <result> = icmp <cond> <ty> <op1>, <op2>
 */
class ICmpInst : public Instruction {
 public:
  ICmpInst(ValueId itype,
           Value* lhs,
           Value* rhs,
           BasicBlock* parent = nullptr,
           const_str_ref name = "")
      : Instruction(itype, Type::TypeBool(), parent, name) {  // cmp return i1
    addOperand(lhs);
    addOperand(rhs);
  }

 public:
  auto lhs() const { return operand(0); }
  auto rhs() const { return operand(1); }

  bool isReverse(ICmpInst* y);

  static bool classof(const Value* v) {
    return v->valueId() >= vICMP_BEGIN && v->valueId() <= vICMP_END;
  }
  void print(std::ostream& os) const override;
  Value* getConstantRepl() override;
};

/* FCmpInst */
class FCmpInst : public Instruction {
 public:
  FCmpInst(ValueId itype,
           Value* lhs,
           Value* rhs,
           BasicBlock* parent = nullptr,
           const_str_ref name = "")
      : Instruction(itype,
                    Type::TypeBool(),  // also return i1
                    parent,
                    name) {
    addOperand(lhs);
    addOperand(rhs);
  }

 public:
  auto lhs() const { return operand(0); }
  auto rhs() const { return operand(1); }

  bool isReverse(FCmpInst* y);

  static bool classof(const Value* v) {
    return v->valueId() >= vFCMP_BEGIN && v->valueId() <= vFCMP_END;
  }
  void print(std::ostream& os) const override;
  Value* getConstantRepl() override;
};

/*
 * @brief Bitcast Instruction
 * @note: <result> = bitcast <ty> <value> to i8*
 */
class BitCastInst : public Instruction {
 public:
  BitCastInst(Type* type, Value* value, BasicBlock* parent = nullptr)
      : Instruction(vBITCAST, type, parent) {
    addOperand(value);
  }

 public:
  auto value() const { return operand(0); }

  static bool classof(const Value* v) { return v->valueId() == vBITCAST; }
  void print(std::ostream& os) const override;
};

/*
 * @brief: memset
 * @details:
 *      call void @llvm.memset.inline.p0.p0.i64(i8* <dest>, i8 0, i64 <len>, i1
 * <isvolatile>)
 */
class MemsetInst : public Instruction {
public:
  MemsetInst(Type* type, Value* value, BasicBlock* parent=nullptr)
      : Instruction(vMEMSET, type, parent) {
    addOperand(value);
  }
public:  // get function
  auto value() const { return operand(0); }
public:  // utils function
  static bool classof(const Value* v) { return v->valueId() == vMEMSET; }
  void print(std::ostream& os) const override;
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
  GetElementPtrInst(Type* base_type, Value* value, Value* idx,
                    BasicBlock* parent=nullptr)
      : Instruction(vGETELEMENTPTR, ir::Type::TypePointer(base_type), parent) {
    _id = 0;
    addOperand(value);
    addOperand(idx);
  }

  //! 2. 高维 Array <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32 <idx>
  GetElementPtrInst(Type* base_type, Value* value, Value* idx,
                    std::vector<size_t> dims,
                    std::vector<size_t> cur_dims,
                    BasicBlock* parent=nullptr)
      : Instruction(vGETELEMENTPTR, ir::Type::TypePointer(ir::Type::TypeArray(base_type, dims)), parent),
        _cur_dims(cur_dims) {
    _id = 1;
    addOperand(value);
    addOperand(idx);
  }

  //! 3. 一维 Array <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32 <idx>
  GetElementPtrInst(Type* base_type, Value* value, Value* idx,
                    std::vector<size_t> cur_dims,
                    BasicBlock* parent=nullptr)
      : Instruction(vGETELEMENTPTR, ir::Type::TypePointer(base_type), parent),
        _cur_dims(cur_dims) {
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
  auto& cur_dims() const { return _cur_dims; }
public:  // check function
  bool is_arrayInst() const { return _id != 0; }
public:
  static bool classof(const Value* v) { return v->valueId() == vGETELEMENTPTR; }
  void print(std::ostream& os) const override;
};

class PhiInst : public Instruction {
 protected:
  size_t mSize;
  std::unordered_map<ir::BasicBlock*,ir::Value*>mbbToVal;


 public:
  PhiInst(BasicBlock* bb,
          Type* type,
          const std::vector<Value*>& vals = {},
          const std::vector<BasicBlock*>& bbs = {})
      : Instruction(vPHI, type, bb), mSize(vals.size()) {
    assert(vals.size() == bbs.size() and
           "number of vals and bbs in phi must be equal!");
    for (size_t i = 0; i < mSize; i++) {
      addOperand(vals[i]);
      addOperand(bbs[i]);
      mbbToVal[bbs[i]]=vals[i];
    }
  }
  auto getValue(size_t k) const { return operand(2 * k); }
  auto getBlock(size_t k) const {
    return operand(2 * k + 1)->dynCast<BasicBlock>();
  }
  Value* getvalfromBB(BasicBlock* bb);
  BasicBlock* getbbfromVal(Value* val);

  auto getsize() { return mSize; }
  void addIncoming(Value* val, BasicBlock* bb) {
    if(mbbToVal.count(bb)){
      assert(false && "Trying to add a duplicated basic block!");
    }
    addOperand(val);
    addOperand(bb);
    // 更新操作数的数量
    mbbToVal[bb]=val;
    mSize++;
  }
  void delValue(Value* val);
  void delBlock(BasicBlock* bb);
  void replaceBlock(BasicBlock* newBB, size_t k);
  void replaceoldtonew(BasicBlock* oldBB, BasicBlock* newBB);
  void refreshMap();

  void print(std::ostream& os) const override;
  Value* getConstantRepl() override;
};

class indVar{
  private:// only for constant beginvar and stepvar
    Constant* mbeginVar;
    Value* mendVar;
    Constant* mstepVar;
    bool mendIsConst;
    BinaryInst* miterInst;
    Instruction* mcmpInst;
  public:
    indVar(Constant* mbegin,Value* mend,Constant* mstep, BinaryInst* bininst, Instruction* cmpinst):
      mbeginVar(mbegin),mendVar(mend),mstepVar(mstep),miterInst(bininst),mcmpInst(cmpinst){
        mendIsConst=dyn_cast<Constant>(mendVar)!=nullptr;
      }
    int getBeginI32(){
      if(not mbeginVar->isInt32())assert("BeginVar is not i32!");
      return mbeginVar->i32();
    }
    int getStepI32(){
      if(not mstepVar->isInt32())assert("StepVar is not i32!");
      return mstepVar->i32();
    }
    bool isEndVarConst(){return mendIsConst;}
    int getEndVar(){
      if(mendIsConst){
        auto mendConst=dyn_cast<Constant>(mendVar);
        return mendConst->i32();
      }
      else
        assert(false && "endVar is not constant");
    }

};

}  // namespace ir

