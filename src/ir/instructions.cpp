#include "ir/instructions.hpp"

#include "ir/utils_ir.hpp"
#include "ir/ConstantValue.hpp"

namespace ir {
//! Value <- User <- Instruction <- XxxInst

static const std::string getInstName(ValueId instID) {
  switch (instID) {
    case vADD:
      return "add";
    case vFADD:
      return "fadd";
    case vSUB:
      return "sub";
    case vFSUB:
      return "fsub";
    case vMUL:
      return "mul";
    case vFMUL:
      return "fmul";
    case vSDIV:
      return "sdiv";
    case vFDIV:
      return "fdiv";
    case vSREM:
      return "srem";
    case vFREM:
      return "frem";
    case vALLOCA:
      return "alloca";
    case vLOAD:
      return "load";
    case vSTORE:
      return "store";
    case vPHI:
      return "phi";
    case vRETURN:
      return "ret";
    case vBR:
      return "br";
    // unary
    case vFNEG:
      return "fneg";
    case vTRUNC:
      return "trunc";
    case vZEXT:
      return "zext";
    case vSEXT:
      return "sext";
    case vFPTRUNC:
      return "fptrunc";
    case vFPTOSI:
      return "fptosi";
    case vSITOFP:
      return "sitofp";
    case vBITCAST:
      return "bitcast";
    // cmp
    case vIEQ:
      return "icmp eq";
    case vINE:
      return "icmp ne";
    case vISGT:
      return "icmp sgt";
    case vISGE:
      return "icmp sge";
    case vISLT:
      return "icmp slt";
    case vISLE:
      return "icmp sle";
    case vFOEQ:
      return "oeq";
    case vFONE:
      return "one";
    case vFOGT:
      return "ogt";
    case vFOGE:
      return "oge";
    case vFOLT:
      return "olt";
    case vFOLE:
      return "ole";
    case vGETELEMENTPTR:
      return "getelementptr";
    default:
      std::cerr << "Error: Unknown instruction ID: " << instID << std::endl;
      assert(false);
  }
}

/*
 * @brief AllocaInst::print
 * @details: alloca <ty>
 */
void AllocaInst::print(std::ostream& os) const {
  dumpAsOpernd(os);
  os << " = " << getInstName(mValueId) << " " << *(baseType());
  if (not mComment.empty()) {
    os << " ; " << mComment << "*";
  }
}

/*
 * @brief StoreInst::print
 * @details:
 *      store <ty> <value>, ptr <pointer>
 * @note:
 *      ptr: ArrayType or PointerType
 */
void StoreInst::print(std::ostream& os) const {
  os << getInstName(mValueId) << " ";
  os << *(value()->type()) << " ";
  value()->dumpAsOpernd(os);
  os << ", ";

  if (ptr()->type()->isPointer())
    os << *(ptr()->type()) << " ";
  else {
    auto ptype = ptr()->type();
    auto atype = ptype->dynCast<ArrayType>();
    os << *(atype->baseType()) << "* ";
  }
  ptr()->dumpAsOpernd(os);
}

void LoadInst::print(std::ostream& os) const {
  // os << name() << " = load ";
  dumpAsOpernd(os);
  os << " = " << getInstName(mValueId) << " ";
  auto ptype = ptr()->type();
  if (ptype->isPointer()) {
    os << *dyn_cast<PointerType>(ptype)->baseType() << ", ";
    os << *ptype << " ";
  } else {
    os << *dyn_cast<ArrayType>(ptype)->baseType() << ", ";
    os << *dyn_cast<ArrayType>(ptype)->baseType() << "* ";
  }
  ptr()->dumpAsOpernd(os);

  // comment
  if (not mComment.empty()) {
    os << " ; " << getInstName(mValueId) << " " << mComment;
  }
}

void ReturnInst::print(std::ostream& os) const {
  os << getInstName(mValueId) << " ";
  auto ret = returnValue();
  if (ret) {
    os << *ret->type() << " ";
    ret->dumpAsOpernd(os);
  } else {
    os << "void";
  }
}

/*
 * @brief Binary Instruction Output
 * @note: <result> = add <ty> <op1>, <op2>
 */
void BinaryInst::print(std::ostream& os) const {
  dumpAsOpernd(os);
  os << " = ";
  os << getInstName(mValueId) << " ";
  // <type>
  os << *type() << " ";

  // <op1>
  lValue()->dumpAsOpernd(os);

  os << ", ";

  // <op2>
  rValue()->dumpAsOpernd(os);

  /* comment */
  if (not lValue()->comment().empty() && not rValue()->comment().empty()) {
    os << " ; " << lValue()->comment();
    os << " " << getInstName(mValueId) << " ";
    os << rValue()->comment();
  }
}

/*
 * @brief: Unary Instruction Output
 * @note:
 *      <result> = sitofp <ty> <value> to <ty2>
 *      <result> = fptosi <ty> <value> to <ty2>
 *      <result> = fneg [fast-math flags]* <ty> <op1>
 */
void UnaryInst::print(std::ostream& os) const {
  dumpAsOpernd(os);
  os << " = ";
  os << getInstName(mValueId) << " ";
  os << *(value()->type()) << " ";
  value()->dumpAsOpernd(os);

  if (mValueId != vFNEG) {
    os << " to " << *type();
  }

  /* comment */
  if (not value()->comment().empty()) {
    os << " ; " << "uop " << value()->comment();
  }
}

bool ICmpInst::isReverse(ICmpInst* y) {
  auto x = this;
  if ((x->valueId() == vISGE && y->valueId() == vISLE) ||
      (x->valueId() == vISLE && y->valueId() == vISGE)) {
    return true;
  } else if ((x->valueId() == vISGT && y->valueId() == vISLT) ||
             (x->valueId() == vISLT && y->valueId() == vISGT)) {
    return true;
  } else if ((x->valueId() == vFOGE && y->valueId() == vFOLE) ||
             (x->valueId() == vFOLE && y->valueId() == vFOGE)) {
    return true;
  } else if ((x->valueId() == vFOGT && y->valueId() == vFOLT) ||
             (x->valueId() == vFOLT && y->valueId() == vFOGT)) {
    return true;
  } else {
    return false;
  }
}
void ICmpInst::print(std::ostream& os) const {
  // <result> = icmp <cond> <ty> <op1>, <op2>   ; yields i1 or <N x i1>:result
  // %res = icmp eq i32, 1, 2
  dumpAsOpernd(os);
  os << " = ";
  os << getInstName(mValueId) << " ";
  // type
  os << *lhs()->type() << " ";
  // op1
  lhs()->dumpAsOpernd(os);
  os << ", ";
  // op2
  rhs()->dumpAsOpernd(os);
  os << " ";
  /* comment */
  if (not lhs()->comment().empty() && not rhs()->comment().empty()) {
    os << " ; " << lhs()->comment() << " " << getInstName(mValueId) << " " << rhs()->comment();
  }
}

bool FCmpInst::isReverse(FCmpInst* y) {
  auto x = this;
  if ((x->valueId() == vISGE && y->valueId() == vISLE) ||
      (x->valueId() == vISLE && y->valueId() == vISGE)) {
    return true;
  } else if ((x->valueId() == vISGT && y->valueId() == vISLT) ||
             (x->valueId() == vISLT && y->valueId() == vISGT)) {
    return true;
  } else if ((x->valueId() == vFOGE && y->valueId() == vFOLE) ||
             (x->valueId() == vFOLE && y->valueId() == vFOGE)) {
    return true;
  } else if ((x->valueId() == vFOGT && y->valueId() == vFOLT) ||
             (x->valueId() == vFOLT && y->valueId() == vFOGT)) {
    return true;
  } else {
    return false;
  }
}
void FCmpInst::print(std::ostream& os) const {
  // <result> = icmp <cond> <ty> <op1>, <op2>   ; yields i1 or <N x i1>:result
  // %res = icmp eq i32, 1, 2
  dumpAsOpernd(os);

  os << " = ";
  os << "fcmp ";
  // cond code
  os << getInstName(mValueId) << " ";
  // type
  os << *lhs()->type() << " ";
  // op1
  lhs()->dumpAsOpernd(os);
  os << ", ";
  // op2
  rhs()->dumpAsOpernd(os);
  /* comment */
  if (not lhs()->comment().empty() && not rhs()->comment().empty()) {
    os << " ; " << lhs()->comment() << " " << getInstName(mValueId) << " " << rhs()->comment();
  }
}

void BranchInst::replaceDest(ir::BasicBlock* olddest, ir::BasicBlock* newdest) {
  if (mIsCond) {
    if (iftrue() == olddest) {
      setOperand(1, newdest);
    } else if (iffalse() == olddest) {
      setOperand(2, newdest);
    } else {
      assert(false and "branch inst replaceDest error");
    }
  } else {
    setOperand(0, newdest);
  }
}
/*
 * @brief: BranchInst::print
 * @details:
 *      br i1 <cond>, label <iftrue>, label <iffalse>
 *      br label <dest>
 */
void BranchInst::print(std::ostream& os) const {
  os << getInstName(mValueId) << " ";
  if (is_cond()) {
    os << "i1 ";
    cond()->dumpAsOpernd(os);
    os << ", ";
    os << "label %" << iftrue()->name() << ", ";
    os << "label %" << iffalse()->name();
    /* comment */
    if (not iftrue()->comment().empty() && not iffalse()->comment().empty()) {
      os << " ; " << "br " << iftrue()->comment() << ", " << iffalse()->comment();
    }

  } else {
    os << "label %" << dest()->name();
    /* comment */
    if (not dest()->comment().empty()) {
      os << " ; " << "br " << dest()->comment();
    }
  }
}

/*
 * @brief: GetElementPtrInst::print
 * @details:
 *      数组: <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32
 * <idx> 指针: <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
 */
void GetElementPtrInst::print(std::ostream& os) const {
  if (is_arrayInst()) {
    dumpAsOpernd(os);
    os << " = " << getInstName(mValueId) << " ";

    const auto dimensions = cur_dims_cnt();
    for (size_t i = 0; i < dimensions; i++) {
      size_t value = cur_dims()[i];
      os << "[" << value << " x ";
    }
    if (_id == 1)
      os << *(dyn_cast<ir::ArrayType>(baseType())->baseType());
    else
      os << *(baseType());
    for (size_t i = 0; i < dimensions; i++)
      os << "]";
    os << ", ";

    for (size_t i = 0; i < dimensions; i++) {
      size_t value = cur_dims()[i];
      os << "[" << value << " x ";
    }
    if (_id == 1)
      os << *(dyn_cast<ir::ArrayType>(baseType())->baseType());
    else
      os << *(baseType());
    for (size_t i = 0; i < dimensions; i++)
      os << "]";
    os << "* ";

    value()->dumpAsOpernd(os);
    os << ", ";
    os << "i32 0, i32 ";
    index()->dumpAsOpernd(os);
  } else {
    dumpAsOpernd(os);
    os << " = " << getInstName(mValueId) << " ";
    os << *(baseType()) << ", " << *type() << " ";
    value()->dumpAsOpernd(os);
    os << ", " << *(index()->type()) << " ";
    index()->dumpAsOpernd(os);
  }
}

void CallInst::print(std::ostream& os) const {
  if (callee()->retType()->isVoid()) {
    if (mIsTail)
      os << "tail ";
    os << "call ";
  } else {
    os << name() << " = ";
    if (mIsTail)
      os << "tail ";
    os << "call ";
  }

  // retType
  os << *type() << " ";
  // func name
  os << "@" << callee()->name() << "(";

  if (operands().size() > 0) {
    // Iterator pointing to the last element
    auto last = operands().end() - 1;
    for (auto it = operands().begin(); it != last; ++it) {
      // it is a iterator, *it get the element in operands,
      // which is the Value* ptr
      auto val = (*it)->value();
      os << *(val->type()) << " ";
      // os << val->name();
      val->dumpAsOpernd(os);
      os << ", ";
    }
    auto lastval = (*last)->value();
    os << *(lastval->type()) << " ";
    // os << lastval->name();
    lastval->dumpAsOpernd(os);
  }
  os << ")";
}

void PhiInst::print(std::ostream& os) const {
  // 在打印的时候对其vals和bbs进行更新
  os << name() << " = ";
  os << "phi " << *(type()) << " ";
  // for all vals, bbs
  for (size_t i = 0; i < mSize; i++) {
    os << "[ ";
    getValue(i)->dumpAsOpernd(os);
    os << ", %" << getBlock(i)->name() << " ]";
    if (i != mSize - 1)
      os << ",";
  }
}

BasicBlock* PhiInst::getbbfromVal(Value* val) {
  //! only return first matched val
  for (size_t i = 0; i < mSize; i++) {
    if (getValue(i) == val)
      return getBlock(i);
  }
  // assert(false && "can't find match basic block!");
  return nullptr;
}

Value* PhiInst::getvalfromBB(BasicBlock* bb) {
  refreshMap();
  if (mbbToVal.count(bb))
    return mbbToVal[bb];
  // assert(false && "can't find match value!");
  return nullptr;
}

void PhiInst::delValue(Value* val) {
  //! only delete the first matched value
  size_t i;
  auto bb = getbbfromVal(val);
  for (i = 0; i < mSize; i++) {
    if (getValue(i) == val)
      break;
  }
  delete_operands(2 * i);
  delete_operands(2 * i);
  mSize--;
  mbbToVal.erase(bb);
}

void PhiInst::delBlock(BasicBlock* bb) {
  if (not mbbToVal.count(bb))
    assert(false and "can't find bb incoming!");
  size_t i;
  for (i = 0; i < mSize; i++) {
    if (getBlock(i) == bb)
      break;
  }
  delete_operands(2 * i);
  delete_operands(2 * i);
  mSize--;
  mbbToVal.erase(bb);
}

void PhiInst::replaceBlock(BasicBlock* newBB, size_t k) {
  assert(k < mSize);
  refreshMap();
  auto val = mbbToVal[getBlock(k)];
  mbbToVal.erase(getBlock(k));
  setOperand(2 * k + 1, newBB);
  mbbToVal[newBB] = val;
}

void PhiInst::replaceoldtonew(BasicBlock* oldbb, BasicBlock* newbb) {
  refreshMap();
  assert(mbbToVal.count(oldbb));
  auto val = mbbToVal[oldbb];
  delBlock(oldbb);
  addIncoming(val, newbb);
}

void PhiInst::refreshMap() {
  mbbToVal.clear();
  for (size_t i = 0; i < mSize; i++) {
    mbbToVal[getBlock(i)] = getValue(i);
  }
}

/*
 * @brief: memset
 * @details:
 *      call void @llvm.memset.p0i8.i64(i8* <dest>, i8 0, i64 <len>, i1 false)
 */
void MemsetInst::print(std::ostream& os) const {
  os << "call void @llvm.memset.p0i8.i64(";
  os << *(dst()->type()) << " ";
  dst()->dumpAsOpernd(os);
  os << ", " << *(val()->type()) << " ";
  val()->dumpAsOpernd(os);
  os << ", " << *(len()->type()) << " ";
  len()->dumpAsOpernd(os);
  os << ", " << *(isVolatile()->type()) << " ";
  isVolatile()->dumpAsOpernd(os);
  os << ")";
}
/**
 *
 */
void FunctionPtrInst::print(std::ostream& os) const {}

void PtrCastInst::print(std::ostream& os) const {}

}  // namespace ir
