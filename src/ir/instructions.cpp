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
    case vSITOFP:
      return "sitofp";
    case vFPTOSI:
      return "fptosi";
    case vTRUNC:
      return "trunc";
    case vZEXT:
      return "zext";
    case vSEXT:
      return "sext";
    case vFPTRUNC:
      return "fptrunc";
    case vFNEG:
      return "fneg";
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

Value* BinaryInst::getConstantRepl() {
  auto lval = lValue();
  auto rval = rValue();
  assert(lval->type() == rval->type());

  if (auto lvalInst = lval->dynCast<Instruction>()) {
    lval = lvalInst->getConstantRepl();
  }
  if (auto rvalInst = rval->dynCast<Instruction>()) {
    rval = rvalInst->getConstantRepl();
  }

  if (not(lval->isa<ConstantValue>() and rval->isa<ConstantValue>()))
    return nullptr;

  int32_t i32val = 0;
  float f32val = 0.0;

  auto clval = lval->dynCast<ConstantValue>();
  auto crval = rval->dynCast<ConstantValue>();

  switch (mValueId) {
    case vADD:
      i32val = clval->i32() + crval->i32();
      break;
    case vSUB:
      i32val = clval->i32() - crval->i32();
      break;
    case vMUL:
      i32val = clval->i32() * crval->i32();
      break;
    case vSDIV:
      i32val = clval->i32() / crval->i32();
      break;
    case vSREM:
      i32val = clval->i32() % crval->i32();
      break;
    case vFADD:
      f32val = clval->f32() + crval->f32();
      break;
    case vFSUB:
      f32val = clval->f32() - crval->f32();
      break;
    case vFMUL:
      f32val = clval->f32() * crval->f32();
      break;
    case vFDIV:
      f32val = clval->f32() / crval->f32();
      break;
    default:
      assert(false and "Error in BinaryInst::getConstantRepl!");
  }
  if (type()->isFloat32())
    return ConstantFloating::gen_f32(f32val);
  else
    return ConstantInteger::gen_i32(i32val);

  // if (lval->isInt32() and rval->isInt32()) {
  //   auto clval = lval->dynCast<ConstantInteger>();
  //   auto crval = rval->dynCast<ConstantInteger>();
  //   switch (valueId()) {
  //     case vADD:
  //       if (clval && crval)
  //         return ConstantInteger::gen_i32(clval->getVal() + crval->getVal());
  //       if (clval && clval->getVal() == 0)
  //         return rval;
  //       if (crval && crval->getVal() == 0)
  //         return lval;
  //       break;
  //     case vSUB:
  //       if (clval && crval)
  //         return ConstantInteger::gen_i32(clval->getVal() - crval->getVal());
  //       if (crval && crval->getVal() == 0)
  //         return lval;
  //       if (lval == rval)
  //         return ConstantInteger::gen_i32(0);
  //       break;
  //     case vMUL:
  //       if (clval and crval)
  //         return ConstantInteger::gen_i32(clval->getVal() * crval->getVal());
  //       if (clval and clval->getVal() == 1)
  //         return rval;
  //       if (crval and crval->getVal() == 1)
  //         return lval;
  //       if (clval and clval->getVal() == 0)
  //         return ConstantInteger::gen_i32(0);
  //       if (crval and crval->getVal() == 0)
  //         return ConstantInteger::gen_i32(0);
  //       break;
  //     case vSDIV:
  //       if (clval and crval)
  //         return ConstantInteger::gen_i32(clval->getVal() / crval->getVal());
  //       if (crval and crval->getVal() == 1)
  //         return lval;
  //       if (clval and clval->getVal() == 0)
  //         return ConstantInteger::gen_i32(0);
  //       if (lval == rval)
  //         return ConstantInteger::gen_i32(1);
  //       break;
  //     case vSREM:
  //       if (clval and crval)
  //         return ConstantInteger::gen_i32(clval->getVal() % crval->getVal());
  //       if (clval and clval->getVal() == 0)
  //         return ConstantInteger::gen_i32(0);
  //       break;
  //     default:
  //       assert(false and "Error in BinaryInst::getConstantRepl!");
  //       break;
  //   }
  //   return nullptr;
  // } else if (lval->isFloat32() and rval->isFloat32()) {
  //   auto clval = lval->dynCast<ConstantFloating>();
  //   auto crval = rval->dynCast<ConstantFloating>();
  //   switch (valueId()) {
  //     case vFADD:
  //       if (clval and crval)
  //         return ConstantFloating::gen_f32(clval->getVal() + crval->getVal());
  //       if (clval and clval->getVal() == 0)
  //         return rval;
  //       if (crval and crval->getVal() == 0)
  //         return lval;
  //       break;
  //     case vFSUB:
  //       if (clval and crval)
  //         return ConstantFloating::gen_f32(clval->getVal() - crval->getVal());
  //       if (crval and crval->getVal() == 0)
  //         return lval;
  //       if (lval == rval)
  //         return ConstantFloating::gen_f32(0);
  //       break;
  //     case vFMUL:
  //       if (clval and crval)
  //         return ConstantFloating::gen_f32(clval->getVal() * crval->getVal());
  //       if (clval and clval->getVal() == 1)
  //         return rval;
  //       if (crval and crval->getVal() == 1)
  //         return lval;
  //       if (clval and clval->getVal() == 0)
  //         return ConstantFloating::gen_f32(0);
  //       if (crval and crval->getVal() == 0)
  //         return ConstantFloating::gen_f32(0);
  //       break;
  //     case vFDIV:
  //       if (clval and crval)
  //         return ConstantFloating::gen_f32(clval->getVal() / crval->getVal());
  //       if (crval and crval->getVal() == 1)
  //         return lval;
  //       if (clval and clval->getVal() == 0)
  //         return ConstantFloating::gen_f32(0);
  //       if (lval == rval)
  //         return ConstantFloating::gen_f32(1);
  //       break;

  //     default:
  //       assert(false and "Error in BinaryInst::getConstantRepl!");
  //       break;
  //   }
  //   return nullptr;
  // }
  // return nullptr;
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

Value* UnaryInst::getConstantRepl() {
  auto val = value();
  if (auto valInst = val->dynCast<Instruction>())
    val = valInst->getConstantRepl();

  if (not val->isa<ConstantValue>())
    return nullptr;

  auto cval = val->dynCast<ConstantValue>();

  switch (valueId()) {
    case vSITOFP:
      return ConstantFloating::gen_f32(cval->i32());
    case vFPTOSI:
      return ConstantInteger::gen_i32(cval->f32());
    case vZEXT:
      return ConstantInteger::gen_i32(cval->i1());
    case vFNEG:
      return ConstantFloating::gen_f32(-cval->f32());
    default:
      std::cerr << mValueId << std::endl;
      assert(false && "Invalid scid from UnaryInst::getConstantRepl");
  }

  return nullptr;
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

Value* ICmpInst::getConstantRepl() {
  auto lval = lhs();
  auto rval = rhs();
  if (auto lhsInst = lval->dynCast<Instruction>())
    lval = lhsInst->getConstantRepl();
  if (auto rhsInst = rval->dynCast<Instruction>())
    rval = rhsInst->getConstantRepl();

  if (not(lval->isa<ConstantValue>() and rval->isa<ConstantValue>()))
    return nullptr;

  auto clval = lval->dynCast<ConstantValue>()->i32();
  auto crval = rval->dynCast<ConstantValue>()->i32();
  switch (valueId()) {
    case vIEQ:
      return ConstantInteger::gen_i1(clval == crval);
    case vINE:
      return ConstantInteger::gen_i1(clval != crval);
    case vISGT:
      return ConstantInteger::gen_i1(clval > crval);
    case vISLT:
      return ConstantInteger::gen_i1(clval < crval);
    case vISGE:
      return ConstantInteger::gen_i1(clval >= crval);
    case vISLE:
      return ConstantInteger::gen_i1(clval <= crval);
    default:
      assert(false and "icmpinst const flod error");
  }
  return nullptr;
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

Value* FCmpInst::getConstantRepl() {
  auto lval = lhs();
  auto rval = rhs();
  if (auto lhsInst = lval->dynCast<Instruction>())
    lval = lhsInst->getConstantRepl();
  if (auto rhsInst = rval->dynCast<Instruction>())
    rval = rhsInst->getConstantRepl();

  if (not(lval->isa<ConstantValue>() and rval->isa<ConstantValue>()))
    return nullptr;

  auto clval = lval->dynCast<ConstantValue>()->f32();
  auto crval = rval->dynCast<ConstantValue>()->f32();
  switch (valueId()) {
    case vFOEQ:
      return ConstantInteger::gen_i1(clval == crval);
    case vFONE:
      return ConstantInteger::gen_i1(clval != crval);
    case vFOGT:
      return ConstantInteger::gen_i1(clval > crval);
    case vFOLT:
      return ConstantInteger::gen_i1(clval < crval);
    case vFOGE:
      return ConstantInteger::gen_i1(clval >= crval);
    case vFOLE:
      return ConstantInteger::gen_i1(clval <= crval);
    default:
      assert(false and "icmpinst const flod error");
  }
  return nullptr;
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

Value* PhiInst::getConstantRepl() {
  if (mSize == 0)
    return nullptr;
  auto curval = getValue(0);
  if (mSize == 1)
    return getValue(0);
  for (size_t i = 1; i < mSize; i++) {
    if (getValue(i) != curval)
      return nullptr;
  }
  return curval;
}

void PhiInst::refreshMap() {
  mbbToVal.clear();
  for (size_t i = 0; i < mSize; i++) {
    mbbToVal[getBlock(i)] = getValue(i);
  }
}

/*
 * @brief: BitcastInst::print
 * @details:
 *      <result> = bitcast <ty> <value> to i8*
 */

void BitCastInst::print(std::ostream& os) const {
  os << name() << " = bitcast ";
  os << *type() << " ";
  value()->dumpAsOpernd(os);
  os << " to i8*";
}

void BitCastInstBeta::print(std::ostream& os) const {
  os << name() << " = bitcast ";
  os << *(value()->type()) << " ";
  value()->dumpAsOpernd(os);
  os << " to " << *mType;
  // os << " to i8*";
}

/*
 * @brief: memset
 * @details:
 *      call void @llvm.memset.p0i8.i64(i8* <dest>, i8 0, i64 <len>, i1 false)
 */
void MemsetInst::print(std::ostream& os) const {
  // assert(dyn_cast<PointerType>(type()) && "type error");
  os << "call void @llvm.memset.p0i8.i64(i8* ";
  value()->dumpAsOpernd(os);
  os << ", i8 0, i64 " << dyn_cast<PointerType>(type())->baseType()->size() << ", i1 false)";

  // os << "call void @llvm.memset.p0i8.i64(" <<
}

void MemsetInstBeta::print(std::ostream& os) const {
  os << "call void @llvm.memset.p0i8.i64(";
  os << *(operand(0)->type()) << " ";
  operand(0)->dumpAsOpernd(os);
  os << ", ";

  os << *(operand(1)->type()) << " ";
  operand(0)->dumpAsOpernd(os);
  os << ", ";

  os << *(operand(2)->type()) << " ";
  operand(0)->dumpAsOpernd(os);
  os << ")";
}
/**
 *
 */
void FunctionPtrInst::print(std::ostream& os) const {}

void PtrCastInst::print(std::ostream& os) const {}

}  // namespace ir
