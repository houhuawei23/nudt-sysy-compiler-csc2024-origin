#include "ir/instructions.hpp"
#include "ir/constant.hpp"
#include "ir/utils_ir.hpp"

namespace ir {
//! Value <- User <- Instruction <- XxxInst

/*
 * @brief AllocaInst::print
 * @details:
 *      alloca <ty>
 */
void AllocaInst::print(std::ostream& os) const {
  os << name() << " = alloca " << *(baseType());
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
  os << "store ";
  os << *(value()->type()) << " ";
  if (ir::isa<ir::Constant>(value())) {
    auto v = dyn_cast<ir::Constant>(value());
    os << *(value()) << ", ";
  } else {
    os << value()->name() << ", ";
  }
  if (ptr()->type()->isPointer())
    os << *(ptr()->type()) << " ";
  else {
    auto ptype = ptr()->type();
    ArrayType* atype = dyn_cast<ArrayType>(ptype);
    os << *(atype->baseType()) << "* ";
  }
  os << ptr()->name();
}

void LoadInst::print(std::ostream& os) const {
  os << name() << " = load ";
  auto ptype = ptr()->type();
  if (ptype->isPointer()) {
    os << *dyn_cast<PointerType>(ptype)->baseType() << ", ";
    os << *ptype << " ";
  } else {
    os << *dyn_cast<ArrayType>(ptype)->baseType() << ", ";
    os << *dyn_cast<ArrayType>(ptype)->baseType() << "* ";
  }
  os << ptr()->name();

  // comment
  if (not mComment.empty()) {
    os << " ; " << "load " << mComment;
  }
}

void ReturnInst::print(std::ostream& os) const {
  os << "ret ";
  auto ret = returnValue();
  if (ret) {
    os << *ret->type() << " ";
    if (ir::isa<ir::Constant>(ret))
      os << *(ret);
    else
      os << ret->name();
  } else {
    os << "void";
  }
}

/*
 * @brief Binary Instruction Output
 *      <result> = add <ty> <op1>, <op2>
 */
void BinaryInst::print(std::ostream& os) const {
  os << name() << " = ";
  auto opstr = [scid = valueId()] {
    switch (scid) {
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
      default:
        return "unknown";
    }
  }();
  os << opstr << " ";
  // <type>
  os << *type() << " ";
  // <op1>
  if (ir::isa<ir::Constant>(lValue()))
    os << *(lValue()) << ", ";
  else
    os << lValue()->name() << ", ";
  // <op2>
  if (ir::isa<ir::Constant>(rValue()))
    os << *(rValue());
  else
    os << rValue()->name();

  /* comment */
  if (not lValue()->comment().empty() && not rValue()->comment().empty()) {
    os << " ; " << lValue()->comment();
    os << " " << opstr << " ";
    os << rValue()->comment();
  }
}

Value* BinaryInst::getConstantRepl() {
  auto lval = lValue();
  auto rval = rValue();
  auto clval = dyn_cast<ir::Constant>(lval);
  auto crval = dyn_cast<ir::Constant>(rval);
  if (lval->isInt32() and rval->isInt32()) {
    switch (valueId()) {
      case vADD:
        if (clval and crval)
          return Constant::gen_i32(clval->i32() + crval->i32());
        if (clval and clval->i32() == 0)
          return rval;
        if (crval and crval->i32() == 0)
          return lval;
        break;
      case vSUB:
        if (clval and crval)
          return Constant::gen_i32(clval->i32() - crval->i32());
        if (crval and crval->i32() == 0)
          return lval;
        if (lval == rval)
          return Constant::gen_i32(0);
        break;
      case vMUL:
        if (clval and crval)
          return Constant::gen_i32(clval->i32() * crval->i32());
        if (clval and clval->i32() == 1)
          return rval;
        if (crval and crval->i32() == 1)
          return lval;
        if (clval and clval->i32() == 0)
          return Constant::gen_i32(0);
        if (crval and crval->i32() == 0)
          return Constant::gen_i32(0);
        break;
      case vSDIV:
        if (clval and crval)
          return Constant::gen_i32(clval->i32() / crval->i32());
        if (crval and crval->i32() == 1)
          return lval;
        if (clval and clval->i32() == 0)
          return Constant::gen_i32(0);
        if (lval == rval)
          return Constant::gen_i32(1);
        break;
      case vSREM:
        if (clval and crval)
          return Constant::gen_i32(clval->i32() % crval->i32());
        if (clval and clval->i32() == 0)
          return Constant::gen_i32(0);
        break;
      default:
        assert(false and "Error in BinaryInst::getConstantRepl!");
        break;
    }
    return nullptr;
  } else if (lval->isFloat32() and rval->isFloat32()) {
    switch (valueId()) {
      case vFADD:
        if (clval and crval)
          return Constant::gen_f32(clval->f32() + crval->f32());
        if (clval and clval->f32() == 0)
          return rval;
        if (crval and crval->f32() == 0)
          return lval;
        break;
      case vFSUB:
        if (clval and crval)
          return Constant::gen_f32(clval->f32() - crval->f32());
        if (crval and crval->f32() == 0)
          return lval;
        if (lval == rval)
          return Constant::gen_f32(0);
        break;
      case vFMUL:
        if (clval and crval)
          return Constant::gen_f32(clval->f32() * crval->f32());
        if (clval and clval->f32() == 1)
          return rval;
        if (crval and crval->f32() == 1)
          return lval;
        if (clval and clval->f32() == 0)
          return Constant::gen_f32(0);
        if (crval and crval->f32() == 0)
          return Constant::gen_f32(0);
        break;
      case vFDIV:
        if (clval and crval)
          return Constant::gen_f32(clval->f32() / crval->f32());
        if (crval and crval->f32() == 1)
          return lval;
        if (clval and clval->f32() == 0)
          return Constant::gen_f32(0);
        if (lval == rval)
          return Constant::gen_f32(1);
        break;

      default:
        assert(false and "Error in BinaryInst::getConstantRepl!");
        break;
    }
    return nullptr;
  }
  return nullptr;
}

/*
 * @brief Unary Instruction Output
 *      <result> = sitofp <ty> <value> to <ty2>
 *      <result> = fptosi <ty> <value> to <ty2>
 *
 *      <result> = fneg [fast-math flags]* <ty> <op1>
 */
void UnaryInst::print(std::ostream& os) const {
  os << name() << " = ";
  if (valueId() == vFNEG) {
    os << "fneg " << *type() << " " << value()->name();
  } else {
    switch (valueId()) {
      case vSITOFP:
        os << "sitofp ";
        break;
      case vFPTOSI:
        os << "fptosi ";
        break;
      case vTRUNC:  // not used
        os << "trunc ";
        break;
      case vZEXT:
        os << "zext ";
        break;
      case vSEXT:  // not used
        os << "sext ";
        break;
      case vFPTRUNC:
        os << "fptrunc ";  // not used
        break;
      default:
        assert(false && "not valid scid");
        break;
    }
    os << *(value()->type()) << " ";
    os << value()->name();
    os << " to " << *type();
  }
  /* comment */
  if (not value()->comment().empty()) {
    os << " ; " << "uop " << value()->comment();
  }
}

Value* UnaryInst::getConstantRepl() {
  if (auto cval = value()->dynCast<Constant>()) {
    switch (valueId()) {
      case vSITOFP:
        return Constant::gen_f32(float(cval->i32()));
        break;
      case vFPTOSI:
        return Constant::gen_i32(int(cval->f32()));
        break;
      case vZEXT:
        return Constant::gen_i32(cval->i1());
        break;
      case vFNEG:
        return Constant::gen_f32(-float(cval->f32()));
      default:
        std::cerr<<valueId()<<std::endl;
        assert(false && "Invalid scid from UnaryInst::getConstantRepl");
        break;
    }
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
  os << name() << " = ";

  os << "icmp ";

  auto cmpstr = [scid = valueId()] {
    switch (scid) {
      case vIEQ:
        return "eq";
      case vINE:
        return "ne";
      case vISGT:
        return "sgt";
      case vISGE:
        return "sge";
      case vISLT:
        return "slt";
      case vISLE:
        return "sle";
      default:
        // assert(false && "unimplemented");
        std::cerr << "Error from ICmpInst::print(), wrong Inst Type!"
                  << std::endl;
        return "unknown";
    }
  }();

  // cond code
  os << cmpstr << " ";
  // type
  os << *lhs()->type() << " ";
  // op1
  os << lhs()->name() << ", ";
  // op2
  os << rhs()->name();
  /* comment */
  if (not lhs()->comment().empty() && not rhs()->comment().empty()) {
    os << " ; " << lhs()->comment() << " " << cmpstr << " " << rhs()->comment();
  }
}

Value* ICmpInst::getConstantRepl() {
  auto clhs = dyn_cast<Constant>(lhs());
  auto crhs = dyn_cast<Constant>(rhs());
  if (not clhs or not crhs)
    return nullptr;
  auto lhsval = clhs->i32();
  auto rhsval = crhs->i32();
  switch (valueId()) {
    case vIEQ:
      return Constant::gen_i1(lhsval == rhsval);
    case vINE:
      return Constant::gen_i1(lhsval != rhsval);
    case vISGT:
      return Constant::gen_i1(lhsval > rhsval);
    case vISLT:
      return Constant::gen_i1(lhsval < rhsval);
    case vISGE:
      return Constant::gen_i1(lhsval >= rhsval);
    case vISLE:
      return Constant::gen_i1(lhsval <= rhsval);
    default:
      assert(false and "icmpinst const flod error");
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
  os << name() << " = ";

  os << "fcmp ";
  // cond code
  auto cmpstr = [scid = valueId()] {
    switch (scid) {
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
      default:
        // assert(false && "unimplemented");
        std::cerr << "Error from FCmpInst::print(), wrong Inst Type!"
                  << std::endl;
        return "unknown";
    }
  }();
  os << cmpstr << " ";
  // type
  os << *lhs()->type() << " ";
  // op1
  os << lhs()->name() << ", ";
  // op2
  os << rhs()->name();
  /* comment */
  if (not lhs()->comment().empty() && not rhs()->comment().empty()) {
    os << " ; " << lhs()->comment() << " " << cmpstr << " " << rhs()->comment();
  }
}

Value* FCmpInst::getConstantRepl() {
  auto clhs = dyn_cast<Constant>(lhs());
  auto crhs = dyn_cast<Constant>(rhs());
  if (not clhs or not crhs)
    return nullptr;
  auto lhsval = clhs->f32();
  auto rhsval = crhs->f32();
  switch (valueId()) {
    case vFOEQ:
      return Constant::gen_i1(lhsval == rhsval);
    case vFONE:
      return Constant::gen_i1(lhsval != rhsval);
    case vFOGT:
      return Constant::gen_i1(lhsval > rhsval);
    case vFOLT:
      return Constant::gen_i1(lhsval < rhsval);
    case vFOGE:
      return Constant::gen_i1(lhsval >= rhsval);
    case vFOLE:
      return Constant::gen_i1(lhsval <= rhsval);
    default:
      assert(false and "fcmpinst const flod error");
  }
}
void BranchInst::replaceDest(ir::BasicBlock* olddest, ir::BasicBlock* newdest) {
  if (mIsCond) {
    if (iftrue() == olddest) {
      setOperand(1, newdest);
    } else {
      setOperand(2, newdest);
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
  os << "br ";
  if (is_cond()) {
    os << "i1 ";
    os << cond()->name() << ", ";
    os << "label %" << iftrue()->name() << ", ";
    os << "label %" << iffalse()->name();
    /* comment */
    if (not iftrue()->comment().empty() && not iffalse()->comment().empty()) {
      os << " ; " << "br " << iftrue()->comment() << ", "
         << iffalse()->comment();
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
    os << name() << " = getelementptr ";

    size_t dimensions = cur_dims_cnt();
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

    os << value()->name() << ", ";
    os << "i32 0, i32 " << index()->name();
  } else {
    os << name() << " = getelementptr " << *(baseType()) << ", " << *type()
       << " ";
    os << value()->name() << ", i32 " << index()->name();
  }
}

void CallInst::print(std::ostream& os) const {
  if (callee()->retType()->isVoid()) {
    os << "call ";
  } else {
    os << name() << " = call ";
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
      os << val->name();
      os << ", ";
    }
    auto lastval = (*last)->value();
    os << *(lastval->type()) << " ";
    os << lastval->name();
  }
  os << ")";
}

void PhiInst::print(std::ostream& os) const {
  // 在打印的时候对其vals和bbs进行更新
  os << name() << " = ";
  os << "phi " << *(type()) << " ";
  // for all vals, bbs
  for (size_t i = 0; i < mSize; i++) {
    os << "[ " << getValue(i)->name() << ", %" << getBlock(i)->name() << " ]";
    if (i != mSize - 1)
      os << ",";
  }
}

BasicBlock* PhiInst::getbbfromVal(Value* val) {
  for (size_t i = 0; i < mSize; i++) {
    if (getValue(i) == val)
      return getBlock(i);
  }
  return nullptr;
}

Value* PhiInst::getvalfromBB(BasicBlock* bb) {
  for (size_t i = 0; i < mSize; i++) {
    if (getBlock(i) == bb)
      return getValue(i);
  }
  return nullptr;
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

void PhiInst::delValue(Value* val) {
  size_t i;
  for (i = 0; i < mSize; i++) {
    if (getValue(i) == val)
      break;
  }
  assert(i != mSize);
  delete_operands(2 * i);
  delete_operands(2 * i);
  mSize--;
}

void PhiInst::delBlock(BasicBlock* bb) {
  size_t i;
  for (i = 0; i < mSize; i++) {
    if (getBlock(i) == bb)
      break;
  }
  assert(i != mSize);
  delete_operands(2 * i);
  delete_operands(2 * i);
  mSize--;
}

void PhiInst::replaceBlock(BasicBlock* newBB, size_t k) {
  setOperand(2 * k + 1, newBB);
}

void PhiInst::replaceoldtonew(BasicBlock* newbb,BasicBlock* oldbb){
  size_t i;
  for (i = 0; i < mSize; i++) {
    if (getBlock(i) == oldbb)
      break;
  }
  setOperand(2 * i + 1, newbb);
}

/*
 * @brief: BitcastInst::print
 * @details:
 *      <result> = bitcast <ty> <value> to i8*
 */
void BitCastInst::print(std::ostream& os) const {
  os << name() << " = bitcast ";
  os << *type() << " " << value()->name();
  os << " to i8*";
}

/*
 * @brief: memset
 * @details:
 *      call void @llvm.memset.p0i8.i64(i8* <dest>, i8 0, i64 <len>, i1 false)
 */
void MemsetInst::print(std::ostream& os) const {
  assert(dyn_cast<PointerType>(type()) && "type error");
  os << "call void @llvm.memset.p0i8.i64(i8* ";
  os << value()->name() << ", i8 0, i64 "
     << dyn_cast<PointerType>(type())->baseType()->size() << ", i1 false)";
}

}  // namespace ir
