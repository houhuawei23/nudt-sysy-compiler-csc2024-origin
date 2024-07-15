#include "ir/builder.hpp"
namespace ir {

Value* IRBuilder::makeBinary(BinaryOp op, Value* lhs, Value* rhs) {
  const Type *ltype = lhs->type(), *rtype = rhs->type();
  if (ltype != rtype) {
    assert(false && "create_eq_beta: type mismatch!");
  }

  auto vid = [btype = ltype->btype(), op] {
    switch (btype) {
      case INT32: {
        switch (op) {
          case BinaryOp::ADD:
            return ValueId::vADD;
          case BinaryOp::SUB:
            return ValueId::vSUB;
          case BinaryOp::MUL:
            return ValueId::vMUL;
          case BinaryOp::DIV:
            return ValueId::vSDIV;
          case BinaryOp::REM:
            return ValueId::vSREM;
          default:
            assert(false && "makeBinary: invalid op!");
        }
      }
      case FLOAT: {
        switch (op) {
          case BinaryOp::ADD:
            return ValueId::vFADD;
          case BinaryOp::SUB:
            return ValueId::vFSUB;
          case BinaryOp::MUL:
            return ValueId::vFMUL;
          case BinaryOp::DIV:
            return ValueId::vFDIV;
          default:
            assert(false && "makeBinary: invalid op!");
        }
      }
    }
    return ValueId::vInvalid;
  }();
  auto res = makeInst<BinaryInst>(vid, lhs->type(), lhs, rhs);
  // auto res = makeInstBeta<BinaryInst>(vid, lhs->type(), lhs, rhs);
  return res;
}

Value* IRBuilder::makeUnary(ValueId vid, Value* val, Type* ty) {
  //! check vid
  Value* res = nullptr;

  if (vid == ValueId::vFNEG) {
    assert(val->type()->isFloatPoint() && "fneg must have float operand");
    res = makeInst<UnaryInst>(vid, Type::TypeFloat32(), val);
    return res;
  }
  //! else
  // assert(ty != nullptr && "must have target type");

  switch (vid) {
    case ValueId::vSITOFP:
      ty = Type::TypeFloat32();
      assert(val->type()->isInt() && "sitofp must have int operand");
      assert(ty->isFloatPoint() && "sitofp must have float type");
      break;
    case ValueId::vFPTOSI:
      ty = Type::TypeInt32();
      assert(val->type()->isFloatPoint() && "fptosi must have float operand");
      assert(ty->isInt() && "fptosi must have int type");
      break;
    case ValueId::vTRUNC:
    case ValueId::vZEXT:
    case ValueId::vSEXT:
      assert(val->type()->isInt() && ty->isInt());
      break;
    case ValueId::vFPTRUNC:
      assert(val->type()->isFloatPoint() && ty->isFloatPoint());
      break;
  }

  res = makeInst<UnaryInst>(vid, ty, val);
  return res;
}

Value* IRBuilder::promoteType(Value* val, Type* target_tpye, Type* base_type) {
  Value* res = val;
  if (val->type()->btype() < target_tpye->btype()) {
    // need to promote
    if (val->type()->isBool()) {
      if (target_tpye->isInt32()) {
        res = makeUnary(ValueId::vZEXT, res, Type::TypeInt32());
      } else if (target_tpye->isFloatPoint()) {
        res = makeUnary(ValueId::vZEXT, res, Type::TypeInt32());
        res = makeUnary(ValueId::vSITOFP, res);
      }
    } else if (val->type()->isInt32()) {
      if (target_tpye->isFloatPoint()) {
        res = makeUnary(ValueId::vSITOFP, res);
      }
    }
  }
  return res;
}

Value* IRBuilder::makeCmp(CmpOp op, Value* lhs, Value* rhs) {
  if (lhs->type() != rhs->type()) {
    assert(false && "create_eq_beta: type mismatch!");
  };

  auto vid = [type = lhs->type()->btype(), op] {
    switch (type) {
      case INT32:
        switch (op) {
          case CmpOp::EQ:
            return ValueId::vIEQ;
          case CmpOp::NE:
            return ValueId::vINE;
          case CmpOp::GT:
            return ValueId::vISGT;
          case CmpOp::GE:
            return ValueId::vISGE;
          case CmpOp::LT:
            return ValueId::vISLT;
          case CmpOp::LE:
            return ValueId::vISLE;
          default:
            assert(false && "makeCmp: invalid op!");
        }
      case FLOAT:
      case DOUBLE:
        switch (op) {
          case CmpOp::EQ:
            return ValueId::vFOEQ;
          case CmpOp::NE:
            return ValueId::vFONE;
          case CmpOp::GT:
            return ValueId::vFOGT;
          case CmpOp::GE:
            return ValueId::vFOGE;
          case CmpOp::LT:
            return ValueId::vFOLT;
          case CmpOp::LE:
            return ValueId::vFOLE;
          default:
            assert(false && "makeCmp: invalid op!");
        }
    }
    assert(false && "makeCmp: invalid type!");
    return ValueId::vInvalid;
  }();

  switch (lhs->type()->btype()) {
    case INT32:
      return makeInst<ICmpInst>(vid, lhs, rhs);
      break;
    case FLOAT:
    case DOUBLE:
      return makeInst<FCmpInst>(vid, lhs, rhs);
      break;
    default:
      assert(false && "create_eq_beta: type mismatch!");
  }
}

Value* IRBuilder::castToBool(Value* val) {
  Value* res = nullptr;
  if (not val->isBool()) {
    if (val->isInt32()) {
      res = makeInst<ICmpInst>(ValueId::vINE, val, Constant::gen_i32(0));
    } else if (val->isFloatPoint()) {
      res = makeInst<FCmpInst>(ValueId::vFONE, val, Constant::gen_f32(0.0));
    }
  } else {
    res = val;
  }
  return res;
}

Value* IRBuilder::makeLoad(Value* ptr) {
  auto type = [ptr] {
    if (ptr->type()->isPointer())
      return ptr->type()->as<PointerType>()->baseType();
    else
      return ptr->type()->as<ArrayType>()->baseType();
  }();
  auto inst = makeInst<LoadInst>(ptr, type);

  if (not ptr->comment().empty()) {
    inst->setComment(ptr->comment());
  }
  return inst;
}

Value* IRBuilder::makeAlloca(Type* base_type,
                             bool is_const,
                             const std::vector<size_t>& dims,
                             const_str_ref comment,
                             size_t capacity) {
  AllocaInst* inst = nullptr;
  const auto entryBlock = mBlock->function()->entry();

  if (dims.size() == 0) {
    inst = makeIdenticalInst<AllocaInst>(base_type, entryBlock, "", is_const);

  } else {
    inst = makeIdenticalInst<AllocaInst>(base_type, dims, entryBlock, "",
                                         is_const, capacity);
  }

  /* hhw, add alloca to function entry block*/
  // entry already has a terminator, br
  entryBlock->emplace_inst(--entryBlock->insts().end(), inst);
  inst->setComment(comment);
  return inst;
}

Value* IRBuilder::makeGetElementPtr(Type* base_type,
                                    Value* value,
                                    Value* idx,
                                    std::vector<size_t> dims,
                                    std::vector<size_t> cur_dims) {
  GetElementPtrInst* inst = nullptr;
  if (dims.size() == 0 && cur_dims.size() == 0) {
    inst = makeInst<GetElementPtrInst>(base_type, value, idx);
  } else if (dims.size() == 0 && cur_dims.size() != 0) {
    inst = makeInst<GetElementPtrInst>(base_type, value, idx, cur_dims);
  } else {
    inst = makeInst<GetElementPtrInst>(base_type, value, idx, dims, cur_dims);
  }
  return inst;
}
};  // namespace ir