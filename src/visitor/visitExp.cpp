#include <any>
#include "visitor/visitor.hpp"

namespace sysy {
/*
 * @brief: Visit Number Expression
 * @details:
 *      number: ILITERAL | FLITERAL; (即: int or float)
 */
std::any SysYIRGenerator::visitNumberExp(SysYParser::NumberExpContext* ctx) {
  ir::Value* res = nullptr;
  if (auto iLiteral = ctx->number()->ILITERAL()) {
    //! 基数 (8, 10, 16)
    const auto text = iLiteral->getText();
    int base = 10;
    if (text.find("0x") == 0 || text.find("0X") == 0) {
      base = 16;
    } else if (text.find("0b") == 0 || text.find("0B") == 0) {
      base = 2;
    } else if (text.find("0") == 0) {
      base = 8;
    }
    res = ir::Constant::gen_i32(std::stol(text, 0, base));
  } else if (auto fLiteral = ctx->number()->FLITERAL()) {
    const auto text = fLiteral->getText();
    res = ir::Constant::gen_f32(std::stof(text));
  }
  return dyn_cast_Value(res);
}

/*
 * @brief: Visit Var Expression
 * @details:
 *      var: ID (LBRACKET exp RBRACKET)*;
 */
std::any SysYIRGenerator::visitVarExp(SysYParser::VarExpContext* ctx) {
  const auto varname = ctx->var()->ID()->getText();
  auto ptr = mTables.lookup(varname);
  assert(ptr && "use undefined variable");

  bool isArray = false;
  if (auto ptype = ptr->type()->dynCast<ir::PointerType>()) {
    isArray =
        ptype->baseType()->isArray() || ptype->baseType()->isPointer();
  }
  if (!isArray) {
    //! 1. scalar
    if (auto gptr = ptr->dynCast<ir::GlobalVariable>()) {  // 全局
      if (gptr->isConst()) {                              // 常量
        ptr = gptr->scalarValue();
      } else {  // 变量
        ptr = mBuilder.makeLoad(ptr);
      }
    } else {  // 局部 (变量 - load, 常量 - ignore)
      if (!ir::isa<ir::Constant>(ptr)) {
        ptr = mBuilder.makeLoad(ptr);
      }
    }
  } else {  //! 2. array
    auto type = ptr->type()->as<ir::PointerType>()->baseType();
    if (type->isArray()) {  // 数组 (eg. int a[2][3]) -> 常规使用
      auto atype = dyn_cast<ir::ArrayType>(type);
      auto base_type = atype->baseType();
      auto dims = atype->dims(), cur_dims(dims);

      int delta = dims.size() - ctx->var()->exp().size();
      for (auto expr : ctx->var()->exp()) {
        auto idx = any_cast_Value(visit(expr));
        dims.erase(dims.begin());
        ptr = mBuilder.makeGetElementPtr(base_type, ptr, idx, dims, cur_dims);
        cur_dims.erase(cur_dims.begin());
      }
      if (ctx->var()->exp().empty()) {
        dims.erase(dims.begin());
        ptr = mBuilder.makeGetElementPtr(
            base_type, ptr, ir::Constant::gen_i32(0), dims, cur_dims);
      } else if (delta > 0) {
        dims.erase(dims.begin());
        ptr = mBuilder.makeGetElementPtr(
            base_type, ptr, ir::Constant::gen_i32(0), dims, cur_dims);
      }

      if (delta == 0)
        ptr = mBuilder.makeLoad(ptr);
      // ptr = mBuilder.makeInst<ir::LoadInst>(ptr);
    } else if (type->isPointer()) {  // 一级及指针 (eg. int a[] OR int
                                      // a[][5]) -> 函数参数
      ptr = mBuilder.makeLoad(ptr);
      // ptr = mBuilder.makeInst<ir::LoadInst>(ptr);
      type = dyn_cast<ir::PointerType>(type)->baseType();
      if (type->isArray()) {  // 二级及以上指针 (eg. int a[][5])
                               // Pointer<Array>
        if (ctx->var()->exp().size()) {
          auto expr_vec = ctx->var()->exp();

          auto idx = any_cast_Value(visit(expr_vec[0]));
          ptr = mBuilder.makeGetElementPtr(type, ptr, idx);
          auto base_type = dyn_cast<ir::ArrayType>(type)->baseType();
          auto dims = dyn_cast<ir::ArrayType>(type)->dims(), cur_dims(dims);
          int delta = dims.size() + 1 - expr_vec.size();
          for (int i = 1; i < expr_vec.size(); i++) {
            idx = any_cast_Value(visit(expr_vec[i]));
            dims.erase(dims.begin());
            ptr =
                mBuilder.makeGetElementPtr(base_type, ptr, idx, dims, cur_dims);
            cur_dims.erase(cur_dims.begin());
          }
          if (delta > 0) {  // 参数传递
            dims.erase(dims.begin());
            ptr = mBuilder.makeGetElementPtr(
                base_type, ptr, ir::Constant::gen_i32(0), dims, cur_dims);
            cur_dims.erase(cur_dims.begin());
          } else if (delta == 0) {
            ptr = mBuilder.makeLoad(ptr);
            // ptr = mBuilder.makeInst<ir::LoadInst>(ptr);
          } else {
            assert(false && "the array dimensions error");
          }
        }
      } else {  // 一级指针
        for (auto expr : ctx->var()->exp()) {
          auto idx = any_cast_Value(visit(expr));
          ptr = mBuilder.makeGetElementPtr(type, ptr, idx);
        }
        if (ctx->var()->exp().size())
          ptr = mBuilder.makeLoad(ptr);
        // ptr = mBuilder.makeInst<ir::LoadInst>(ptr);
      }
    } else {
      assert(false && "type error");
    }
  }
  return dyn_cast_Value(ptr);
}

/*
 * @brief: visit lvalue
 * @details:
 *      lValue: ID (LBRACKET exp RBRACKET)*
 */
std::any SysYIRGenerator::visitLValue(SysYParser::LValueContext* ctx) {
  //! lvalue must be a pointer
  const auto name = ctx->ID()->getText();
  auto ptr = mTables.lookup(name);
  assert(ptr && "use undefined variable");

  bool isArray = false;
  if (auto ptype = dyn_cast<ir::PointerType>(ptr->type())) {
    isArray =
        ptype->baseType()->isArray() || ptype->baseType()->isPointer();
  }
  if (!isArray) {  //! 1. scalar
    return dyn_cast_Value(ptr);
  } else {  //! 2. array
    ir::Type* type = dyn_cast<ir::PointerType>(ptr->type())->baseType();
    if (type->isArray()) {  // 数组 (eg. int a[2][3]) -> 常规使用
      auto atype = dyn_cast<ir::ArrayType>(type);
      auto base_type = atype->baseType();
      auto dims = atype->dims(), cur_dims(dims);
      for (auto expr : ctx->exp()) {
        auto idx = any_cast_Value(visit(expr));
        dims.erase(dims.begin());
        ptr = mBuilder.makeGetElementPtr(base_type, ptr, idx, dims, cur_dims);
        cur_dims.erase(cur_dims.begin());
      }
    } else if (type->isPointer()) {  // 指针 (eg. int a[] OR int a[][5]) ->
                                      // 函数参数
      ptr = mBuilder.makeLoad(ptr);
      type = dyn_cast<ir::PointerType>(type)->baseType();

      if (type->isArray()) {
        auto expr_vec = ctx->exp();

        auto idx = any_cast_Value(visit(expr_vec[0]));
        ptr = mBuilder.makeGetElementPtr(type, ptr, idx);
        auto base_type = dyn_cast<ir::ArrayType>(type)->baseType();
        auto dims = dyn_cast<ir::ArrayType>(type)->dims(), cur_dims(dims);

        for (int i = 1; i < expr_vec.size(); i++) {
          idx = any_cast_Value(visit(expr_vec[i]));
          dims.erase(dims.begin());
          ptr = mBuilder.makeGetElementPtr(base_type, ptr, idx, dims, cur_dims);
          cur_dims.erase(cur_dims.begin());
        }
      } else {
        for (auto expr : ctx->exp()) {
          auto idx = any_cast_Value(visit(expr));
          ptr = mBuilder.makeGetElementPtr(type, ptr, idx);
        }
      }
    } else {
      assert(false && "type error");
    }
  }

  return dyn_cast_Value(ptr);
}

/*
 * @brief Visit Unary Expression
 * @details:
 *      + - ! exp
 */
std::any SysYIRGenerator::visitUnaryExp(SysYParser::UnaryExpContext* ctx) {
  ir::Value* res = nullptr;
  auto exp = any_cast_Value(visit(ctx->exp()));
  if (ctx->ADD()) {
    res = exp;
  } else if (ctx->NOT()) {
    const auto true_target = mBuilder.false_target();
    const auto false_target = mBuilder.true_target();
    mBuilder.pop_tf();
    mBuilder.push_tf(true_target, false_target);
    res = exp;
  } else if (ctx->SUB()) {
    if (auto cexp = dyn_cast<ir::Constant>(exp)) {
      //! constant
      switch (cexp->type()->btype()) {
        case ir::INT32:
          res = ir::Constant::gen_i32(-cexp->i32());
          break;
        case ir::FLOAT:
          res = ir::Constant::gen_f32(-cexp->f32());
          break;
        case ir::DOUBLE:
          assert(false && "Unsupport Double");
          break;
        default:
          assert(false && "Unsupport btype");
      }
    } else if (ir::isa<ir::LoadInst>(exp) || ir::isa<ir::BinaryInst>(exp)) {
      switch (exp->type()->btype()) {
        case ir::INT32:
          res = mBuilder.makeBinary(ir::BinaryOp::SUB, ir::Constant::gen_i32(0),
                                    exp);
          break;
        case ir::FLOAT:
          res = mBuilder.makeUnary(ir::ValueId::vFNEG, exp);
          break;
        default:
          assert(false && "Unsupport btype");
      }
    } else {
      assert(false && "invalid value type");
    }
  } else {
    assert(false && "invalid expression");
  }
  return dyn_cast_Value(res);
}

std::any SysYIRGenerator::visitParenExp(SysYParser::ParenExpContext* ctx) {
  return any_cast_Value(visit(ctx->exp()));
}

/*
 * @brief Visit Multiplicative Expression
 *      exp (MUL | DIV | MODULO) exp
 * @details
 *      1. mul: 整型乘法
 *      2. fmul: 浮点型乘法
 *
 *      3. udiv: 无符号整型除法 ???
 *      4. sdiv: 有符号整型除法
 *      5. fdiv: 有符号浮点型除法
 *
 *      6. urem: 无符号整型取模 ???
 *      7. srem: 有符号整型取模1
 *      8. frem: 有符号浮点型取模
 */
std::any SysYIRGenerator::visitMultiplicativeExp(
    SysYParser::MultiplicativeExpContext* ctx) {
  auto op1 = any_cast_Value(visit(ctx->exp(0)));
  auto op2 = any_cast_Value(visit(ctx->exp(1)));
  ir::Value* res;
  if (ir::isa<ir::Constant>(op1) && ir::isa<ir::Constant>(op2)) {
    //! both constant (常数折叠)
    const auto cop1 = dyn_cast<ir::Constant>(op1);
    const auto cop2 = dyn_cast<ir::Constant>(op2);
    const auto higher_Btype =
        std::max(cop1->type()->btype(), cop2->type()->btype());

    int32_t ans_i32;
    float ans_f32;
    double ans_f64;

    switch (higher_Btype) {
      case ir::INT32:
        if (ctx->MUL())
          ans_i32 = cop1->i32() * cop2->i32();
        else if (ctx->DIV())
          ans_i32 = cop1->i32() / cop2->i32();
        else if (ctx->MODULO())
          ans_i32 = cop1->i32() % cop2->i32();
        else
          assert(false && "Unknown Binary Operator");
        res = ir::Constant::gen_i32(ans_i32);
        break;
      case ir::FLOAT:
        if (ctx->MUL())
          ans_f32 = cop1->f32() * cop2->f32();
        else if (ctx->DIV())
          ans_f32 = cop1->f32() / cop2->f32();
        else
          assert(false && "Unknown Binary Operator");
        res = ir::Constant::gen_f32(ans_f32);
        break;
      case ir::DOUBLE:
        assert(false && "Unsupported DOUBLE");
        break;
      default:
        assert(false && "Unknown BType");
        break;
    }
  } else {  //! not all constant
    auto hi_type = op1->type();
    if (op1->type()->btype() < op2->type()->btype()) {
      hi_type = op2->type();
    }

    op1 = mBuilder.promoteType(op1, hi_type);  // base i32
    op2 = mBuilder.promoteType(op2, hi_type);

    if (ctx->MUL()) {
      res = mBuilder.makeBinary(ir::BinaryOp::MUL, op1, op2);
    } else if (ctx->DIV()) {
      res = mBuilder.makeBinary(ir::BinaryOp::DIV, op1, op2);
    } else if (ctx->MODULO()) {
      res = mBuilder.makeBinary(ir::BinaryOp::REM, op1, op2);
    } else {
      assert(false && "not support");
    }
  }
  return dyn_cast_Value(res);
}

/*
 * @brief Visit Additive Expression
 * @details:
 *      exp (ADD | SUB) exp
 */
std::any SysYIRGenerator::visitAdditiveExp(
    SysYParser::AdditiveExpContext* ctx) {
  auto lhs = any_cast_Value(visit(ctx->exp()[0]));
  auto rhs = any_cast_Value(visit(ctx->exp()[1]));
  ir::Value* res;

  if (ir::isa<ir::Constant>(lhs) && ir::isa<ir::Constant>(rhs)) {
    //! constant
    auto clhs = dyn_cast<ir::Constant>(lhs);
    auto crhs = dyn_cast<ir::Constant>(rhs);
    // bool is_ADD = ctx->ADD();

    auto higher_BType = std::max(clhs->type()->btype(), crhs->type()->btype());

    int32_t ans_i32;
    float ans_f32;
    double ans_f64;

    switch (higher_BType) {
      case ir::INT32: {
        if (ctx->ADD())
          ans_i32 = clhs->i32() + crhs->i32();
        else
          ans_i32 = clhs->i32() - crhs->i32();
        res = ir::Constant::gen_i32(ans_i32);
      } break;

      case ir::FLOAT: {
        if (ctx->ADD())
          ans_f32 = clhs->f32() + crhs->f32();
        else
          ans_f32 = clhs->f32() - crhs->f32();
        res = ir::Constant::gen_f32(ans_f32);
      } break;

      case ir::DOUBLE: {
        assert(false && "not support double");
      } break;

      default:
        assert(false && "not support");
    }

  } else {
    //! not all constant
    auto hi_type = lhs->type();
    if (lhs->type()->btype() < rhs->type()->btype()) {
      hi_type = rhs->type();
    }
    lhs = mBuilder.promoteType(lhs, hi_type);  // base i32
    rhs = mBuilder.promoteType(rhs, hi_type);

    if (ctx->ADD()) {
      res = mBuilder.makeBinary(ir::BinaryOp::ADD, lhs, rhs);
    } else if (ctx->SUB()) {
      res = mBuilder.makeBinary(ir::BinaryOp::SUB, lhs, rhs);
    } else {
      assert(false && "not support");
    }
  }

  return dyn_cast_Value(res);
}
//! exp (LT | GT | LE | GE) exp
std::any SysYIRGenerator::visitRelationExp(
    SysYParser::RelationExpContext* ctx) {
  auto lhsptr = any_cast_Value(visit(ctx->exp()[0]));
  auto rhsptr = any_cast_Value(visit(ctx->exp()[1]));

  ir::Value* res;

  auto hi_type = lhsptr->type();
  if (lhsptr->type()->btype() < rhsptr->type()->btype()) {
    hi_type = rhsptr->type();
  }
  lhsptr = mBuilder.promoteType(lhsptr, hi_type);  // base i32
  rhsptr = mBuilder.promoteType(rhsptr, hi_type);

  if (ctx->GT()) {
    res = mBuilder.makeCmp(ir::CmpOp::GT, lhsptr, rhsptr);
  } else if (ctx->GE()) {
    res = mBuilder.makeCmp(ir::CmpOp::GE, lhsptr, rhsptr);
  } else if (ctx->LT()) {
    res = mBuilder.makeCmp(ir::CmpOp::LT, lhsptr, rhsptr);
  } else if (ctx->LE()) {
    res = mBuilder.makeCmp(ir::CmpOp::LE, lhsptr, rhsptr);
  } else {
    std::cerr << "Unknown relation operator!" << std::endl;
  }
  return dyn_cast_Value(res);
}

//! exp (EQ | NE) exp
/**
 * i1  vs i1     -> i32 vs i32       (zext)
 * i1  vs i32    -> i32 vs i32       (zext)
 * i1  vs float  -> float vs float   (zext, sitofp)
 * i32 vs float  -> float vs float   (sitofp)
 */
std::any SysYIRGenerator::visitEqualExp(SysYParser::EqualExpContext* ctx) {
  auto lhsptr = any_cast_Value(visit(ctx->exp()[0]));
  auto rhsptr = any_cast_Value(visit(ctx->exp()[1]));
  ir::Value* res;

  auto hi_type = lhsptr->type();
  if (lhsptr->type()->btype() < rhsptr->type()->btype()) {
    hi_type = rhsptr->type();
  }

  lhsptr = mBuilder.promoteType(lhsptr, hi_type);  // base i32
  rhsptr = mBuilder.promoteType(rhsptr, hi_type);

  // same type
  if (ctx->EQ()) {
    res = mBuilder.makeCmp(ir::CmpOp::EQ, lhsptr, rhsptr);
  } else if (ctx->NE()) {
    res = mBuilder.makeCmp(ir::CmpOp::NE, lhsptr, rhsptr);
  } else {
    std::cerr << "not valid equal exp" << std::endl;
  }
  return dyn_cast_Value(res);
}

/*
 * @brief visit And Expressions
 * @details:
 *      exp: exp AND exp;
 * @note:
 *       - before you visit one exp, you must prepare its true and false
 * target
 *       1. push the thing you protect
 *       2. call the function
 *       3. pop to reuse OR use tmp var to log
 * // exp: lhs AND rhs
 * // know exp's true/false target block
 * // lhs's true target = rhs block
 * // lhs's false target = exp false target
 * // rhs's true target = exp true target
 * // rhs's false target = exp false target
 */
std::any SysYIRGenerator::visitAndExp(SysYParser::AndExpContext* ctx) {
  const auto cur_func = mBuilder.curBlock()->function();

  auto rhs_block = cur_func->newBlock();
  rhs_block->addComment("rhs_block");

  {
    //! 1 visit lhs exp to get its value
    //! diff with OrExp
    mBuilder.push_tf(rhs_block, mBuilder.false_target());
    auto lhs_value = any_cast_Value(visit(ctx->exp(0)));  // recursively visit
    //! may chage by visit, need to re get
    const auto lhs_t_target = mBuilder.true_target();
    const auto lhs_f_target = mBuilder.false_target();
    mBuilder.pop_tf();  // match with push_tf

    lhs_value = mBuilder.castToBool(lhs_value);

    mBuilder.makeInst<ir::BranchInst>(lhs_value, lhs_t_target, lhs_f_target);

    //! 2 [for CFG] link cur_block and target
    // visit may change the cur_block, so need to reload the cur block
    ir::BasicBlock::block_link(mBuilder.curBlock(), lhs_t_target);
    ir::BasicBlock::block_link(mBuilder.curBlock(), lhs_f_target);
  }

  //! 3 visit and generate code for rhs block
  mBuilder.set_pos(rhs_block);

  auto rhs_value = any_cast_Value(visit(ctx->exp(1)));

  return rhs_value;
}

//! exp OR exp
// lhs OR rhs
// know exp's true/false target block (already in builder's stack)
// lhs true target = exp true target
// lhs false target = rhs block
// rhs true target = exp true target
// rhs false target = exp false target
std::any SysYIRGenerator::visitOrExp(SysYParser::OrExpContext* ctx) {
  auto cur_func = mBuilder.curBlock()->function();

  auto rhs_block = cur_func->newBlock();
  rhs_block->addComment("rhs_block");

  {
    //! 1 visit lhs exp to get its value
    mBuilder.push_tf(mBuilder.true_target(), rhs_block);
    auto lhs_value = any_cast_Value(visit(ctx->exp(0)));
    const auto lhs_t_target = mBuilder.true_target();
    const auto lhs_f_target = mBuilder.false_target();
    mBuilder.pop_tf();  // match with push_tf

    lhs_value = mBuilder.castToBool(lhs_value);

    mBuilder.makeInst<ir::BranchInst>(lhs_value, lhs_t_target, lhs_f_target);

    //! 2 [for CFG] link cur_block and target
    // visit may change the cur_block, so need to reload the cur block
    ir::BasicBlock::block_link(mBuilder.curBlock(), lhs_t_target);
    ir::BasicBlock::block_link(mBuilder.curBlock(), lhs_f_target);
  }

  //! 3 visit and generate code for rhs block
  mBuilder.set_pos(rhs_block);

  auto rhs_value = any_cast_Value(visit(ctx->exp(1)));

  return rhs_value;
}

}  // namespace sysy