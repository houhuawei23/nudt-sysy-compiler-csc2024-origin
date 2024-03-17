#include "visitor.hpp"

namespace sysy
{
    /*
     * @brief Visit Number Expression
     *      exp (MUL | DIV | MODULO) exp
     * @details
     *      number: ILITERAL | FLITERAL; (即: int or float)
     */
    std::any SysYIRGenerator::visitNumberExp(SysYParser::NumberExpContext *ctx) {
        ir::Value *res = nullptr;
        if (auto iLiteral = ctx->number()->ILITERAL()) {  //! int
            std::string s = iLiteral->getText();
            
            //! 基数 (8, 10, 16)
            int base = 10;
            if (s.length() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) base = 16;
            else if (s[0] == '0') base = 8;

            res = ir::Constant::gen(std::stoi(s, 0, base));
        }
        else if (auto fctx = ctx->number()->FLITERAL()) {  //! float
            std::string s = fctx->getText();
            float f = std::stof(s);
            res = ir::Constant::gen(f, ir::getMC(f));
        }
        return res;
    }

std::any SysYIRGenerator::visitVarExp(SysYParser::VarExpContext* ctx) {
    ir::Value* res = nullptr;
    bool isarray = not ctx->var()->LBRACKET().empty();
    std::string varname = ctx->var()->ID()->getText();

    auto valueptr = _tables.lookup(varname);
    if (valueptr == nullptr) {
        std::cerr << "Use undefined variable: \"" << varname << "\""
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!isarray) {  //! scalar
        if (res = ir::dyn_cast<ir::Constant>(valueptr)) {
        } else {
            res = _builder.create_load(valueptr, {}, _builder.getvarname());
        }
    } else {  //! array
        // TODO
    }
    return res;
}

/*
 * @brief Visit Unary Expression
 *      + - (NOT) exp
 */
std::any SysYIRGenerator::visitUnaryExp(SysYParser::UnaryExpContext* ctx) {
    // ir::Value* op = any_cast_Value(visit(ctx->exp()));
    // ir::Value* res;

    // if (ir::isa<ir::Constant>(op)) {  //! 1. 常量 -> 常量折叠
    //     ir::Constant* cop = ir::dyn_cast<ir::Constant>(op);
    //     if (ctx->NOT()) {
    //         auto ans = !(cop->is_float() ? cop->f() : cop->i());
    //         if (typeid(ans) == typeid(float)) res = ir::Constant::gen(ans,
    //         ir::getMC(ans)); else res = ir::Constant::gen(ans);
    //     } else if (ctx->ADD()) {
    //         // TODO
    //     } else {
    //         // TODO
    //     }
    // } else {  //! 2. 变量 -> 生成 NOT 指令
    //     if (op->is_int()) {
    //         // int32 类型
    //         res = _builder.create_fneg(ir::Type::int_type(), op,
    //         _builder.getvarname());
    //     } else {
    //         // float 类型
    //         res = _builder.create_fneg(ir::Type::float_type(), op,
    //         _builder.getvarname());
    //     }
    //     //! TODO
    // }

    ir::Value* res = nullptr;
    auto exp = any_cast_Value(visit(ctx->exp()));
    if (ctx->SUB()) {
        //! Constant, static type cast
        if (auto cexp = ir::dyn_cast<ir::Constant>(exp)) {
            if (exp->is_int()) {
                res = ir::Constant::gen(-cexp->i());
            } else if (exp->is_float()) {
                res = ir::Constant::gen(-cexp->f(), ir::getMC(-cexp->f()));
            }
        } else if (ir::isa<ir::AllocaInst>(exp) &&
                   ir::dyn_cast<ir::AllocaInst>(exp)->is_scalar()) {
            // 如果是 AllocaInst and 标量
            //! TODO:
            int a = 5;
        } else if (exp->is_int()) {
        } else if (exp->is_float()) {
        }
        //! TODO
    } else if (ctx->NOT()) {
        //! TODO: if else, then is used
        auto true_target = builder().false_target();
        auto false_target = builder().true_target();
        builder().pop_tf();
        builder().push_tf(true_target, false_target);
        res = exp;
    } else if (ctx->ADD()) {
        res = exp;
    }
    return res;
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
 *      7. srem: 有符号整型取模
 *      8. frem: 有符号浮点型取模
 */
std::any SysYIRGenerator::visitMultiplicativeExp(
    SysYParser::MultiplicativeExpContext* ctx) {
    ir::Value* op1 = any_cast_Value(visit(ctx->exp()[0]));
    ir::Value* op2 = any_cast_Value(visit(ctx->exp()[1]));
    ir::Value* res;
    if (ir::isa<ir::Constant>(op1) &&
        ir::isa<ir::Constant>(op2)) {  //! 1. 常量 -> 常量折叠
        ir::Constant* cop1 = ir::dyn_cast<ir::Constant>(op1);
        ir::Constant* cop2 = ir::dyn_cast<ir::Constant>(op2);
        if (ctx->DIV()) {
            auto ans = (cop1->is_float() ? cop1->f() : cop1->i()) /
                       (cop2->is_float() ? cop2->f() : cop2->i());
            if (typeid(ans) == typeid(float))
                res = ir::Constant::gen(ans, ir::getMC(ans));
            else
                res = ir::Constant::gen(ans);
        } else if (ctx->MUL()) {
            auto ans = (cop1->is_float() ? cop1->f() : cop1->i()) *
                       (cop2->is_float() ? cop2->f() : cop2->i());
            if (typeid(ans) == typeid(float))
                res = ir::Constant::gen(ans, ir::getMC(ans));
            else
                res = ir::Constant::gen(ans);
        } else {  // MODULO
            if (cop1->is_int() && cop2->is_int()) {
                int ans = cop1->i() % cop2->i();
                res = ir::Constant::gen(ans);
            } else {
                std::cerr << "Operands of modulo must be integer!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    } else {  //! 2. 变量 -> 生成 MUL | FMUL | UDIV | SDIV | FDIV | UREM | SREM
              //! | FREM 指令
        if (op1->is_int() && op2->is_int()) {
            // int32 类型
            if (ctx->MUL())
                res = _builder.create_mul(ir::Type::int_type(), op1, op2,
                                          _builder.getvarname());
            else if (ctx->DIV())
                res = _builder.create_div(ir::Type::int_type(), op1, op2,
                                          _builder.getvarname());
            else
                res = _builder.create_rem(ir::Type::int_type(), op1, op2,
                                          _builder.getvarname());
        } else if (op1->is_float() && op2->is_float()) {
            // float 类型
            if (ctx->MUL())
                res = _builder.create_mul(ir::Type::float_type(), op1, op2,
                                          _builder.getvarname());
            else if (ctx->DIV())
                res = _builder.create_div(ir::Type::float_type(), op1, op2,
                                          _builder.getvarname());
            else
                res = _builder.create_rem(ir::Type::float_type(), op1, op2,
                                          _builder.getvarname());
        } else {
            // 需要进行隐式类型转换
            if (op1->is_int() && op2->is_float()) {
                auto sitof = _builder.create_sitof(ir::Type::float_type(), op1,
                                                   _builder.getvarname());
                if (ctx->MUL())
                    res = _builder.create_mul(ir::Type::float_type(), sitof,
                                              op2, _builder.getvarname());
                else if (ctx->DIV())
                    res = _builder.create_div(ir::Type::float_type(), sitof,
                                              op2, _builder.getvarname());
                else
                    res = _builder.create_rem(ir::Type::float_type(), sitof,
                                              op2, _builder.getvarname());
            } else {
                auto sitof = _builder.create_sitof(ir::Type::float_type(), op2,
                                                   _builder.getvarname());
                if (ctx->MUL())
                    res = _builder.create_mul(ir::Type::float_type(), op1,
                                              sitof, _builder.getvarname());
                else if (ctx->DIV())
                    res = _builder.create_div(ir::Type::float_type(), op1,
                                              sitof, _builder.getvarname());
                else
                    res = _builder.create_rem(ir::Type::float_type(), op1,
                                              sitof, _builder.getvarname());
            }
        }
    }
    return res;
}

/*
 * @brief Visit Additive Expression
 *      exp (ADD | SUB) exp
 */
std::any SysYIRGenerator::visitAdditiveExp(
    SysYParser::AdditiveExpContext* ctx) {
    ir::Value* op1 = any_cast_Value(visit(ctx->exp()[0]));
    ir::Value* op2 = any_cast_Value(visit(ctx->exp()[1]));
    ir::Value* res;

    if (ir::isa<ir::Constant>(op1) &&
        ir::isa<ir::Constant>(op2)) {  //! 1. 常量 -> 常量折叠
        ir::Constant* cop1 = ir::dyn_cast<ir::Constant>(op1);
        ir::Constant* cop2 = ir::dyn_cast<ir::Constant>(op2);
        if (cop1->is_float() || cop2->is_float()) {
            float sum, f1, f2;

            if (cop1->is_int())
                f1 = float(cop1->i());
            else
                f1 = cop1->f();
            if (cop2->is_int())
                f2 = float(cop2->i());
            else
                f2 = cop2->f();

            if (ctx->ADD())
                sum = f1 + f2;
            else
                sum = f1 - f2;
            res = ir::Constant::gen(sum, ir::getMC(sum));
        } else {
            int sum;
            if (ctx->ADD())
                sum = cop1->i() + cop2->i();
            else
                sum = cop1->i() - cop2->i();
            res = ir::Constant::gen(sum);
        }
    } else {  //! 2. 变量 -> 生成 ADD | fADD | SUB | fSUB 指令
        if (op1->is_int() && op2->is_int()) {
            // int32 类型加减
            if (ctx->SUB())
                res = _builder.create_sub(ir::Type::int_type(), op1, op2,
                                          _builder.getvarname());
            else
                res = _builder.create_add(ir::Type::int_type(), op1, op2,
                                          _builder.getvarname());
        } else if (op1->is_float() && op2->is_float()) {
            // float 类型加减
            if (ctx->SUB())
                res = _builder.create_sub(ir::Type::float_type(), op1, op2,
                                          _builder.getvarname());
            else
                res = _builder.create_add(ir::Type::float_type(), op1, op2,
                                          _builder.getvarname());
        } else {
            // 需要进行隐式类型转换 (int op float)
            if (op1->is_int()) {
                op1 = _builder.create_sitof(ir::Type::float_type(), op1,
                                            _builder.getvarname());
            }
            if (op2->is_int()) {
                op2 = _builder.create_sitof(ir::Type::float_type(), op2,
                                            _builder.getvarname());
            }
            if (ctx->SUB())
                res = _builder.create_sub(ir::Type::float_type(), op1, op2,
                                          _builder.getvarname());
            else
                res = _builder.create_add(ir::Type::float_type(), op1, op2,
                                          _builder.getvarname());
        }
    }
    return res;
}
//! exp (LT | GT | LE | GE) exp
std::any SysYIRGenerator::visitRelationExp(
    SysYParser::RelationExpContext* ctx) {
    auto lhsptr = any_cast_Value(visit(ctx->exp()[0]));
    auto rhsptr = any_cast_Value(visit(ctx->exp()[1]));
    ir::Value* res;
    bool isfloat = lhsptr->is_float() || rhsptr->is_float();
    if (isfloat) {
        if (lhsptr->is_int())
            lhsptr = _builder.create_sitof(ir::Type::float_type(), lhsptr,
                                           _builder.getvarname());
        if (rhsptr->is_int())
            rhsptr = _builder.create_sitof(ir::Type::float_type(), rhsptr,
                                           _builder.getvarname());
    }
    //! TODO
    if (ctx->GT()) {
        if (isfloat) {
            res = _builder.create_fogt(lhsptr, rhsptr, _builder.getvarname());
        } else {
            res = _builder.create_isgt(lhsptr, rhsptr, _builder.getvarname());
        }
    } else if (ctx->GE()) {
        if (isfloat) {
            res = _builder.create_foge(lhsptr, rhsptr, _builder.getvarname());
        } else {
            res = _builder.create_isge(lhsptr, rhsptr, _builder.getvarname());
        }
    } else if (ctx->LT()) {
        if (isfloat) {
            res = _builder.create_folt(lhsptr, rhsptr, _builder.getvarname());
        } else {
            res = _builder.create_islt(lhsptr, rhsptr, _builder.getvarname());
        }
    } else if (ctx->LE()) {
        if (isfloat) {
            res = _builder.create_fole(lhsptr, rhsptr, _builder.getvarname());
        } else {
            res = _builder.create_isle(lhsptr, rhsptr, _builder.getvarname());
        }
    }
    return res;
}

//! exp (EQ | NE) exp
std::any SysYIRGenerator::visitEqualExp(SysYParser::EqualExpContext* ctx) {
    //! TODO
    auto lhsptr = any_cast_Value(visit(ctx->exp()[0]));
    auto rhsptr = any_cast_Value(visit(ctx->exp()[1]));
    ir::Value* res;
    bool isfloat = lhsptr->is_float() || rhsptr->is_float();
    if (isfloat) {
        if (lhsptr->is_int()) {
            lhsptr = _builder.create_sitof(ir::Type::float_type(), lhsptr,
                                           _builder.getvarname());
        }
        if (rhsptr->is_int()) {
            rhsptr = _builder.create_sitof(ir::Type::float_type(), rhsptr,
                                           _builder.getvarname());
        }
    }
    if (ctx->EQ()) {
        if (isfloat) {
            res = _builder.create_foeq(lhsptr, rhsptr, _builder.getvarname());
        } else {
            res = _builder.create_ieq(lhsptr, rhsptr, _builder.getvarname());
        }
    } else if (ctx->NE()) {
        if (isfloat) {
            res = _builder.create_fone(lhsptr, rhsptr, _builder.getvarname());
        } else {
            res = _builder.create_ine(lhsptr, rhsptr, _builder.getvarname());
        }
    }
    return res;
}
/*
- before you visit one exp, you must prepare its true and false target
1. push the thing you protect
2. call the function
3. pop to reuse OR use tmp var to log
*/
//! exp : exp AND exp
// exp: lhs AND rhs

std::any SysYIRGenerator::visitAndExp(SysYParser::AndExpContext* ctx) {
    //! TODO
    // know exp's true/false target block
    // lhs's true target = rhs block
    // lhs's false target = exp false target
    // rhs's true target = exp true target
    // rhs's false target = exp false target
    auto cur_block = builder().block();
    auto cur_func = cur_block->parent();

    // create rhs block as lhs's true target
    // lhs's false target is exp false target
    auto rhs_block = cur_func->new_block();

    builder().push_tf(rhs_block, builder().false_target());  //! diff with OrExp
    // builder().push_true_target(rhs_block);
    // builder().push_false_target(builder().false_target());

    //! visit lhs exp to get its value
    auto lhs_value = any_cast_Value(visit(ctx->exp(0)));  // recursively visit
    //* cast to i1
    if (ir::isa<ir::ICmpInst>(lhs_value) || ir::isa<ir::FCmpInst>(lhs_value)) {
        // pass
        // do nothing
    } else if (lhs_value->is_int()) {
        lhs_value = builder().create_ine(lhs_value, ir::Constant::gen(0),
                                         builder().getvarname());
    } else if (lhs_value->is_float()) {
        lhs_value = builder().create_fone(lhs_value, ir::Constant::gen(0.0),
                                          builder().getvarname());
    }
    rhs_block->set_name(builder().getvarname());
    // pop to get lhs t/f target
    auto lhs_t_target = builder().true_target();
    auto lhs_f_target = builder().false_target();
    builder().pop_tf();  // match with push_tf

    // create cond br inst
    auto cond_br = builder().create_br(lhs_value, lhs_t_target, lhs_f_target);

    //! [for CFG] link cur_block and target
    // visit may change the cur_block, so need to reload the cur block
    cur_block = builder().block();

    ir::BasicBlock::block_link(cur_block, lhs_t_target);
    ir::BasicBlock::block_link(cur_block, lhs_f_target);

    //! [for CFG] link over

    //! visit and generate code for rhs block
    builder().set_pos(rhs_block, rhs_block->begin());

    auto rhs_value = visit(ctx->exp(1));

    return rhs_value;
}

//! exp OR exp
// lhs OR rhs
std::any SysYIRGenerator::visitOrExp(SysYParser::OrExpContext* ctx) {
    //! TODO
    // know exp's true/false target block (already in builder's stack)
    // lhs true target = exp true target
    // lhs false target = rhs block
    // rhs true target = exp true target
    // rhs false target = exp false target
    auto cur_block = builder().block();  // pre block
    auto cur_func = cur_block->parent();

    // create rhs block as lhs's false target
    // lhs's true target is exp true target
    auto rhs_block = cur_func->new_block();

    //! push target
    builder().push_tf(builder().true_target(), rhs_block);  // match with pop_tf

    //! visit lhs exp to get its value
    auto lhs_value = any_cast_Value(visit(ctx->exp(0)));  // recursively visit
    //* cast to i1
    if (ir::isa<ir::ICmpInst>(lhs_value) || ir::isa<ir::FCmpInst>(lhs_value)) {
        // pass
        // do nothing
    } else if (lhs_value->is_int()) {
        lhs_value = builder().create_ine(lhs_value, ir::Constant::gen(0),
                                         builder().getvarname());
    } else if (lhs_value->is_float()) {
        lhs_value = builder().create_fone(lhs_value, ir::Constant::gen(0.0),
                                          builder().getvarname());
    }
    rhs_block->set_name(builder().getvarname());

    //! pop to get lhs t/f target
    auto lhs_t_target = builder().true_target();
    auto lhs_f_target = builder().false_target();
    builder().pop_tf();  // match with push_tf

    // create condbr
    builder().create_br(lhs_value, lhs_t_target, lhs_f_target);
    //! lhs is finished,

    //! [for CFG] link cur_block and target
    // visit may change the cur_block, so need to reload the cur block
    cur_block = builder().block();  // after block

    ir::BasicBlock::block_link(cur_block, lhs_t_target);
    ir::BasicBlock::block_link(cur_block, lhs_f_target);

    //! [for CFG] link over

    //! visit and generate code for rhs block
    builder().set_pos(rhs_block, rhs_block->begin());

    auto rhs_value = visit(ctx->exp(1));

    return rhs_value;
}  // namespace sysy

}  // namespace sysy