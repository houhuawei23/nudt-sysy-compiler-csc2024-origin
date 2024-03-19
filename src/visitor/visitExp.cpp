#include "visitor.hpp"

namespace sysy {
/*
 * @brief Visit Number Expression
 *      exp (MUL | DIV | MODULO) exp
 * @details
 *      number: ILITERAL | FLITERAL; (即: int or float)
 */
std::any SysYIRGenerator::visitNumberExp(SysYParser::NumberExpContext* ctx) {
    ir::Value* res = nullptr;
    if (auto iLiteral = ctx->number()->ILITERAL()) {  //! int
            std::string s = iLiteral->getText();
            
            //! 基数 (8, 10, 16)
            int base = 10;
            if (s.length() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) base = 16;
            else if (s[0] == '0') base = 8;

        res = ir::Constant::gen_i32(std::stoi(s, 0, base));
    } else if (auto fctx = ctx->number()->FLITERAL()) {  //! float
        std::string s = fctx->getText();
        float f = std::stof(s);
        res = ir::Constant::gen_f64(f);
    }
    return res;
}

std::any SysYIRGenerator::visitVarExp(SysYParser::VarExpContext* ctx) {
    bool isarray = not ctx->var()->LBRACKET().empty();
    std::string varname = ctx->var()->ID()->getText();

    auto res = _tables.lookup(varname);
    if (res == nullptr) {
        std::cerr << "Use undefined variable: \"" << varname << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!isarray) {  //! 1. scalar
        if (auto cres = ir::dyn_cast<ir::Constant>(res)) {
        } else {
            res = _builder.create_load(res, {}, _builder.getvarname());
        }
    } else {  //! 2. array
        if (auto res_array = ir::dyn_cast<ir::AllocaInst>(res)) {
            auto type = res_array->base_type();
            int current_dimension = 1;
            std::vector<ir::Value*> dims = res_array->dims();
            for (auto expr : ctx->var()->exp()) {
                ir::Value* idx = any_cast_Value(visit(expr));
                res = _builder.create_getelementptr(type, res, 
                                                    idx, current_dimension, 
                                                    dims, _builder.getvarname(), 1);
                current_dimension++;
            }
            res = _builder.create_load(res, {}, _builder.getvarname());
        } else {
            assert(false && "this is not an array");
            // res = _builder.create_load(res, {}, _builder.getvarname());
        }
    }
    return res;
}

/*
 * @brief Visit Unary Expression
 *      + - (NOT) exp
 */
std::any SysYIRGenerator::visitUnaryExp(SysYParser::UnaryExpContext* ctx) {
    ir::Value* res = nullptr;
    auto exp = any_cast_Value(visit(ctx->exp()));

    if (ctx->SUB()) {
        //! Constant, static type cast
        // alloca arr
        if (auto cexp = ir::dyn_cast<ir::Constant>(exp)) {
            //
            if (exp->is_i32()) {
                res = ir::Constant::gen_i32(-cexp->i32());
            } else if (exp->is_float()) {
                res = ir::Constant::gen_f64(-cexp->f64());
            }
        } else if (ir::isa<ir::LoadInst>(exp)) {
            // 如果是 AllocaInst and 标量
            if (exp->is_i32()) {
                // create sub 0
                res = builder().create_sub(ir::Type::i32_type(),
                                           ir::Constant::gen_i32(0), exp,
                                           builder().getvarname());
            } else if (exp->is_float()) {
                // fneg
                res = builder().create_fneg(ir::Type::double_type(), exp,
                                            builder().getvarname());
            } else {
                assert(false && "not known type!");
            }
        } else if (exp->is_i32()) {
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
    ir::Value* op1 = any_cast_Value(visit(ctx->exp(0)));
    ir::Value* op2 = any_cast_Value(visit(ctx->exp(1)));
    ir::Value* res;
    if (ir::isa<ir::Constant>(op1) &&
        ir::isa<ir::Constant>(op2)) {  //! 1. both 常量 -> 常量折叠

        ir::Constant* cop1 = ir::dyn_cast<ir::Constant>(op1);
        ir::Constant* cop2 = ir::dyn_cast<ir::Constant>(op2);
        if (ctx->DIV()) {
            auto ans = (cop1->is_float() ? cop1->f64() : cop1->i32()) /
                       (cop2->is_float() ? cop2->f64() : cop2->i32());
            if (typeid(ans) == typeid(float))
                res = ir::Constant::gen_f64(ans);
            else
                res = ir::Constant::gen_i32(ans);
        } else if (ctx->MUL()) {
            auto ans = (cop1->is_float() ? cop1->f64() : cop1->i32()) *
                       (cop2->is_float() ? cop2->f64() : cop2->i32());
            if (typeid(ans) == typeid(float))
                res = ir::Constant::gen_f64(ans);
            else
                res = ir::Constant::gen_i32(ans);
        } else {  // MODULO
            if (cop1->is_i32() && cop2->is_i32()) {
                int ans = cop1->i32() % cop2->i32();
                res = ir::Constant::gen_i32(ans);
            } else {
                std::cerr << "Operands of modulo must be integer!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    } else {  //! 2. 变量 -> 生成 MUL | FMUL | UDIV | SDIV | FDIV | UREM | SREM
              //! | FREM 指令

        if (op1->type() == op2->type()) {  // same type
            auto type = op1->type();

            if (ctx->MUL())
                res =
                    _builder.create_mul(type, op1, op2, _builder.getvarname());
            else if (ctx->DIV())
                res =
                    _builder.create_div(type, op1, op2, _builder.getvarname());
            else
                res =
                    _builder.create_rem(type, op1, op2, _builder.getvarname());
        } else if (op1->is_i32() && op2->is_float()) {  // 需要进行隐式类型转换
            auto ftype = ir::Type::float_type();
            auto sitof =
                _builder.create_sitof(ftype, op1, _builder.getvarname());
            if (ctx->MUL())
                res = _builder.create_mul(ftype, sitof, op2,
                                          _builder.getvarname());
            else if (ctx->DIV())
                res = _builder.create_div(ftype, sitof, op2,
                                          _builder.getvarname());
            else
                res = _builder.create_rem(ftype, sitof, op2,
                                          _builder.getvarname());
        } else if (op1->is_float() && op2->is_i32()) {  // 需要进行隐式类型转换
            auto ftype = ir::Type::float_type();
            auto sitof =
                _builder.create_sitof(ftype, op2, _builder.getvarname());
            if (ctx->MUL())
                res = _builder.create_mul(ftype, op1, sitof,
                                          _builder.getvarname());
            else if (ctx->DIV())
                res = _builder.create_div(ftype, op1, sitof,
                                          _builder.getvarname());
            else
                res = _builder.create_rem(ftype, op1, sitof,
                                          _builder.getvarname());
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
    auto ftype = ir::Type::double_type();

    if (ir::isa<ir::Constant>(op1) &&
        ir::isa<ir::Constant>(op2)) {  //! 1. 常量 -> 常量折叠
        ir::Constant* cop1 = ir::dyn_cast<ir::Constant>(op1);
        ir::Constant* cop2 = ir::dyn_cast<ir::Constant>(op2);

        if (cop1->is_float() || cop2->is_float()) {
            float sum, f1, f2;

            if (cop1->is_i32())
                f1 = float(cop1->i32());
            else
                f1 = cop1->f64();

            if (cop2->is_i32())
                f2 = float(cop2->i32());
            else
                f2 = cop2->f64();

            if (ctx->ADD())
                sum = f1 + f2;
            else
                sum = f1 - f2;

            res = ir::Constant::gen_f64(sum);
        } else {  // both int
            int sum;
            if (ctx->ADD())
                sum = cop1->i32() + cop2->i32();
            else
                sum = cop1->i32() - cop2->i32();
            res = ir::Constant::gen_i32(sum);
        }
    } else {  //! 2. 变量 -> 生成 ADD | fADD | SUB | fSUB 指令
        if (op1->is_i32() && op2->is_i32()) {
            // int32 类型加减
            if (ctx->SUB())
                res = _builder.create_sub(ir::Type::i32_type(), op1, op2,
                                          _builder.getvarname());
            else
                res = _builder.create_add(ir::Type::i32_type(), op1, op2,
                                          _builder.getvarname());
        } else if (op1->is_float() && op2->is_float()) {
            // float 类型加减
            if (ctx->SUB())
                res =
                    _builder.create_sub(ftype, op1, op2, _builder.getvarname());
            else
                res =
                    _builder.create_add(ftype, op1, op2, _builder.getvarname());
        } else {
            // 需要进行隐式类型转换 (int op float)
            if (op1->is_i32()) {
                op1 = _builder.create_sitof(ftype, op1, _builder.getvarname());
            }
            if (op2->is_i32()) {
                op2 = _builder.create_sitof(ftype, op2, _builder.getvarname());
            }
            if (ctx->SUB())
                res =
                    _builder.create_sub(ftype, op1, op2, _builder.getvarname());
            else
                res =
                    _builder.create_add(ftype, op1, op2, _builder.getvarname());
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
        if (lhsptr->is_i32())
            lhsptr = _builder.create_sitof(ir::Type::float_type(), lhsptr,
                                           _builder.getvarname());
        if (rhsptr->is_i32())
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
        if (lhsptr->is_i32()) {
            lhsptr = _builder.create_sitof(ir::Type::float_type(), lhsptr,
                                           _builder.getvarname());
        }
        if (rhsptr->is_i32()) {
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

    if (not lhs_value->is_i1()) {
        if (lhs_value->is_i32()) {
            // better wrap it to a simple method
            lhs_value = builder().create_ine(
                lhs_value, ir::Constant::gen_i32(0), builder().getvarname());
        } else if (lhs_value->is_float()) {
            lhs_value = builder().create_fone(
                lhs_value, ir::Constant::gen_f64(0.0), builder().getvarname());
        }
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

    if (not lhs_value->is_i1()) {
        if (lhs_value->is_i32()) {
            // better wrap it to a simple method
            lhs_value = builder().create_ine(
                lhs_value, ir::Constant::gen_i32(0), builder().getvarname());
        } else if (lhs_value->is_float()) {
            lhs_value = builder().create_fone(
                lhs_value, ir::Constant::gen_f64(0.0), builder().getvarname());
        }
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
}

}  // namespace sysy