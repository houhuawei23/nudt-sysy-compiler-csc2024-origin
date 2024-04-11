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
        res = ir::Constant::gen_f32(f);
    }
    return dyn_cast_Value(res);
}

/*
 * @brief: Visit Var Expression
 * @details:
 *      var: ID (LBRACKET exp RBRACKET)*;
 */
std::any SysYIRGenerator::visitVarExp(SysYParser::VarExpContext* ctx) {
    std::string varname = ctx->var()->ID()->getText();
    auto res = _tables.lookup(varname);
    assert(res && "use undefined variable");

    bool isArray = false;
    if (auto ptype = dyn_cast<ir::PointerType>(res->type())) {
        isArray = ptype->base_type()->is_array() || ptype->base_type()->is_pointer();
    }
    if (!isArray) {  //! 1. scalar
        if (ir::isa<ir::GlobalVariable>(res)) {  // 全局
            auto gres = dyn_cast<ir::GlobalVariable>(res);
            if (gres->is_const()) {  // 常量
                res = gres->scalar_value();
            } else {  // 变量
                res = _builder.create_load(res);
            }
        } else {  // 局部 (变量 - load, 常量 - ignore)
            if (!ir::isa<ir::Constant>(res)) {
                res = _builder.create_load(res);
            }
        }
    } else {  //! 2. array
        ir::Type* type = dyn_cast<ir::PointerType>(res->type())->base_type();
        if (type->is_array()) {  // 数组 (eg. int a[2][3]) -> 常规使用
            auto atype = dyn_cast<ir::ArrayType>(type);
            auto base_type = atype->base_type();
            std::vector<int> dims = atype->dims(), cur_dims(dims);

            int delta = dims.size() - ctx->var()->exp().size();
            for (auto expr : ctx->var()->exp()) {
                ir::Value* idx = any_cast_Value(visit(expr));
                dims.erase(dims.begin());
                res = _builder.create_getelementptr(base_type, res, idx, dims, cur_dims);
                cur_dims.erase(cur_dims.begin());
            }
            if (ctx->var()->exp().empty()) {
                dims.erase(dims.begin());
                res = _builder.create_getelementptr(base_type, res, ir::Constant::gen_i32(0), dims, cur_dims);
            } else if (delta == 1) {
                dims.erase(dims.begin());
                res = _builder.create_getelementptr(base_type, res, ir::Constant::gen_i32(0), dims, cur_dims);
            }

            if (delta == 0) res = _builder.create_load(res);
        } else if (type->is_pointer()) {  // 指针 (eg. int a[] OR int a[][5]) -> 函数参数
            res = _builder.create_load(res);
            type = dyn_cast<ir::PointerType>(type)->base_type();
            if (type->is_array()) {  // 二级及以上指针
                if (ctx->var()->exp().size()) {
                    auto expr_vec = ctx->var()->exp();

                    ir::Value* idx = any_cast_Value(visit(expr_vec[0]));
                    res = _builder.create_getelementptr(type, res, idx);
                    auto base_type = dyn_cast<ir::ArrayType>(type)->base_type();
                    std::vector<int> dims = dyn_cast<ir::ArrayType>(type)->dims(), cur_dims(dims);
                    int delta = dims.size() + 1 - expr_vec.size();
                    for (int i = 1; i < expr_vec.size(); i++) {
                        idx = any_cast_Value(visit(expr_vec[i]));
                        dims.erase(dims.begin());
                        res = _builder.create_getelementptr(base_type, res, idx, dims, cur_dims);
                        cur_dims.erase(cur_dims.begin());
                    }

                    if (delta == 0) res = _builder.create_load(res);
                }
            } else {  // 一级指针
                for (auto expr : ctx->var()->exp()) {
                    ir::Value* idx = any_cast_Value(visit(expr));
                    res = _builder.create_getelementptr(type, res, idx);
                }
                if (ctx->var()->exp().size()) res = _builder.create_load(res);
            }
        } else {
            assert(false && "type error");
        }
    }
    return dyn_cast_Value(res);
}

/*
 * @brief Visit Unary Expression
 * @details:
 *      + - ! exp
 */
std::any SysYIRGenerator::visitUnaryExp(SysYParser::UnaryExpContext* ctx) {
    ir::Value* res = nullptr;
    auto exp = any_cast_Value(visit(ctx->exp()));

    if (ctx->SUB()) {
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
                    res = builder().create_binary_beta(ir::Value::SUB, ir::Constant::gen_i32(0), exp);
                    break;
                case ir::FLOAT:
                    res = builder().create_unary_beta(ir::Value::vFNEG, exp);
                    break;
                default:
                    assert(false && "Unsupport btype");
            }
        } else {
            assert(false && "invalid value type");
        }
    } else if (ctx->NOT()) {
        auto true_target = builder().false_target();
        auto false_target = builder().true_target();
        builder().pop_tf();
        builder().push_tf(true_target, false_target);
        res = exp;
    } else if (ctx->ADD()) {
        res = exp;
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
    ir::Value* op1 = any_cast_Value(visit(ctx->exp(0)));
    ir::Value* op2 = any_cast_Value(visit(ctx->exp(1)));
    ir::Value* res;
    if (ir::isa<ir::Constant>(op1) && ir::isa<ir::Constant>(op2)) {  //! both constant (常数折叠)
        ir::Constant* cop1 = dyn_cast<ir::Constant>(op1);
        ir::Constant* cop2 = dyn_cast<ir::Constant>(op2);
        auto higher_Btype =
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
        op1 = builder().type_promote(op1, hi_type);  // base i32
        op2 = builder().type_promote(op2, hi_type);

        if (ctx->MUL()) {
            res = builder().create_binary_beta(ir::Value::MUL, op1, op2);
        } else if (ctx->DIV()) {
            res = builder().create_binary_beta(ir::Value::DIV, op1, op2);
        } else if (ctx->MODULO()) {
            res = builder().create_binary_beta(ir::Value::REM, op1, op2);
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
    ir::Value* lhs = any_cast_Value(visit(ctx->exp()[0]));
    ir::Value* rhs = any_cast_Value(visit(ctx->exp()[1]));
    ir::Value* res;

    if (ir::isa<ir::Constant>(lhs) && ir::isa<ir::Constant>(rhs)) {
        //! constant
        ir::Constant* clhs = dyn_cast<ir::Constant>(lhs);
        ir::Constant* crhs = dyn_cast<ir::Constant>(rhs);
        // bool is_ADD = ctx->ADD();

        auto higher_BType =
            std::max(clhs->type()->btype(), crhs->type()->btype());

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
        lhs = builder().type_promote(lhs, hi_type);  // base i32
        rhs = builder().type_promote(rhs, hi_type);

        if (ctx->ADD()) {
            res = builder().create_binary_beta(ir::Value::ADD, lhs, rhs);
        } else if (ctx->SUB()) {
            res = builder().create_binary_beta(ir::Value::SUB, lhs, rhs);
        } else {
            assert(false && "not support");
        }
    }

    return dyn_cast_Value(res);
}
//! exp (LT | GT | LE | GE) exp
std::any SysYIRGenerator::visitRelationExp(SysYParser::RelationExpContext* ctx) {
    auto lhsptr = any_cast_Value(visit(ctx->exp()[0]));
    auto rhsptr = any_cast_Value(visit(ctx->exp()[1]));

    ir::Value* res;

    auto hi_type = lhsptr->type();
    if (lhsptr->type()->btype() < rhsptr->type()->btype()) {
        hi_type = rhsptr->type();
    }
    lhsptr = builder().type_promote(lhsptr, hi_type);  // base i32
    rhsptr = builder().type_promote(rhsptr, hi_type);

    if (ctx->GT()) {
        res = builder().create_cmp(ir::Value::GT, lhsptr, rhsptr);
    } else if (ctx->GE()) {
        res = builder().create_cmp(ir::Value::GE, lhsptr, rhsptr);
    } else if (ctx->LT()) {
        res = builder().create_cmp(ir::Value::LT, lhsptr, rhsptr);
    } else if (ctx->LE()) {
        res = builder().create_cmp(ir::Value::LE, lhsptr, rhsptr);
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

    lhsptr = builder().type_promote(lhsptr, hi_type);  // base i32
    rhsptr = builder().type_promote(rhsptr, hi_type);

    // same type
    if (ctx->EQ()) {
        res = builder().create_cmp(ir::Value::EQ, lhsptr, rhsptr);
    } else if (ctx->NE()) {
        res = builder().create_cmp(ir::Value::NE, lhsptr, rhsptr);
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
    auto cur_func = builder().block()->parent();

    auto rhs_block = cur_func->new_block();
    rhs_block->append_comment("rhs");
    //! 1 visit lhs exp to get its value
    builder().push_tf(rhs_block,
                      builder().false_target());          //! diff with OrExp
    auto lhs_value = any_cast_Value(visit(ctx->exp(0)));  // recursively visit
    //! may chage by visit, need to re get
    auto lhs_t_target = builder().true_target();
    auto lhs_f_target = builder().false_target();
    builder().pop_tf();  // match with push_tf

    lhs_value = builder().cast_to_i1(lhs_value);

    builder().create_br(lhs_value, lhs_t_target, lhs_f_target);

    //! 2 [for CFG] link cur_block and target
    // visit may change the cur_block, so need to reload the cur block
    ir::BasicBlock::block_link(builder().block(), lhs_t_target);
    ir::BasicBlock::block_link(builder().block(), lhs_f_target);

    //! 3 visit and generate code for rhs block
    builder().set_pos(rhs_block, rhs_block->begin());
    rhs_block->set_name(builder().get_bbname());
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
    auto cur_func = builder().block()->parent();

    auto rhs_block = cur_func->new_block();
    rhs_block->append_comment("rhs");

    //! 1 visit lhs exp to get its value
    builder().push_tf(builder().true_target(), rhs_block);
    auto lhs_value = any_cast_Value(visit(ctx->exp(0)));
    auto lhs_t_target = builder().true_target();
    auto lhs_f_target = builder().false_target();
    builder().pop_tf();  // match with push_tf

    lhs_value = builder().cast_to_i1(lhs_value);
    builder().create_br(lhs_value, lhs_t_target, lhs_f_target);

    //! 2 [for CFG] link cur_block and target
    // visit may change the cur_block, so need to reload the cur block
    ir::BasicBlock::block_link(builder().block(), lhs_t_target);
    ir::BasicBlock::block_link(builder().block(), lhs_f_target);

    //! 3 visit and generate code for rhs block
    builder().set_pos(rhs_block, rhs_block->begin());
    rhs_block->set_name(builder().get_bbname());
    auto rhs_value = any_cast_Value(visit(ctx->exp(1)));

    return rhs_value;
}

}  // namespace sysy