#include "visitor.hpp"

namespace sysy {
std::any SysYIRGenerator::visitNumberExp(SysYParser::NumberExpContext* ctx) {
    // number: ILITERAL | FLITERAL;

    ir::Value* res = nullptr;
    if (auto iLiteral = ctx->number()->ILITERAL()) {  // int
        std::string s = iLiteral->getText();
        // dec
        int base = 10;
        // hex
        if (s.length() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
            base = 16;

        } else if (s[0] == '0') {  // oct
            base = 8;
        }

        res = ir::Constant::gen(std::stoi(s, 0, base));
    } else if (auto fctx = ctx->number()->FLITERAL()) {  // float
        std::string s = fctx->getText();
        float f = std::stof(s);

        res = ir::Constant::gen(f, ir::getMC(f));  // stod?
        // change to machine code when print
        // didn't realize hexadecimal floating numbers
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
    if (not isarray) {  // scalar
        res = _builder.create_load(valueptr, {}, _builder.getvarname());
    } else {  // array element
        // pass
    }
    return res;
}
/*
SUB: 如果 exp：
- 为常量 Constant：生成 Constant(-exp)
- 是标量 AllocaInst: 获得？？
-
*/
std::any SysYIRGenerator::visitUnaryExp(SysYParser::UnaryExpContext* ctx) {
    //! TODO
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

std::any SysYIRGenerator::visitMultiplicativeExp(
    SysYParser::MultiplicativeExpContext* ctx) {
    ir::Value* op1 = any_cast_Value(visit(ctx->exp()[0]));
    ir::Value* op2 = any_cast_Value(visit(ctx->exp()[1]));
    ir::Value* res;
    if (ir::isa<ir::Constant>(op1) &&
        ir::isa<ir::Constant>(op2)) {  // constant folding
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
    } else {
        // res=_builder.create_binary()
    }
    return res;
}

/*
 * @brief Visit Additive Expression
 *      exp ADD exp
 */
std::any SysYIRGenerator::visitAdditiveExp(SysYParser::AdditiveExpContext* ctx) {
    //! Generate operand (得到操作数)
    ir::Value* adder1 = any_cast_Value(visit(ctx->exp()[0]));
    ir::Value* adder2 = any_cast_Value(visit(ctx->exp()[1]));
    ir::Value* res;

    bool isAdd = ctx->ADD() != nullptr;

    if (ir::isa<ir::Constant>(adder1) && ir::isa<ir::Constant>(adder2)) {
        ir::Constant* cadder1 = ir::dyn_cast<ir::Constant>(adder1);
        ir::Constant* cadder2 = ir::dyn_cast<ir::Constant>(adder2);
        if (cadder1->is_float() || cadder2->is_float()) {
            float sum;
            float f1, f2;
            
            if (cadder1->is_int()) f1 = float(cadder1->i());
            else f1 = cadder1->f();
            
            if (cadder2->is_int()) f2 = float(cadder2->i());
            else f2 = cadder2->f();
            
            sum = f1 + f2;
            
            if (isAdd) sum = f1 + f2;

            res = ir::Constant::gen(sum, ir::getMC(sum));

            // auto sub = _builder.create_binary()
        } else {  // just int
            int sum;
            sum = cadder1->i() + cadder2->i();
            if (isAdd) sum = cadder1->i() + cadder2->i();
            res = ir::Constant::gen(sum);

            auto add = _builder.create_add(cadder1, cadder2);
        }
    } else {
        // res=_builder.create_binary(add)
    }
    return res;
}
//! exp (LT | GT | LE | GE) exp
std::any SysYIRGenerator::visitRelationExp(
    SysYParser::RelationExpContext* ctx) {
    //! TODO
    return nullptr;
}

//! exp (EQ | NE) exp
std::any SysYIRGenerator::visitEqualExp(SysYParser::EqualExpContext* ctx) {
    //! TODO
    return nullptr;
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
    if (lhs_value->is_int()) {
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
    cur_block = builder().block();  // after block
    // link cur_block and target
    cur_block->add_next_block(lhs_t_target);
    cur_block->add_next_block(lhs_f_target);

    lhs_t_target->add_pre_block(cur_block);
    lhs_f_target->add_pre_block(cur_block);
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
    // builder().push_true_target(builder().true_target());
    // builder().push_false_target(rhs_block);
    builder().push_tf(builder().true_target(), rhs_block);  // match with pop_tf

    //! visit lhs exp to get its value
    auto lhs_value = any_cast_Value(visit(ctx->exp(0)));  // recursively visit
    //* cast to i1
    if (lhs_value->is_int()) {
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

    cur_block->add_next_block(lhs_t_target);
    cur_block->add_next_block(lhs_f_target);

    lhs_t_target->add_pre_block(cur_block);
    lhs_f_target->add_pre_block(cur_block);
    //! [for CFG] link over

    //! visit and generate code for rhs block
    builder().set_pos(rhs_block, rhs_block->begin());

    auto rhs_value = visit(ctx->exp(1));

    return rhs_value;
}  // namespace sysy

}  // namespace sysy