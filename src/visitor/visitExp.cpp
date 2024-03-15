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
        res = exp;
        builder().set_not();
    } else if (ctx->ADD()) {
        //
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

//! exp AND exp
std::any SysYIRGenerator::visitAndExp(SysYParser::AndExpContext* ctx) {
    //! TODO
    return nullptr;
}

//! exp OR exp
std::any SysYIRGenerator::visitOrExp(SysYParser::OrExpContext* ctx) {
    //! TODO
    return nullptr;
}  // namespace sysy

}  // namespace sysy