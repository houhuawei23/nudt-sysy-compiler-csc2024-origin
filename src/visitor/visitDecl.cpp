#include "visitor.hpp"

namespace sysy {
std::any SysYIRGenerator::visitDecl(SysYParser::DeclContext* ctx) {
    return _tables.isModuleScope() ? visitDeclGlobal(ctx) : visitDeclLocal(ctx);
}
/**
 * @brief
 * decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
 * varDef: lValue (ASSIGN initValue)?;
 * lValue: ID (LBRACKET exp RBRACKET)*;
 * initValue: exp | LBRACE (initValue (COMMA initValue)*)? RBRACE;
 * @param ctx
 * @return std::any
 */
std::any SysYIRGenerator::visitDeclLocal(SysYParser::DeclContext* ctx) {
    // std::cout << ctx->getText() << std::endl;
    // auto btype_pointer_type =
    // ir::Type::pointer_type(any_cast_Type(visit(ctx->btype())));
    auto btype = any_cast_Type(visit(ctx->btype()));

    bool is_const = ctx->CONST();

    for (auto varDef : ctx->varDef()) {
        visitVarDef_beta(varDef, btype, is_const);
    }
    return nullptr;
}

// varDef: lValue (ASSIGN initValue)?;
// lValue: ID (LBRACKET exp RBRACKET)*;
// int a = 5;
// %a = alloca i32
// store i32 5, i32* %a

// int a[5] = {1, 2, 3}
/*
exp:
    LPAREN exp RPAREN				# parenExp
    | lValue						# lValueExp
    | number						# numberExp
    | string						# stringExp
    | call							# callExp
    | (ADD | SUB | NOT) exp			# unaryExp
    | exp (MUL | DIV | MODULO) exp	# multiplicativeExp
    | exp (ADD | SUB) exp			# additiveExp
    | exp (LT | GT | LE | GE) exp	# relationExp
    | exp (EQ | NE) exp				# equalExp
    | exp AND exp					# andExp
    | exp OR exp					# orExp;
*/
void SysYIRGenerator::visitVarDef_beta(SysYParser::VarDefContext* ctx,
                                       ir::Type* btype,
                                       bool is_const) {
    /// lValue
    auto name = ctx->lValue()->ID()->getText();
    // if arr need to get dims
    std::vector<ir::Value*> dims;
    for (auto dim : ctx->lValue()->exp()) {
        dims.push_back(any_cast_Value(visit(dim)));
    }
    //! create alloca inst
    auto ptr_type = ir::Type::pointer_type(btype);
    auto alloca_ptr = _builder.create_alloca(btype, dims, name, is_const);
    // _builder.func();
    _tables.insert(name, alloca_ptr);  // check re decl err

    /// initValue
    ir::Value* init = nullptr;
    if (ctx->ASSIGN()) {         // parse initValue
        if (dims.size() == 0) {  // scalar
            init = any_cast_Value(visit(ctx->initValue()->exp()));
            // init is Constant
            //! if init is Constant, do dynamic_cast; else return nullptr
            //! if init is not Constant, generate i2f/f2i inst
            if (auto cinit = ir::dyn_cast<ir::Constant>(init)) {
                // if const, may do implicit conversion
                if (btype->is_int() && init->is_float()) {  // f2i
                    init = ir::Constant::gen((int)cinit->f());
                } else if (btype->is_float() && init->is_int()) {  // i2f
                    init = ir::Constant::gen((float)cinit->i());
                }
            } else if (btype->is_float() && init->is_int()) {  // i2f
                //! TODO
                // init = _builder.create_i2f(init);
            } else if (btype->is_int() && init->is_float()) {  // f2i
                //! TODO
                // init = _builder.create_f2i(init);
            }
        }

        auto store = _builder.create_store(init, alloca_ptr, {}, "store");
    }
}

// decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
std::any SysYIRGenerator::visitDeclGlobal(SysYParser::DeclContext* ctx) {
    // std::cout << ctx->getText() << std::endl;
    auto btype = ir::Type::pointer_type(any_cast_Type(visit(ctx->btype())));

    bool is_const = ctx->CONST();

    for (auto varDef : ctx->varDef()) {
        visitVarDef_beta(varDef, btype, is_const);
    }

    return 0;
}

std::any SysYIRGenerator::visitBtype(SysYParser::BtypeContext* ctx) {
    return ctx->INT() ? ir::Type::int_type() : ir::Type::float_type();
}
}  // namespace sysy