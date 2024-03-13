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
    auto btype = ir::Type::pointer_type(any_cast_Type(visit(ctx->btype())));

    bool is_const = ctx->CONST();

    for (auto varDef : ctx->varDef()) {
        visitVarDef_beta(varDef, btype, is_const);
    }
    return 0;
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
    auto alloca_ptr = _builder.create_alloca(btype, dims, name, is_const);
    // _builder.func();
    _tables.insert(name, alloca_ptr);  // check re decl err

    /// initValue
    ir::Value* init_value = nullptr;
    if (ctx->ASSIGN()) {         // parse initValue
        if (dims.size() == 0) {  // scalar
            init_value = any_cast_Value(visit(ctx->initValue()->exp()));
            // is Constant
            if (ir::isa<ir::Constant>(init_value)) {
                // assert(false);
                std::cout << "is const" << std::endl;
                assert(true);
            }
        }
        // int l = 5; 5 is a const value
        // inot_value =
        auto store = _builder.create_store(init_value, alloca_ptr, {}, "store");
        // return 0;
    }
}

std::any SysYIRGenerator::visitDeclGlobal(SysYParser::DeclContext* ctx) {
    return 0;
}

std::any SysYIRGenerator::visitBtype(SysYParser::BtypeContext* ctx) {
    return ctx->INT() ? ir::Type::int_type() : ir::Type::float_type();
}
}  // namespace sysy