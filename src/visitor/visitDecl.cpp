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
    // auto test = safe_any_cast<ir::Type>(visit(ctx->btype()));
    // assert(test == btype);
    bool is_const = ctx->CONST();
    ir::Value* res = nullptr;

    for (auto varDef : ctx->varDef()) {
        res = visitVarDef_beta(varDef, btype, is_const);
    }
    return res;
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
    | var						# varExp
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
ir::Value* SysYIRGenerator::visitVarDef_beta(SysYParser::VarDefContext* ctx,
                                             ir::Type* btype,
                                             bool is_const) {
    auto name = ctx->lValue()->ID()->getText();

    // array
    std::vector<ir::Value*> dims;
    for (auto dim : ctx->lValue()->exp()) {
        dims.push_back(any_cast_Value(visit(dim)));
    }
    
    if(is_const) {  //! const
        if (dims.size() == 0) {  //! 1.1 标量
            if(ctx->ASSIGN()) {
                auto init = any_cast_Value(visit(ctx->initValue()->exp()));
                if(auto cinit = ir::dyn_cast<ir::Constant>(init)) {
                    if (btype->is_int() && cinit->is_float()) {
                        cinit = ir::Constant::gen((int)cinit->f());
                    } else if (btype->is_float() && cinit->is_int()) {
                        cinit = ir::Constant::gen((float)cinit->i(), ir::getMC((float)cinit->i()));
                    }
                    _tables.insert(name, cinit);
                    return cinit;
                } else {
                    //TODO
                }
            } else {
                std::cerr << "const without initialization!" << std::endl;
                exit(EXIT_FAILURE);
            }
        } else {  //! 1.2 数组
            // TODO
        }
        return nullptr;
    } else {  //! 非const
        //! create alloca inst
        auto alloca_ptr = _builder.create_alloca(btype, dims, _builder.getvarname(), is_const);
        _tables.insert(name, alloca_ptr);

        //! create store inst
        ir::Value* init = nullptr;
        if (ctx->ASSIGN()) {
            if (dims.size() == 0) {  //! 1. scalar
                init = any_cast_Value(visit(ctx->initValue()->exp()));
                if (auto cinit = ir::dyn_cast<ir::Constant>(init)) {  //! 1.1 常量
                    if (btype->is_int() && init->is_float()) {
                        init = ir::Constant::gen((int)cinit->f());
                    } else if (btype->is_float() && init->is_int()) {  // i2f
                        init = ir::Constant::gen((float)cinit->i(),ir::getMC((float)cinit->i()));
                    }
                    init = ir::dyn_cast<ir::Constant>(init);
                    auto store = _builder.create_store(init, alloca_ptr, {}, "store");
                } else {  //! 1.2 变量
                    if (init->is_float() && btype->is_int()) {
                        auto ftosi = _builder.create_ftosi(ir::Type::int_type(), init, _builder.getvarname());
                        auto stroe = _builder.create_store(ftosi, alloca_ptr, {}, "store");
                    } else if (init->is_int() && btype->is_float()) {
                        auto sitof = _builder.create_sitof(ir::Type::float_type(), init, _builder.getvarname());
                        auto store = _builder.create_store(sitof, alloca_ptr, {}, "store");
                    }
                }
            } else {  //! 2. array
                // TODO
            }
        }
        return alloca_ptr;
    }
}

// decl: CONST? btype varDef (COMMA varDef)* SEMICOLON;
std::any SysYIRGenerator::visitDeclGlobal(SysYParser::DeclContext* ctx) {
    // std::cout << ctx->getText() << std::endl;
    auto btype = ir::Type::pointer_type(any_cast_Type(visit(ctx->btype())));

    bool is_const = ctx->CONST();
    ir::Value* res = nullptr;
    for (auto varDef : ctx->varDef()) {
        res = visitVarDef_beta(varDef, btype, is_const);
    }

    return res;
}

std::any SysYIRGenerator::visitBtype(SysYParser::BtypeContext* ctx) {
    return ctx->INT() ? ir::Type::int_type() : ir::Type::float_type();
}
}  // namespace sysy