#include "value.hpp"
#include "visitor.hpp"
#include <any>
#include <vector>
namespace sysy {
std::any SysYIRGenerator::visitDecl(SysYParser::DeclContext *ctx) {
    _tables.isModuleScope();
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
std::any SysYIRGenerator::visitDeclLocal(SysYParser::DeclContext *ctx) {
    std::cout << ctx->getText() << std::endl;
    auto btype = ir::Type::pointer_type(
        std::any_cast<ir::Type *>(visitBtype(ctx->btype())));

    bool is_const = ctx->CONST();

    for (auto varDef : ctx->varDef()) {
        // lValue
        auto name = varDef->lValue()->ID()->getText();
        // if arr need to get dims
        std::vector<ir::Value *> dims;

        auto alloca_ptr =
            _builder.create_alloca_inst(btype, dims, name, is_const);
        // _builder.func();
        // _tables.insert(name, alloca); // check re decl err
        if (varDef->ASSIGN()) { // parse initValue
            // just scalar
            auto init_value =
                std::any_cast<ir::Value *>(visit(varDef->initValue()->exp()));
            // int l = 5; 5 is a const value
            // inot_value =
            auto store =
                _builder.create_store_inst(init_value, alloca_ptr, {}, "store");
            return 0;
        }
    }
    return 0;
}

std::any SysYIRGenerator::visitDeclGlobal(SysYParser::DeclContext *ctx) {
    return 0;
}

std::any SysYIRGenerator::visitBtype(SysYParser::BtypeContext *ctx) {
    return ctx->INT() ? ir::Type::int_type() : ir::Type::float_type();
}
} // namespace sysy