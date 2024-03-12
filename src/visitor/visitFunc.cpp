#include "visitor.hpp"
#include <any>
#include <typeinfo>
using namespace std;

namespace sysy {
/**
 * @brief
 *
 * @param ctx
 * @return std::any
 *
 * funcType: VOID | INT | FLOAT;
 */
std::any SysYIRGenerator::visitFuncType(SysYParser::FuncTypeContext *ctx) {
    // std::cout << "visitFuncType" << std::endl;
    // std::cout << ctx->getText() << std::endl;

    // std::cout << "void: " << ctx->VOID() << std::endl;
    // std::cout << "int: " << ctx->INT() << std::endl;
    // std::cout << "float: " << ctx->FLOAT() << std::endl;

    if (ctx->INT()) {
        return ir::Type::int_type();
    } else if (ctx->FLOAT()) {
        return ir::Type::float_type();
    } else if (ctx->VOID()) {
        return ir::Type::void_type();
    }
    assert(false);
    return 0;
}

/**
 * @brief
 *
 * @param ctx
 * @return std::any
 *
 * func: funcType ID LPAREN funcFParams? RPAREN blockStmt;
 * ```
 * int func();
 * int func(int a);
 * int func(int a) {
 *  return a + 5;
 * }
 * ```
 */


std::any SysYIRGenerator::visitFunc(SysYParser::FuncContext *ctx) {
    _builder.func_add();
    std::string func_name = ctx->ID()->getText();
    std::vector<ir::Type *> param_types;
    std::vector<std::string> param_names;

    // if have formal params
    if (ctx->funcFParams()) {
        // funcFParams: funcFParam (COMMA funcFParam)*;
        auto params = ctx->funcFParams()->funcFParam();
        // cout << params->getText() << endl;
        // std::cout << typeid(params).name() << std::endl;
        for (auto param : params) {
            // std::cout << param->getText() << std::endl;
        }
    }
    // any_cast: cast any to whated type
    ir::Type *ret_type = any_cast_Type(visit(ctx->funcType()));

    // empty param types
    ir::Type *func_type = ir::Type::function_type(ret_type, {});
    // add func to module
    ir::Function *func = module()->add_function(true, func_type, func_name);

    // this->module()->insert??

    // create and enter function scope
    // it will be automatically destroyed when return from this visitFfunc
    ir::SymbolTableBeta::FunctionScope scope(_tables);
    // create entry block with the same params of func
    ir::BasicBlock *entry = func->add_block("entry_main");

    if (ctx->funcFParams()) { // has formal params
        // pass
    }

    _builder.set_position(entry, entry->end());
    visitBlockStmt(ctx->blockStmt());
    return func;
}

} // namespace sysy
