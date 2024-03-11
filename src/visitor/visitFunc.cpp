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
    std::cout << "visitFunc" << std::endl;
    std::cout << ctx->getText() << std::endl;
    std::cout << ctx->ID()->getText() << std::endl;
    // visitChildren(ctx);
    _builder.func_add();
    std::string func_name = ctx->ID()->getText();
    // std::vector<Type*> paramTypes;
    // std::vector<std::string> paramNames;

    // if have formal params
    if (ctx->funcFParams()) {
        auto params = ctx->funcFParams()->funcFParam();
        // cout << params->getText() << endl;
        // std::cout << typeid(params).name() << std::endl;
        for (auto param : params) {
            // std::cout << param->getText() << std::endl;
        }
    }
    // any_cast: cast any to whated type
    ir::Type *ret_type =
        std::any_cast<ir::Type *>(visitFuncType(ctx->funcType()));
    // ir::Type * ret_type = visitFuncType(ctx->funcType());
    // empty param types
    ir::Type *func_type = ir::Type::function_type(ret_type, {});
    // add func to module
    ir::Function *func =
        this->get_module()->add_function(true, func_type, func_name);

    // this->get_module()->insert??

    // create and enter function scope
    // it will be automatically destroyed when return from this visitFfunc
    ir::SymbolTableBeta::FunctionScope scope(_tables);
    // create entry block with the same params of func
    ir::BasicBlock *entry = func->add_bblock("entry_main");

    if (ctx->funcFParams()) { // has formal params
        // pass
    }

    _builder.set_position(entry, entry->end());
    visitBlockStmt(ctx->blockStmt());
    return func;
}

} // namespace sysy
