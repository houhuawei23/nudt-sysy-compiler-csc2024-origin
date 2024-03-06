#include "visitor.hpp"
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
std::any
SysYIRGenerator::visitFunc(SysYParser::FuncContext* ctx)
{
    std::cout << "visitFunc" << std::endl;
    std::cout << ctx->getText() << std::endl;
    std::cout << ctx->ID()->getText() << std::endl;
    // visitChildren(ctx);

    auto func_name = ctx->ID()->getText();
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

    return 0;
}

} // namespace sysy
