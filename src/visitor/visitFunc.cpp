#include "visitor.hpp"

namespace sysy {
/**
 * @brief
 *
 * @param ctx
 * @return std::any
 *
 * func: funcType ID LPAREN funcFParams? RPAREN blockStmt;
 */
std::any
SysYIRGenerator::visitFunc(SysYParser::FuncContext* ctx)
{
    std::cout << "visitFunc" << std::endl;
    std::cout << ctx->getText() << std::endl;
    std::cout << ctx->ID()->getText() << std::endl;
    visitChildren(ctx);
    return 0;
}

} // namespace sysy
