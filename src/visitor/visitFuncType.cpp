#include "visitor.hpp"

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
    std::cout << "visitFuncType" << std::endl;
    std::cout << ctx->getText() << std::endl;

    std::cout << "void: " << ctx->VOID() << std::endl;
    std::cout << "int: " << ctx->INT() << std::endl;
    std::cout << "float: " << ctx->FLOAT() << std::endl;

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

} // namespace sysy
