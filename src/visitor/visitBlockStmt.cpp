#include "visitor.hpp"

namespace sysy {

/**
 * @brief
 *
 * @param ctx
 * @return std::any
 *
 * blockStmt: LBRACE blockItem* RBRACE;
 */
std::any SysYIRGenerator::visitBlockStmt(SysYParser::BlockStmtContext *ctx) {
    std::cout << "visitBloclStmt" << std::endl;
    std::cout << ctx->getText() << std::endl;
    return 0;
}
} // namespace sysy