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

    ir::SymbolTableBeta::BlockScope scope(_tables);
    // visit children item
    for (auto item: ctx->blockItem()) {
        visitBlockItem(item);
    }
    // 
    return 0;
}
} // namespace sysy