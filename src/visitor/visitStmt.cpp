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
    // std::cout << "visitBloclStmt" << std::endl;
    // std::cout << ctx->getText() << std::endl;

    ir::SymbolTableBeta::BlockScope scope(_tables);
    // visit children item
    for (auto item : ctx->blockItem()) {
        visitBlockItem(item);
    }
    //
    return 0;
}
std::any SysYIRGenerator::visitReturnStmt(SysYParser::ReturnStmtContext *ctx) {
    // returnStmt: RETURN exp? SEMICOLON;

    auto value = ctx->exp() ? any_cast_Value(visit(ctx->exp())) : nullptr;
    auto curr_block = _builder.block();
    auto func = curr_block->parent();

    if (value) {
        // just for `ret i32 0`
        // value = ir::Constant::get((int)(dynamic_cast<Constant
        // *>(value)->int()))
        value = value;
    }
    ir::Value *res = _builder.create_return(value, curr_block);
    return res;
}

} // namespace sysy