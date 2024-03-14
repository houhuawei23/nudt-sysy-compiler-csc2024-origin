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

    //need type check and type transformation

    if (value) {
        // just for `ret i32 0`
        // value = ir::Constant::get((int)(dynamic_cast<Constant
        // *>(value)->int()))
        value = value;
    }
    ir::Value *res = _builder.create_return(value, curr_block);
    return res;
}

std::any SysYIRGenerator::visitAssignStmt(SysYParser::AssignStmtContext* ctx){
    ir::Value* lvalueptr=any_cast_Value(visit(ctx->lValue()));
    ir::Value* expptr=any_cast_Value(visit(ctx->exp()));
    ir::Value* res=nullptr;
    res=_builder.create_store(expptr,lvalueptr);//didn'e realize type check
    
    return res;
}

std::any SysYIRGenerator::visitLValue(SysYParser::LValueContext *ctx){
    std::string name=ctx->ID()->getText();
    ir::Value* res=_tables.lookup(name);
    if(res==nullptr){
        std::cerr<<"Use undefined variable: \""<<name<<"\""<<std::endl;
        exit(EXIT_FAILURE);
    }
    bool isscalar=ctx->exp().empty();
    if(isscalar)return res;
    else{
        // if lvalue is not a scalar
        //TODO!
        //pass
        return res;
    }
}
} // namespace sysy