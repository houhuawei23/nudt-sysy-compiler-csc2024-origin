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
std::any SysYIRGenerator::visitBlockStmt(SysYParser::BlockStmtContext* ctx) {
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
std::any SysYIRGenerator::visitReturnStmt(SysYParser::ReturnStmtContext* ctx) {
    // returnStmt: RETURN exp? SEMICOLON;

    auto value = ctx->exp() ? any_cast_Value(visit(ctx->exp())) : nullptr;
    auto curr_block = _builder.block();
    auto func = curr_block->parent();

    assert(ir::isa<ir::Function>(func) && "ret stmt block parent err!");

    // need type check and type transformation
    if (func->ret_type()->is_int() && value->is_float()) {
        //! TODO: : %r = fptosi float %v to i32
        // value = _builder.create_float2int(value);
        assert(false && "not implement!");
    } else if (func->ret_type()->is_float() && value->is_int()) {
        //! TODO: create
        // value = _builder.create_int2float(value);
        assert(false && "not implement!");

    } else if (func->ret_type() != value->type()) {
        //! TODO: type check
        assert(false && "return type check err!");
    }

    ir::Value* res = _builder.create_return(value, curr_block);
    return res;
}

std::any SysYIRGenerator::visitAssignStmt(SysYParser::AssignStmtContext* ctx) {
    ir::Value* lvalueptr = any_cast_Value(visit(ctx->lValue()));
    ir::Value* expptr = any_cast_Value(visit(ctx->exp()));
    ir::Value* res = nullptr;
    if(lvalueptr->is_int() && expptr->is_float()){
        //!TODO f2i
    }
    else if(lvalueptr->is_float() && expptr->is_int()){
        //!TODO i2f
    }
    else if(ir::dyn_cast<ir::PointerType>(lvalueptr->type())->base_type()!=expptr->type()){
        std::cerr<<"Type "<<*expptr->type()<<" can not convert to type "<<*lvalueptr->type()<<std::endl;
        exit(EXIT_FAILURE);
    }
    res = _builder.create_store(expptr, lvalueptr);  

    return res;
}

std::any SysYIRGenerator::visitLValue(SysYParser::LValueContext* ctx) {
    std::string name = ctx->ID()->getText();
    ir::Value* res = _tables.lookup(name);
    if (res == nullptr) {
        std::cerr << "Use undefined variable: \"" << name << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
    bool isscalar = ctx->exp().empty();
    if (isscalar)
        return res;
    else {
        // if lvalue is not a scalar
        // TODO!
        // pass
        return res;
    }
}
}  // namespace sysy