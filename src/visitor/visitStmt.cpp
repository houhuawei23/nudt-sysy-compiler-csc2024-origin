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

    ir::Value* res = _builder.create_return(value);
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

//! ifStmt: IF LPAREN exp RPAREN stmt (ELSE stmt)?;
/*
if(exp) {} else {}
if(exp) {}
create true_block, false_block or
       then_block, else_block
如果是 exp = !exp, 则调换 true_block 和 false_block
的指针，正常地访问!右边的exp, 实际上实现了 ! NOT 操作。
*/
std::any SysYIRGenerator::visitIfStmt(SysYParser::IfStmtContext* ctx) {
    //! TODO
    builder().if_inc();
    auto cur_block = builder().block();
    auto cur_func = cur_block->parent();

    auto then_block = cur_func->add_block();
    auto else_block = cur_func->add_block();

    // link the basic block
    cur_block->add_next_block(then_block);
    cur_block->add_next_block(else_block);

    then_block->add_pre_block(cur_block);
    else_block->add_pre_block(cur_block);

    //! push the then and else block to the stack
    builder().push_true_target(then_block);
    builder().push_false_target(else_block);

    auto merge_block = cur_func->add_block();

    //! visit cond, create icmp and br inst
    builder().reset_not();
    auto cond = any_cast_Value(visit(ctx->exp()));  // load
    if (cond) {                                     //! visit
        if (cond->is_int()) {
            cond = builder().create_ine(cond, ir::Constant::gen(0),
                                        builder().getvarname());  // cur_block
        }
        if (cond->is_float()) {
            cond = builder().create_fone(cond, ir::Constant::gen(0.0),
                                         builder().getvarname());  // cur_block
        }
        if (builder().is_not()) {
            builder().create_br(cond, else_block, then_block);  // cur_block
        } else {
            builder().create_br(cond, then_block, else_block);  // cur_block
        }
    }
    
    //! then block
    builder().set_pos(then_block, then_block->end());
    then_block->set_name(builder().getvarname());
    visit(ctx->stmt(0));               // may change the basic block
    builder().create_br(merge_block);  

    //! else block
    builder().set_pos(else_block, else_block->end());
    else_block->set_name(builder().getvarname());
    if (auto elsestmt = ctx->stmt(1))
        visit(elsestmt);
    builder().create_br(merge_block);  

    //! merge block
    builder().set_pos(merge_block, merge_block->end());
    merge_block->set_name(builder().getvarname());
    // builder().
    return nullptr;
}

std::any SysYIRGenerator::visitWhileStmt(SysYParser::WhileStmtContext* ctx) {
    //! TODO
    return nullptr;
}

std::any SysYIRGenerator::visitBreakStmt(SysYParser::BreakStmtContext* ctx) {
    //! TODO
    return nullptr;
}

std::any SysYIRGenerator::visitContinueStmt(
    SysYParser::ContinueStmtContext* ctx) {
    //! TODO
    return nullptr;
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