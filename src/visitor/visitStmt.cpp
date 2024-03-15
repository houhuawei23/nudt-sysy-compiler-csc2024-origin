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
    ir::Value* res = nullptr;
    
    for (auto item : ctx->blockItem()) {
        // visit(item);
        // res = any_cast_Value(visit(item));
        if(res = safe_any_cast<ir::Value>(visit(item))) {
            if(ir::isa<ir::ReturnInst>(res)) {
                // std::cout<< "Returninst!" << std::endl;
                break;
            } else {
            // std::cout<< "INNNN!" << std::endl;

            }

            // int a = 5;
            // int b = 5;
        }
        // if (ir::isa<ir::Value>(visit(item))) {
        //     res = any_cast_Value(visit(item));
        //     if (auto ires = ir::dyn_cast<ir::ReturnInst>(res)) {
        //         break;  // do not visit block item following return stmt
        //     }
        // }
        // auto res = any_cast_Value(visit(item));
        // if(auto ires = ir::dyn_cast<ir::ReturnInst>(res)) {
        //     break; // do not visit block item following return stmt
        // }

        // if (ir::isa<ir::Instruction>(res)) {
        //     auto ires = ir::dyn_cast<ir::Instruction>(res);
        //     if (ires->is_terminal()) {
        //         // pass?
        //     }
        // }
    }
    //
    return res;
}
std::any SysYIRGenerator::visitReturnStmt(SysYParser::ReturnStmtContext* ctx) {
    // returnStmt: RETURN exp? SEMICOLON;
    // TODO: return stmt terminate a block!! what does following code for?
    auto value = ctx->exp() ? any_cast_Value(visit(ctx->exp())) : nullptr;
    auto curr_block = builder().block();
    auto func = curr_block->parent();

    assert(ir::isa<ir::Function>(func) && "ret stmt block parent err!");

    // need type check and type transformation
    if (func->ret_type()->is_int() && value->is_float()) {
        //! TODO: : %r = fptosi float %v to i32
        // value = builder().create_float2int(value);
        assert(false && "not implement!");
    } else if (func->ret_type()->is_float() && value->is_int()) {
        //! TODO: create
        // value = builder().create_int2float(value);
        assert(false && "not implement!");

    } else if (func->ret_type() != value->type()) {
        //! TODO: type check
        assert(false && "return type check err!");
    }

    ir::Value* res = builder().create_return(value);
    // auto new_block = func->new_block();
    // new_block->set_name(builder().getvarname());
    // builder().set_pos(new_block, new_block->begin());
    return res;
}

std::any SysYIRGenerator::visitAssignStmt(SysYParser::AssignStmtContext* ctx) {
    ir::Value* lvalueptr = any_cast_Value(visit(ctx->lValue()));
    ir::Value* expptr = any_cast_Value(visit(ctx->exp()));
    ir::Value* res = nullptr;
    if (lvalueptr->is_int() && expptr->is_float()) {
        //! TODO f2i
    } else if (lvalueptr->is_float() && expptr->is_int()) {
        //! TODO i2f
    } else if (ir::dyn_cast<ir::PointerType>(lvalueptr->type())->base_type() !=
               expptr->type()) {
        std::cerr << "Type " << *expptr->type() << " can not convert to type "
                  << *lvalueptr->type() << std::endl;
        exit(EXIT_FAILURE);
    }
    res = builder().create_store(expptr, lvalueptr);

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
void block_link(ir::BasicBlock* pre, ir::BasicBlock* next) {
    pre->add_next_block(next);
    next->add_pre_block(pre);
}
// 短路求值？
std::any SysYIRGenerator::visitIfStmt(SysYParser::IfStmtContext* ctx) {
    builder().if_inc();
    auto cur_block = builder().block();
    auto cur_func = cur_block->parent();

    auto then_block = cur_func->new_block();
    auto else_block = cur_func->new_block();
    auto merge_block = cur_func->new_block();

    //* link the basic block
    // block_link(cur_block, then_block);
    // block_link(cur_block, else_block);

    //! VISIT cond
    //* push the then and else block to the stack
    builder().push_tf(then_block, else_block);

    //* visit cond, create icmp and br inst
    auto cond = any_cast_Value(visit(ctx->exp()));  // load

    //* pop to get lhs t/f target
    auto lhs_t_target = builder().true_target();
    auto lhs_f_target = builder().false_target();
    builder().pop_tf();  // match with push_tf

    //* cast to i1
    if (cond->is_int()) {
        cond = builder().create_ine(cond, ir::Constant::gen(0),
                                    builder().getvarname());
    } else if (cond->is_float()) {
        cond = builder().create_fone(cond, ir::Constant::gen(0.0),
                                     builder().getvarname());
    }
    //* create cond br inst
    builder().create_br(cond, lhs_t_target, lhs_f_target);
    //! VISIT cond end

    //! VISIT then block
    then_block->set_name(builder().getvarname());
    builder().set_pos(then_block, then_block->begin());
    visit(ctx->stmt(0));  //* may change the basic block
    builder().create_br(merge_block);

    //! VISIT else block
    else_block->set_name(builder().getvarname());
    builder().set_pos(else_block, else_block->begin());
    if (auto elsestmt = ctx->stmt(1))
        visit(elsestmt);
    builder().create_br(merge_block);

    //! SETUP builder fo merge block
    builder().set_pos(merge_block, merge_block->begin());
    merge_block->set_name(builder().getvarname());

    return merge_block;
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