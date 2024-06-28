#include "visitor/visitor.hpp"

namespace sysy {

/*
 * @brief visit BlockStmt
 * @details:
 *      blockStmt: LBRACE blockItem* RBRACE;
 *      blockItem: decl | stmt;
 *      stmt:
 *            assignStmt
 *          | expStmt
 *          | ifStmt
 *          | whileStmt
 *          | breakStmt
 *          | continueStmt
 *          | returnStmt
 *          | blockStmt
 *          | emptyStmt;
 */
std::any SysYIRGenerator::visitBlockStmt(SysYParser::BlockStmtContext* ctx) {
    ir::SymbolTableBeta::BlockScope scope(_tables);
    for (auto item : ctx->blockItem()) {
        visit(item);
        if (auto teststmt = item->stmt()) {
            // break, continue, return 后的语句不再翻译
            auto bk = teststmt->breakStmt();
            auto ct = teststmt->continueStmt();
            auto ret = teststmt->returnStmt();
            if (bk || ct || ret)
                break;
        }
    }
    return nullptr;
}

/*
 * @brief Visit ReturnStmt
 *      returnStmt: RETURN exp? SEMICOLON;
 */
std::any SysYIRGenerator::visitReturnStmt(SysYParser::ReturnStmtContext* ctx) {
    ir::Value* res = nullptr;
    auto value = ctx->exp() ? any_cast_Value(visit(ctx->exp())) : nullptr;

    auto func = builder().block()->parent();

    assert(ir::isa<ir::Function>(func) && "ret stmt block parent err!");

    // 匹配 返回值类型 与 函数定义类型
    if (func->ret_type()->is_void()) {
        if (ctx->exp())
            assert(false && "the returned value is not matching the function");
    } else {
        if (ctx->exp()) {
            if (auto cvalue = dyn_cast<ir::Constant>(value)) {  //! 常值
                if (func->ret_type()->is_i32() && cvalue->is_float()) {
                    value = ir::Constant::gen_i32(cvalue->f32());
                } else if (func->ret_type()->is_float() && cvalue->is_i32()) {
                    value = ir::Constant::gen_f32(cvalue->i32());
                } else if (func->ret_type() != cvalue->type()) {
                    assert(false && "invalid type");
                }
            } else {  //! 变量
                if (func->ret_type()->is_i32() && value->is_float()) {
                    value = _builder.create_unary_beta(
                        ir::ValueId::vFPTOSI, value, ir::Type::i32_type());
                } else if (func->ret_type()->is_float() && value->is_i32()) {
                    value = builder().create_unary_beta(
                        ir::ValueId::vSITOFP, value, ir::Type::float_type());
                } else if (func->ret_type() != value->type()) {
                    assert(false && "invalid type");
                }
            }
        } else {
            assert(false && "the returned value is not matching the function");
        }

        // store res to ret_value
        // auto store = builder().create_store(value, func->ret_value_ptr());
        auto store = builder().makeInst<ir::StoreInst>(
            value, func->ret_value_ptr(), builder().block());
    }
    // auto br = builder().create_br(func->exit());
    auto br = builder().makeInst<ir::BranchInst>(func->exit(), builder().block());
    ir::BasicBlock::block_link(builder().block(), func->exit());

    res = br;
    // 生成 return 语句后立马创建一个新块，并设置 builder
    auto new_block = func->new_block();

    new_block->set_idx(builder().get_bbidx());
    builder().set_pos(new_block, new_block->insts().begin());

    return dyn_cast_Value(res);
}

/*
 * @brief visit assign stmt
 * @details:
 *      assignStmt: lValue ASSIGN exp SEMICOLON
 */
std::any SysYIRGenerator::visitAssignStmt(SysYParser::AssignStmtContext* ctx) {
    ir::Value* lvalue_ptr = any_cast_Value(visit(ctx->lValue()));  // 左值
    ir::Type* lvar_pointee_type = nullptr;
    if (lvalue_ptr->type()->is_pointer())
        lvar_pointee_type =
            dyn_cast<ir::PointerType>(lvalue_ptr->type())->base_type();
    else
        lvar_pointee_type =
            dyn_cast<ir::ArrayType>(lvalue_ptr->type())->base_type();
    ir::Value* exp = any_cast_Value(visit(ctx->exp()));  // 右值

    ir::Value* res = nullptr;

    if (auto cexp = dyn_cast<ir::Constant>(exp)) {  //! 1. 右值为常值
        switch (lvar_pointee_type->btype()) {
            case ir::INT32:
                exp = ir::Constant::gen_i32(cexp->i32());
                break;
            case ir::FLOAT:
                exp = ir::Constant::gen_f32(cexp->f32());
                break;
            default:
                assert(false && "not valid btype");
        }
    } else {  //! 2. 右值为变量
        if (lvar_pointee_type == exp->type()) {
            exp = exp;
        } else if (lvar_pointee_type->is_i32() && exp->is_float32()) {
            exp = _builder.create_unary_beta(ir::ValueId::vFPTOSI, exp,
                                             ir::Type::i32_type());

        } else if (lvar_pointee_type->is_float32() && exp->is_i32()) {
            exp = builder().create_unary_beta(ir::ValueId::vSITOFP, exp,
                                              ir::Type::float_type());
        }
    }
    // res = _builder.create_store(exp, lvalue_ptr);
    res = builder().makeInst<ir::StoreInst>(exp, lvalue_ptr, builder().block());

    return dyn_cast_Value(res);
}

/*
 * @brief: visit If Stmt
 * @details:
 *      ifStmt: IF LPAREN exp RPAREN stmt (ELSE stmt)?;
 */
std::any SysYIRGenerator::visitIfStmt(SysYParser::IfStmtContext* ctx) {
    builder().if_inc();
    auto cur_block = builder().block();
    auto cur_func = cur_block->parent();

    auto then_block = cur_func->new_block();
    auto else_block = cur_func->new_block();
    auto merge_block = cur_func->new_block();

    then_block->append_comment("if" + std::to_string(builder().if_cnt()) +
                               "_then");
    else_block->append_comment("if" + std::to_string(builder().if_cnt()) +
                               "_else");
    merge_block->append_comment("if" + std::to_string(builder().if_cnt()) +
                                "_merge");

    {  //! 1. VISIT cond
        //* push the then and else block to the stack
        builder().push_tf(then_block, else_block);

        //* visit cond, create icmp and br inst
        auto cond = any_cast_Value(visit(ctx->exp()));  // load
        assert(cond != nullptr && "any_cast result nullptr!");
        //* pop to get lhs t/f target
        auto lhs_t_target = builder().true_target();
        auto lhs_f_target = builder().false_target();

        builder().pop_tf();

        cond = builder().cast_to_i1(cond);  //* cast to i1
        //* create cond br inst
        // builder().create_br(cond, lhs_t_target, lhs_f_target);
        builder().makeInst<ir::BranchInst>(cond, lhs_t_target, lhs_f_target, builder().block());

        //! [CFG] link the basic block
        ir::BasicBlock::block_link(builder().block(), lhs_t_target);
        ir::BasicBlock::block_link(builder().block(), lhs_f_target);
    }

    {  //! VISIT then block
        then_block->set_idx(builder().get_bbidx());
        builder().set_pos(then_block, then_block->insts().begin());
        visit(ctx->stmt(0));  //* may change the basic block

        // builder().create_br(merge_block);
        builder().makeInst<ir::BranchInst>(merge_block, builder().block());
        ir::BasicBlock::block_link(builder().block(), merge_block);
    }

    //! VISIT else block
    {
        else_block->set_idx(builder().get_bbidx());
        builder().set_pos(else_block, else_block->insts().begin());
        if (auto elsestmt = ctx->stmt(1))
            visit(elsestmt);

        // builder().create_br(merge_block);
        builder().makeInst<ir::BranchInst>(merge_block, builder().block());
        ir::BasicBlock::block_link(builder().block(), merge_block);
    }

    //! SETUP builder fo merge block
    builder().set_pos(merge_block, merge_block->insts().begin());
    merge_block->set_idx(builder().get_bbidx());

    return dyn_cast_Value(merge_block);
}

/*
这里的while的翻译:
while(judge_block){loop_block}
next_block
*/
std::any SysYIRGenerator::visitWhileStmt(SysYParser::WhileStmtContext* ctx) {
    builder().while_inc();
    auto cur_func = builder().block()->parent();
    // create new blocks
    auto next_block = cur_func->new_block();
    auto loop_block = cur_func->new_block();
    auto judge_block = cur_func->new_block();

    //! block comment
    next_block->append_comment("while" + std::to_string(builder().while_cnt()) +
                               "_next");
    loop_block->append_comment("while" + std::to_string(builder().while_cnt()) +
                               "_loop");
    judge_block->append_comment(
        "while" + std::to_string(builder().while_cnt()) + "_judge");

    // set header and exit block
    builder().push_loop(judge_block, next_block);

    // jump without cond, directly jump to judge block
    // builder().create_br(judge_block);
    builder().makeInst<ir::BranchInst>(judge_block, builder().block());
    ir::BasicBlock::block_link(builder().block(), judge_block);

    judge_block->set_idx(builder().get_bbidx());
    builder().set_pos(judge_block, judge_block->insts().begin());

    builder().push_tf(loop_block, next_block);
    auto cond = any_cast_Value((visit(ctx->exp())));

    auto tTarget = builder().true_target();
    auto fTarget = builder().false_target();
    builder().pop_tf();

    cond = builder().cast_to_i1(cond);
    // builder().create_br(cond, tTarget, fTarget);
    builder().makeInst<ir::BranchInst>(cond, tTarget, fTarget, builder().block());
    //! [CFG] link
    ir::BasicBlock::block_link(builder().block(), tTarget);
    ir::BasicBlock::block_link(builder().block(), fTarget);

    // visit loop block
    loop_block->set_idx(builder().get_bbidx());
    builder().set_pos(loop_block, loop_block->insts().begin());
    visit(ctx->stmt());

    // builder().create_br(judge_block);
    builder().makeInst<ir::BranchInst>(judge_block, builder().block());
    ir::BasicBlock::block_link(builder().block(), judge_block);

    // pop header and exit block
    builder().pop_loop();

    // visit next block
    next_block->set_idx(builder().get_bbidx());
    builder().set_pos(next_block, next_block->insts().begin());

    return dyn_cast_Value(next_block);
}

std::any SysYIRGenerator::visitBreakStmt(SysYParser::BreakStmtContext* ctx) {
    auto breakDest = builder().exit();
    if (not breakDest) {
        std::cerr << "Break stmt is not in a loop!" << std::endl;
        exit(EXIT_FAILURE);
    }
    // auto res = builder().create_br(breakDest);
    auto res = builder().makeInst<ir::BranchInst>(breakDest, builder().block());
    ir::BasicBlock::block_link(builder().block(), breakDest);

    // create a basic block
    auto cur_func = builder().block()->parent();
    auto next_block = cur_func->new_block();

    next_block->set_idx(builder().get_bbidx());
    builder().set_pos(next_block, next_block->insts().begin());
    return dyn_cast_Value(next_block);
}
std::any SysYIRGenerator::visitContinueStmt(
    SysYParser::ContinueStmtContext* ctx) {
    auto continueDest = builder().header();
    if (not continueDest) {
        std::cerr << "Break stmt is not in a loop!" << std::endl;
        exit(EXIT_FAILURE);
    }
    // auto res = builder().create_br(continueDest);
    auto res = builder().makeInst<ir::BranchInst>(continueDest, builder().block());
    ir::BasicBlock::block_link(builder().block(), continueDest);
    // create a basic block
    auto cur_func = builder().block()->parent();
    auto next_block = cur_func->new_block();

    next_block->set_idx(builder().get_bbidx());
    builder().set_pos(next_block, next_block->insts().begin());
    return dyn_cast_Value(res);
}

}  // namespace sysy