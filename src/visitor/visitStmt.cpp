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
            if (bk || ct || ret) break;
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
        else
            res = builder().create_return(value);
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
                    value =_builder.create_unary_beta(ir::Value::vFPTOSI, value, ir::Type::i32_type());
                } else if (func->ret_type()->is_float() && value->is_i32()) {
                    value = builder().create_unary_beta(ir::Value::vSITOFP, value, ir::Type::float_type());
                } else if (func->ret_type() != value->type()) {
                    assert(false && "invalid type");
                }
            }
        } else {
            assert(false && "the returned value is not matching the function");
        }

        // store res to ret_value
        auto store = builder().create_store(value, func->ret_value_ptr());
        auto br = builder().create_br(func->exit());
        ir::BasicBlock::block_link(builder().block(), func->exit());
        
        res = br;
    }

    // 生成 return 语句后立马创建一个新块，并设置 builder
    auto new_block = func->new_block();
    new_block->set_name(builder().get_bbname());
    builder().set_pos(new_block, new_block->begin());

    return dyn_cast_Value(res);
}

/*
 * @brief visit assign stmt
 * @details:
 *      assignStmt: lValue ASSIGN exp SEMICOLON
 */
std::any SysYIRGenerator::visitAssignStmt(SysYParser::AssignStmtContext* ctx) {
    ir::Value* lvalue_ptr = any_cast_Value(visit(ctx->lValue()));  // 左值
    ir::Value* exp = any_cast_Value(visit(ctx->exp()));            // 右值

    ir::Value* res = nullptr;

    if (auto cexp = dyn_cast<ir::Constant>(exp)) {  //! 1. 右值为常值
        if (lvalue_ptr->is_i32() && cexp->is_float()) {
            exp = ir::Constant::gen_i32(cexp->f32());
        } else if (lvalue_ptr->is_float() && cexp->is_i32()) {
            exp = ir::Constant::gen_f32(cexp->i32());
        } else if (auto tmp = dyn_cast<ir::PointerType>(lvalue_ptr->type())) {
            if (tmp->base_type() != cexp->type()) {
                std::cerr << "Type " << *cexp->type()
                          << " can not convert to type " << *cexp->type()
                          << std::endl;
                assert(false);
            }
        }
    } else {  //! 2. 右值为变量
        if (lvalue_ptr->is_i32() && exp->is_float()) {
            exp = _builder.create_unary_beta(ir::Value::vFPTOSI, exp, ir::Type::i32_type());
        } else if (lvalue_ptr->is_float() && exp->is_i32()) {
            exp = builder().create_unary_beta(ir::Value::vSITOFP, exp, ir::Type::float_type());
        } else if (dyn_cast<ir::PointerType>(lvalue_ptr->type())->base_type() !=
                   exp->type()) {
            std::cerr << "Type " << *exp->type() << " can not convert to type "
                      << *lvalue_ptr->type() << std::endl;
            assert(false);
        }
    }
    res = _builder.create_store(exp, lvalue_ptr, "store");

    return dyn_cast_Value(res);
}

/*
 * @brief visit If Stmt
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

    then_block->append_comment("if" + std::to_string(builder().if_cnt()) + "_then");
    else_block->append_comment("if" + std::to_string(builder().if_cnt()) + "_else");
    merge_block->append_comment("if" + std::to_string(builder().if_cnt()) + "_merge");

    {   //! 1. VISIT cond
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
        builder().create_br(cond, lhs_t_target, lhs_f_target);

        //! [CFG] link the basic block
        ir::BasicBlock::block_link(builder().block(), lhs_t_target);
        ir::BasicBlock::block_link(builder().block(), lhs_f_target);
    }

    {  //! VISIT then block
        then_block->set_name(builder().get_bbname());
        builder().set_pos(then_block, then_block->begin());
        visit(ctx->stmt(0));  //* may change the basic block

        builder().create_br(merge_block);
        ir::BasicBlock::block_link(builder().block(), merge_block);
    }

    //! VISIT else block
    {
        else_block->set_name(builder().get_bbname());
        builder().set_pos(else_block, else_block->begin());
        if (auto elsestmt = ctx->stmt(1))
            visit(elsestmt);

        builder().create_br(merge_block);
        ir::BasicBlock::block_link(builder().block(), merge_block);
    }

    //! SETUP builder fo merge block
    builder().set_pos(merge_block, merge_block->begin());
    merge_block->set_name(builder().get_bbname());

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
    next_block->append_comment("while" + std::to_string(builder().while_cnt()) + "_next");
    loop_block->append_comment("while" + std::to_string(builder().while_cnt()) + "_loop");
    judge_block->append_comment("while" + std::to_string(builder().while_cnt()) + "_judge");

    // set header and exit block
    builder().push_loop(judge_block, next_block);

    // jump without cond, directly jump to judge block
    builder().create_br(judge_block);
    ir::BasicBlock::block_link(builder().block(), judge_block);

    judge_block->set_name(builder().get_bbname());
    builder().set_pos(judge_block, judge_block->begin());

    builder().push_tf(loop_block, next_block);
    auto cond = any_cast_Value((visit(ctx->exp())));

    auto tTarget = builder().true_target();
    auto fTarget = builder().false_target();
    builder().pop_tf();

    cond = builder().cast_to_i1(cond);
    builder().create_br(cond, tTarget, fTarget);
    //! [CFG] link
    ir::BasicBlock::block_link(builder().block(), tTarget);
    ir::BasicBlock::block_link(builder().block(), fTarget);

    // visit loop block
    loop_block->set_name(builder().get_bbname());
    builder().set_pos(loop_block, loop_block->begin());
    visit(ctx->stmt());

    builder().create_br(judge_block);
    ir::BasicBlock::block_link(builder().block(), judge_block);

    // pop header and exit block
    builder().pop_loop();

    // visit next block
    next_block->set_name(builder().get_bbname());
    builder().set_pos(next_block, next_block->begin());

    return dyn_cast_Value(next_block);
}

std::any SysYIRGenerator::visitBreakStmt(SysYParser::BreakStmtContext* ctx) {
    auto breakDest = builder().exit();
    if (not breakDest) {
        std::cerr << "Break stmt is not in a loop!" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto res = builder().create_br(breakDest);
    ir::BasicBlock::block_link(builder().block(), breakDest);

    // create a basic block
    auto cur_func = builder().block()->parent();
    auto next_block = cur_func->new_block();
    next_block->set_name(builder().get_bbname());
    builder().set_pos(next_block, next_block->begin());
    return dyn_cast_Value(next_block);
}
std::any SysYIRGenerator::visitContinueStmt(
    SysYParser::ContinueStmtContext* ctx) {
    auto continueDest = builder().header();
    if (not continueDest) {
        std::cerr << "Break stmt is not in a loop!" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto res = builder().create_br(continueDest);
    ir::BasicBlock::block_link(builder().block(), continueDest);
    // create a basic block
    auto cur_func = builder().block()->parent();
    auto next_block = cur_func->new_block();
    next_block->set_name(builder().get_bbname());
    builder().set_pos(next_block, next_block->begin());
    return dyn_cast_Value(res);
}

/*
 * @brief visit lvalue
 * @details:
 *      lValue: ID (LBRACKET exp RBRACKET)*
 */
std::any SysYIRGenerator::visitLValue(SysYParser::LValueContext* ctx) {
    std::string name = ctx->ID()->getText();
    ir::Value* res = _tables.lookup(name);

    if (res == nullptr) {
        std::cerr << "Use undefined variable: \"" << name << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (ctx->exp().empty()) {
        //! 1. scalar
        return dyn_cast_Value(res);

    } else {  //! 2. array
        if (auto res_array = dyn_cast<ir::AllocaInst>(res)) {
            auto type = res_array->base_type();
            int current_dimension = 1;
            std::vector<ir::Value*> dims = res_array->dims();
            for (auto expr : ctx->exp()) {
                ir::Value* idx = any_cast_Value(visit(expr));
                res = _builder.create_getelementptr(type, res, idx,
                                                    current_dimension, dims, 1);
                current_dimension++;
            }
        } else if (auto res_array = dyn_cast<ir::GlobalVariable>(res)) {
            auto type = res_array->base_type();
            int current_dimension = 1;
            std::vector<ir::Value*> dims = res_array->dims();
            for (auto expr : ctx->exp()) {
                ir::Value* idx = any_cast_Value(visit(expr));
                res = _builder.create_getelementptr(type, res, idx,
                                                    current_dimension, dims, 1);
                current_dimension++;
            }
        } else {
            assert(false && "this is not an array");
        }
    }

    return dyn_cast_Value(res);
}
}  // namespace sysy