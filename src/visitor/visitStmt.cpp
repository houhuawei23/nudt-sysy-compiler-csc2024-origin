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
        // res = any_cast_Value(visit(item));
        if (res = safe_any_cast<ir::Value>(visit(item))) {
            if (ir::isa<ir::ReturnInst>(res)) {
                break;
            } else {
                // TODO
            }
        }
        auto teststmt = item->stmt();
        if (teststmt) {
            auto bktest = teststmt->breakStmt();
            auto cttest = teststmt->continueStmt();
            if (bktest || cttest)
                break;
        }
        // 如果一个block中的stmt是break或者continue,那么后面的语句就可以不被翻译了
    }
    return res;
}

/*
 * @brief Visit ReturnStmt
 *      returnStmt: RETURN exp? SEMICOLON;
 */
std::any SysYIRGenerator::visitReturnStmt(SysYParser::ReturnStmtContext* ctx) {
    auto value = ctx->exp() ? any_cast_Value(visit(ctx->exp())) : nullptr;

    auto curr_block = builder().block();
    auto func = curr_block->parent();
    
    ir::Value* res = nullptr;
    assert(ir::isa<ir::Function>(func) && "ret stmt block parent err!");

    // 匹配 返回值类型 与 函数定义类型
    if (func->ret_type()->is_void()) {
        if (ctx->exp()) assert(false && "the returned value is not matching the function");
        else res = builder().create_return(value);
    } else {
        if (ctx->exp()) {
            if (auto cvalue = ir::dyn_cast<ir::Constant>(value)) {  //! 常值
                if (func->ret_type()->is_i32() && cvalue->is_float()) {
                    value = ir::Constant::gen_i32(cvalue->f64());
                } else if (func->ret_type()->is_float() && cvalue->is_i32()) {
                    value = ir::Constant::gen_f64(cvalue->i32());
                } else if (func->ret_type() != value->type()) {
                    assert(false);
                }
            } else {  //! 变量
                if (func->ret_type()->is_i32() && value->is_float()) {
                    value = _builder.create_ftosi(ir::Type::i32_type(), value, _builder.getvarname());
                } else if (func->ret_type()->is_float() && value->is_i32()) {
                    value = _builder.create_sitof(ir::Type::float_type(), value, _builder.getvarname());
                } else if (func->ret_type() != value->type()) {
                    assert(false && "invalid type");
                }
            }
        } else {
            assert(false && "the returned value is not matching the function");
        }
        
        // store res to re_value
        auto store = builder().create_store(value, func->ret_value_ptr());
        auto br = builder().create_br(func->exit());
        res = br;
        // res = builder().create_return(value);

    }
    
    // 生成 return 语句后立马创建一个新块，并设置 builder
    auto new_block = func->new_block();
    new_block->set_name(builder().get_bbname());
    builder().set_pos(new_block, new_block->begin());

    return res;
}

/*
 * @brief visit assign stmt
 * @details: 
 *      assignStmt: lValue ASSIGN exp SEMICOLON
 */
std::any SysYIRGenerator::visitAssignStmt(SysYParser::AssignStmtContext* ctx) {
    ir::Value* lvalue_ptr = any_cast_Value(visit(ctx->lValue()));  // 左值
    ir::Value* exp = safe_any_cast<ir::Value>(visit(ctx->exp()));  // 右值

    ir::Value* res = nullptr;

    if (auto cexp = ir::dyn_cast<ir::Constant>(exp)) {  //! 1. 右值为常值
        if (lvalue_ptr->is_i32() && cexp->is_float()) {
            exp = ir::Constant::gen_i32(cexp->f64());
        } else if (lvalue_ptr->is_float() && cexp->is_i32()) {
            exp = ir::Constant::gen_f64(cexp->i32());
        } else if (auto tmp = ir::dyn_cast<ir::PointerType>(lvalue_ptr->type())) {
            if (tmp->base_type() != cexp->type()) {
                std::cerr << "Type " << *cexp->type() << " can not convert to type " << *cexp->type() << std::endl;
                assert(false);
            }
        }
    } else {  //! 2. 右值为变量
        if (lvalue_ptr->is_i32() && exp->is_float()) {
            exp = _builder.create_ftosi(ir::Type::i32_type(), exp, _builder.getvarname());
        } else if (lvalue_ptr->is_float() && exp->is_i32()) {
            exp = _builder.create_sitof(ir::Type::float_type(), exp, _builder.getvarname());
        } else if (ir::dyn_cast<ir::PointerType>(lvalue_ptr->type())->base_type() != exp->type()) {
            std::cerr << "Type " << *exp->type()
                      << " can not convert to type " << *lvalue_ptr->type()
                      << std::endl;
            assert(false);
        }
    }
    res = _builder.create_store(exp, lvalue_ptr, "store");

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

// 短路求值？
// cond =
std::any SysYIRGenerator::visitIfStmt(SysYParser::IfStmtContext* ctx) {
    builder().if_inc();
    auto cur_block = builder().block();
    auto cur_func = cur_block->parent();

    auto then_block = cur_func->new_block();
    auto else_block = cur_func->new_block();
    auto merge_block = cur_func->new_block();

    // //* link the basic block
    // ir::BasicBlock::block_link(cur_block, then_block);
    // ir::BasicBlock::block_link(cur_block, else_block);

    //! VISIT cond
    //* push the then and else block to the stack
    builder().push_tf(then_block, else_block);

    //* visit cond, create icmp and br inst
    auto cond = safe_any_cast<ir::Value>(visit(ctx->exp()));  // load
    assert(cond != nullptr && "any_cast result nullptr!");
    //* pop to get lhs t/f target
    auto lhs_t_target = builder().true_target();
    auto lhs_f_target = builder().false_target();
    builder().pop_tf();  // match with push_tf

    //* cast to i1

    // if ()
    if (not cond->is_i1()) {
        if (cond->is_i32()) {
            cond = builder().create_ine(cond, ir::Constant::gen_i32(0),
                                        builder().getvarname());
        } else if (cond->is_float()) {
            cond = builder().create_fone(cond, ir::Constant::gen_f64(0.0),
                                         builder().getvarname());
        }
    }

    //* create cond br inst
    builder().create_br(cond, lhs_t_target, lhs_f_target);
    //! VISIT cond end

    //* link the basic block
    cur_block = builder().block();
    ir::BasicBlock::block_link(cur_block, lhs_t_target);
    ir::BasicBlock::block_link(cur_block, lhs_t_target);

    //! VISIT then block
    then_block->set_name(builder().get_bbname());
    builder().set_pos(then_block, then_block->begin());
    visit(ctx->stmt(0));  //* may change the basic block

    builder().create_br(merge_block);
    ir::BasicBlock::block_link(then_block, merge_block);

    //! VISIT else block
    else_block->set_name(builder().get_bbname());
    builder().set_pos(else_block, else_block->begin());
    if (auto elsestmt = ctx->stmt(1))
        visit(elsestmt);

    builder().create_br(merge_block);
    ir::BasicBlock::block_link(else_block, merge_block);
    //! SETUP builder fo merge block
    builder().set_pos(merge_block, merge_block->begin());
    merge_block->set_name(builder().get_bbname());

    return merge_block;
}

/*
这里的while的翻译:
while(judge_block){loop_block}
next_block
*/
std::any SysYIRGenerator::visitWhileStmt(SysYParser::WhileStmtContext* ctx) {
    //! TODO
    builder().while_inc();
    auto cur_block = builder().block();
    auto cur_func = cur_block->parent();
    // create new blocks
    auto next_block = cur_func->new_block();
    auto loop_block = cur_func->new_block();
    auto judge_block = cur_func->new_block();

    // set header and exit block
    builder().push_loop(judge_block, next_block);

    // jump without cond, directly jump to judge block
    builder().create_br(judge_block);
    judge_block->set_name(builder().get_bbname());
    builder().set_pos(judge_block, judge_block->begin());

    builder().push_tf(loop_block, next_block);
    auto cond = any_cast_Value((visit(ctx->exp())));

    auto tTarget = builder().true_target();
    auto fTarget = builder().false_target();
    builder().pop_tf();

    if (not cond->is_i1()) {
        if (cond->is_i32()) {
            cond = builder().create_ine(cond, ir::Constant::gen_i32(0),
                                        builder().getvarname());
        } else if (cond->is_float()) {
            cond = builder().create_fone(cond, ir::Constant::gen_f64(0.0),
                                         builder().getvarname());
        }
    }

    builder().create_br(cond, tTarget, fTarget);

    cur_block = builder().block();
    ir::BasicBlock::block_link(cur_block, tTarget);
    ir::BasicBlock::block_link(cur_block, fTarget);

    // visit loop block
    loop_block->set_name(builder().get_bbname());
    builder().set_pos(loop_block, loop_block->begin());
    visit(ctx->stmt());
    builder().create_br(judge_block);

    ir::BasicBlock::block_link(loop_block, judge_block);

    // pop header and exit block
    builder().pop_loop();

    // visit next block
    next_block->set_name(builder().get_bbname());
    builder().set_pos(next_block, next_block->begin());

    return next_block;
}

std::any SysYIRGenerator::visitBreakStmt(SysYParser::BreakStmtContext* ctx) {
    auto breakDest = builder().exit();
    if (not breakDest) {
        std::cerr << "Break stmt is not in a loop!" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto res = builder().create_br(breakDest);

    //create a basic block
    auto cur_func=builder().block()->parent();
    auto next_block=cur_func->new_block();
    next_block->set_name(builder().get_bbname());
    builder().set_pos(next_block,next_block->begin());
    return next_block;
}
std::any SysYIRGenerator::visitContinueStmt(
    SysYParser::ContinueStmtContext* ctx) {
    auto continueDest = builder().header();
    if (not continueDest) {
        std::cerr << "Break stmt is not in a loop!" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto res = builder().create_br(continueDest);
    //create a basic block
    auto cur_func=builder().block()->parent();
    auto next_block=cur_func->new_block();
    next_block->set_name(builder().get_bbname());
    builder().set_pos(next_block,next_block->begin());
    return res;
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

    if (ctx->exp().empty()) return res;  //! 1. scalar
    else {  //! 2. array
        if (auto res_array = ir::dyn_cast<ir::AllocaInst>(res)) {
            auto type = res_array->base_type();
            int current_dimension = 1;
            std::vector<ir::Value*> dims = res_array->dims();
            for (auto expr : ctx->exp()) {
                ir::Value* idx = any_cast_Value(visit(expr));
                res = _builder.create_getelementptr(type, res, 
                                                    idx, current_dimension, 
                                                    dims, _builder.getvarname(), 1);
                current_dimension++;
            }
        } else {
            assert(false && "this is not an array");
        }

        return res;
    }
}
}  // namespace sysy