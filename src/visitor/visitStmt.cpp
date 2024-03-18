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
        if(res = safe_any_cast<ir::Value>(visit(item))) {
            if(ir::isa<ir::ReturnInst>(res)) {
                break;
            } else {
                // TODO
            }
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

    assert(ir::isa<ir::Function>(func) && "ret stmt block parent err!");

    // 匹配 返回值类型 与 函数定义类型
    if (func->ret_type()->is_void()) {
        if (ctx->exp()) assert(false);
        else {
            std::cout << "void return" << std::endl;
            auto res = builder().create_return(value);
            return res;
        }
    } else {
        if (ctx->exp()) {
            if (auto cvalue = ir::dyn_cast<ir::Constant>(value)) {
                if (func->ret_type()->is_i32() && cvalue->is_float()) {
                    value = ir::Constant::gen_i32(cvalue->f64());
                } else if (func->ret_type()->is_float() && cvalue->is_i32()) {
                    value = ir::Constant::gen_f64(cvalue->i32());
                } else if (func->ret_type() != value->type()) {
                    assert(false);
                }
            } else {
                if (func->ret_type()->is_i32() && value->is_float()) {
                    value = _builder.create_ftosi(ir::Type::i32_type(), value, _builder.getvarname());
                } else if (func->ret_type()->is_float() && value->is_i32()) {
                    value = _builder.create_sitof(ir::Type::float_type(), value, _builder.getvarname());
                } else if (func->ret_type() != value->type()) {
                    assert(false);
                }
            }
        } else {
            assert(false);
        }
        auto res = builder().create_return(value);
        return res;
    }
}

std::any SysYIRGenerator::visitAssignStmt(SysYParser::AssignStmtContext* ctx) {
    ir::Value* lvalueptr = any_cast_Value(visit(ctx->lValue()));
    auto expptr = any_cast_Value(visit(ctx->exp()));
    ir::Value* res = nullptr;

    if (auto res = ir::dyn_cast<ir::Constant>(expptr)) {
        if (lvalueptr->is_i32() && res->is_float()) {
            expptr = ir::Constant::gen_i32(res->f64());
        } else if (lvalueptr->is_float() && res->is_i32()) {
            expptr = ir::Constant::gen_f64(res->i32());
        } else if (ir::dyn_cast<ir::PointerType>(lvalueptr->type())->base_type() != res->type()) {
            std::cerr << "Type " << *res->type() << " can not convert to type " << *res->type() << std::endl;
            assert(false);
        }
    } else {
        if (lvalueptr->is_i32() && expptr->is_float()) {
            expptr = _builder.create_ftosi(ir::Type::i32_type(), expptr, _builder.getvarname());
        } else if (lvalueptr->is_float() && expptr->is_i32()) {
            expptr = _builder.create_sitof(ir::Type::float_type(), expptr, _builder.getvarname());
        } else if (ir::dyn_cast<ir::PointerType>(lvalueptr->type())->base_type() != expptr->type()) {
            std::cerr << "Type " << *expptr->type() << " can not convert to type " << *lvalueptr->type() << std::endl;
            assert(false);
        }
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
    auto cond = any_cast_Value(visit(ctx->exp()));  // load

    //* pop to get lhs t/f target
    auto lhs_t_target = builder().true_target();
    auto lhs_f_target = builder().false_target();
    builder().pop_tf();  // match with push_tf

    //* cast to i1

    // if ()
    if(not cond->is_i1()) {
        if (cond->is_i32()) {
            cond = builder().create_ine(cond, ir::Constant::gen_i32(0), builder().getvarname());
        } else if (cond->is_float()) {
            cond = builder().create_fone(cond, ir::Constant::gen_f64(0.0), builder().getvarname());
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
    then_block->set_name(builder().getvarname());
    builder().set_pos(then_block, then_block->begin());
    visit(ctx->stmt(0));  //* may change the basic block

    builder().create_br(merge_block);
    ir::BasicBlock::block_link(then_block, merge_block);
    
    //! VISIT else block
    else_block->set_name(builder().getvarname());
    builder().set_pos(else_block, else_block->begin());
    if (auto elsestmt = ctx->stmt(1))
        visit(elsestmt);

    builder().create_br(merge_block);
    ir::BasicBlock::block_link(else_block, merge_block);
    //! SETUP builder fo merge block
    builder().set_pos(merge_block, merge_block->begin());
    merge_block->set_name(builder().getvarname());

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
    auto cur_func=cur_block->parent();

    auto next_block=cur_func->new_block();
    auto loop_block=cur_func->new_block();
    auto judge_block=cur_func->new_block();


    // jump without cond, directly jump to judge block
    builder().create_br(judge_block);
    judge_block->set_name(builder().getvarname());
    builder().set_pos(judge_block,judge_block->begin());


    builder().push_tf(loop_block,next_block);
    auto cond=any_cast_Value((visit(ctx->exp())));

    auto tTarget=builder().true_target();
    auto fTarget=builder().false_target();
    builder().pop_tf();


    if(not cond->is_i1()) {
        if (cond->is_i32()) {
            cond = builder().create_ine(cond, 
                                            ir::Constant::gen_i32(0),
                                            builder().getvarname());
        } else if (cond->is_float()) {
            cond = builder().create_fone(cond, 
                                            ir::Constant::gen_f64(0.0),
                                            builder().getvarname());
        }
    }

    builder().create_br(cond, tTarget, fTarget);

    cur_block=builder().block();
    ir::BasicBlock::block_link(cur_block, tTarget);
    ir::BasicBlock::block_link(cur_block, fTarget);

    //visit loop block
    loop_block->set_name(builder().getvarname());
    builder().set_pos(loop_block,loop_block->begin());
    visit(ctx->stmt());
    builder().create_br(judge_block);
    
    ir::BasicBlock::block_link(loop_block,judge_block);

    //visit next block
    next_block->set_name(builder().getvarname());
    builder().set_pos(next_block,next_block->begin());
    
    return next_block;
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