#include "pass/optimize/GepSplit.hpp"

namespace pass {
void GepSplit::run(ir::Function* func, TopAnalysisInfoManager* tp) {
    if (func->isOnlyDeclare()) return;
    constexpr bool Debug = false;
    for (auto bb : func->blocks()) {
        auto& instructions = bb->insts();
        for (auto iter = instructions.begin(); iter != instructions.end();) {
            auto inst = *iter;
            if (Debug) {
                inst->print(std::cerr);
                std::cerr << std::endl;
            }
            if (inst->isa<ir::GetElementPtrInst>()) {
                auto gep_inst = dyn_cast<ir::GetElementPtrInst>(inst);
                int id = gep_inst->getid();
                auto begin = iter;
                if (id == 0) {
                    /* Pointer Op */
                    iter++;
                    split_pointer(gep_inst, bb, begin);
                } else {
                    /* Array Op */
                    auto end = iter; end++;
                    int num = 1;
                    while (end != instructions.end() && (*end)->isa<ir::GetElementPtrInst>()) {
                        auto preInst = std::prev(end);
                        auto curInst = dyn_cast<ir::GetElementPtrInst>(*end);
                        if (curInst->value() == (*preInst) && curInst->getid() != 0) {
                            end++; num++;
                        } else {
                            break;
                        }
                    }
                    iter = end;
                    split_array(begin, bb, end);  // [begin, end)
                }
            } else {
                iter++;
            }
        }
    }
}

/*
 * @brief: split_pointer function
 * @details: 
 *    <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
 *    ->  ptrtoint、add、inttoptr
 */
void GepSplit::split_pointer(ir::GetElementPtrInst* inst, 
                             ir::BasicBlock* insertBlock,
                             ir::inst_iterator insertPos) {
    ir::IRBuilder builder;
    builder.set_pos(insertBlock, insertPos);

    auto btype = inst->baseType();
    int stride = 1;
    if (btype->isArray()) {
        auto dims = dyn_cast<ir::ArrayType>(btype)->dims();
        for (int i = 0; i < dims.size(); i++) {
            stride *= dims[i];
        }
        btype = dyn_cast<ir::ArrayType>(btype)->baseType();
    }

    auto base = inst->value();
    auto index = inst->index();

    /* 1. 转换成int方便后续计算 */
    auto op_ptr = builder.makeUnary(ir::ValueId::vPTRTOINT, base, ir::Type::TypeInt64());

    /* 2. 计算偏移量 */
    ir::Value* offset = nullptr;
    if (index->isa<ir::ConstantValue>()) {
        auto cindex = index->dynCast<ir::ConstantValue>();
        offset = ir::ConstantInteger::gen_i64(cindex->i32() * 4 * stride);
    } else {
        auto index_64 = builder.makeUnary(ir::ValueId::vSEXT, index, ir::Type::TypeInt64());
        offset = builder.makeBinary(ir::BinaryOp::MUL, index_64, ir::ConstantInteger::gen_i64(4 * stride));
    }

    /* 3. 相加得到目标地址 */
    op_ptr = builder.makeBinary(ir::BinaryOp::ADD, op_ptr, offset);

    /* 4. 转换成ptr进行数据存储 */
    op_ptr = builder.makeUnary(ir::ValueId::vINTTOPTR, op_ptr, ir::Type::TypePointer(btype));

    /* 5. 替换 */
    inst->replaceAllUseWith(op_ptr);
    insertBlock->force_delete_inst(inst);
}

/*
 * @brief: split_array function
 * @details: 
 *    <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32 <idx>
 *    -> ptrtoint、add、inttoptr
 */
void GepSplit::split_array(ir::inst_iterator begin, 
                           ir::BasicBlock* insertBlock,
                           ir::inst_iterator end) {
    constexpr bool Debug = false;
    ir::IRBuilder builder;
    builder.set_pos(insertBlock, begin);

    if (Debug) {
        std::cerr << "Debug split_array: \n";
        auto iter = begin;
        while (iter != end) {
            (*iter)->print(std::cerr);
            std::cerr << "\n";
            iter++;
        }
    }

    auto inst = dyn_cast<ir::GetElementPtrInst>(*begin);
    auto btype = inst->baseType();
    if (btype->isArray()) {
        btype = dyn_cast<ir::ArrayType>(btype)->baseType();
    }

    auto iter = begin;

    /* 处理GetElementPtr指令 */
    while (iter != end) {
        inst = dyn_cast<ir::GetElementPtrInst>(*iter);

        auto dims = inst->cur_dims();
        auto base = inst->value();
        auto index = inst->index();

        bool is_constant = index->isa<ir::ConstantValue>();

        /* 1. 转换成int方便后续计算 */
        auto op_ptr = builder.makeUnary(ir::ValueId::vPTRTOINT, base, ir::Type::TypeInt64());

        /* 2. 计算偏移量 */
        /* 2.1 乘法运算 */
        for (int i = 1; i < dims.size(); i++) {
            if (is_constant) {
                auto index_constant = dyn_cast<ir::ConstantValue>(index);
                index = ir::ConstantInteger::gen_i64(index_constant->i32() * dims[i]);
            } else {
                if (!index->isInt64()) {
                    auto index_64 = builder.makeUnary(ir::ValueId::vSEXT, index, ir::Type::TypeInt64());
                    index = builder.makeBinary(ir::BinaryOp::MUL, index_64, ir::ConstantInteger::gen_i64(dims[i]));
                } else {
                    index = builder.makeBinary(ir::BinaryOp::MUL, index, ir::ConstantInteger::gen_i64(dims[i]));
                }
            }
        }

        /* 2.2 *4运算 */
        {
            if (is_constant) {
                auto index_constant = dyn_cast<ir::ConstantValue>(index);
                index = ir::ConstantInteger::gen_i64(index_constant->i64() * 4);
            } else {
                if (index->isInt64()) {
                    index = builder.makeBinary(ir::BinaryOp::MUL, index, ir::ConstantInteger::gen_i64(4));
                } else {
                    auto index_64 = builder.makeUnary(ir::ValueId::vSEXT, index, ir::Type::TypeInt64());
                    index = builder.makeBinary(ir::BinaryOp::MUL, index_64, ir::ConstantInteger::gen_i64(4));
                }
            }
        }

        /* 2.3 add运算 */
        {
            op_ptr = builder.makeBinary(ir::BinaryOp::ADD, op_ptr, index);
        }

        iter++;

        /* 3. 转换成ptr进行数据存储 */
        op_ptr = builder.makeUnary(ir::ValueId::vINTTOPTR, op_ptr, inst->type());

        /* 4. 替换 */
        inst->replaceAllUseWith(op_ptr);
    }

    /* 删除GetElementPtr指令 */
    iter = begin;
    while (iter != end) {
        inst = dyn_cast<ir::GetElementPtrInst>(*iter);
        iter++;
        insertBlock->force_delete_inst(inst);
    }
}
};