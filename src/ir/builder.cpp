#include "ir/builder.hpp"
namespace ir {

Value* IRBuilder::create_binary_beta(BinaryOp op, Value* lhs, Value* rhs) {
    Type *ltype = lhs->type(), *rtype = rhs->type();
    if (ltype != rtype) {
        assert(false && "create_eq_beta: type mismatch!");
    }
    Value* res = nullptr;
    ValueId vid;
    switch (ltype->btype()) {
        case INT32: {
            switch (op) {
                case BinaryOp::ADD:
                    vid = ValueId::vADD;
                    break;
                case BinaryOp::SUB:
                    vid = ValueId::vSUB;
                    break;
                case BinaryOp::MUL:
                    vid = ValueId::vMUL;
                    break;
                case BinaryOp::DIV:
                    vid = ValueId::vSDIV;
                    break;
                case BinaryOp::REM:
                    vid = ValueId::vSREM;
                    break;
                default:
                    assert(false && "create_binary_beta: invalid op!");
            }
            res = makeInst<BinaryInst>(vid, Type::i32_type(), lhs, rhs, _block);
        } break;
        case FLOAT: {
            switch (op) {
                case BinaryOp::ADD:
                    vid = ValueId::vFADD;
                    break;
                case BinaryOp::SUB:
                    vid = ValueId::vFSUB;
                    break;
                case BinaryOp::MUL:
                    vid = ValueId::vFMUL;
                    break;
                case BinaryOp::DIV:
                    vid = ValueId::vFDIV;
                    break;
                default:
                    assert(false && "create_binary_beta: invalid op!");
            }
            res =
                makeInst<BinaryInst>(vid, Type::float_type(), lhs, rhs, _block);
        } break;
        case DOUBLE: {
            assert(false && "create_binary_beta: invalid type!");
        }
    }
    return res;
}

Value* IRBuilder::create_unary_beta(ValueId vid, Value* val, Type* ty) {
    //! check vid
    Value* res = nullptr;

    if (vid == ValueId::vFNEG) {
        assert(val->type()->is_float() && "fneg must have float operand");
        // res = create_unary(ValueId::vFNEG, Type::float_type(), val);
        res = makeInst<UnaryInst>(vid, Type::float_type(), val, _block);
        return dyn_cast_Value(res);
    }
    //! else
    assert(ty != nullptr && "must have target type");

    switch (vid) {
        case ValueId::vSITOFP:
            assert(val->type()->is_int() && "sitofp must have int operand");
            assert(ty->is_float() && "sitofp must have float type");
            break;
        case ValueId::vFPTOSI:
            assert(val->type()->is_float() && "fptosi must have float operand");
            assert(ty->is_int() && "fptosi must have int type");
            break;
        case ValueId::vTRUNC:
        case ValueId::vZEXT:
        case ValueId::vSEXT:
            assert(val->type()->is_int() && ty->is_int());
            break;
        case ValueId::vFPTRUNC:
            assert(val->type()->is_float() && ty->is_float());
            break;
    }

    res = makeInst<UnaryInst>(vid, ty, val, _block);
    return dyn_cast_Value(res);
}

Value* IRBuilder::type_promote(Value* val, Type* target_tpye, Type* base_type) {
    Value* res = val;
    if (val->type()->btype() < target_tpye->btype()) {
        // need to promote
        if (val->type()->is_i1()) {
            if (target_tpye->is_i32()) {
                res = create_unary_beta(ValueId::vZEXT, res, Type::i32_type());

            } else if (target_tpye->is_float()) {
                res = create_unary_beta(ValueId::vZEXT, res, Type::i32_type());
                res = create_unary_beta(ValueId::vSITOFP, res,
                                        Type::float_type());
            }
        } else if (val->type()->is_i32()) {
            if (target_tpye->is_float()) {
                res = create_unary_beta(ValueId::vSITOFP, res,
                                        Type::float_type());
            }
        }
    }
    return res;
}

Value* IRBuilder::create_cmp(CmpOp op, Value* lhs, Value* rhs) {
    if (lhs->type() != rhs->type()) {
        assert(false && "create_eq_beta: type mismatch!");
    };

    auto vid = [type = lhs->type()->btype(), op] {
        switch (type) {
            case INT32:
                switch (op) {
                    case CmpOp::EQ:
                        return ValueId::vIEQ;
                    case CmpOp::NE:
                        return ValueId::vINE;
                    case CmpOp::GT:
                        return ValueId::vISGT;
                    case CmpOp::GE:
                        return ValueId::vISGE;
                    case CmpOp::LT:
                        return ValueId::vISLT;
                    case CmpOp::LE:
                        return ValueId::vISLE;
                    default:
                        assert(false && "create_cmp: invalid op!");
                }
            case FLOAT:
            case DOUBLE:
                switch (op) {
                    case CmpOp::EQ:
                        return ValueId::vFOEQ;
                    case CmpOp::NE:
                        return ValueId::vFONE;
                    case CmpOp::GT:
                        return ValueId::vFOGT;
                    case CmpOp::GE:
                        return ValueId::vFOGE;
                    case CmpOp::LT:
                        return ValueId::vFOLT;
                    case CmpOp::LE:
                        return ValueId::vFOLE;
                    default:
                        assert(false && "create_cmp: invalid op!");
                }
        }
        assert(false && "create_cmp: invalid type!");
        return ValueId::vInvalid;
    }();

    switch (lhs->type()->btype()) {
        case INT32:
            return makeInst<ICmpInst>(vid, lhs, rhs, _block);
            break;
        case FLOAT:
        case DOUBLE:
            return makeInst<FCmpInst>(vid, lhs, rhs, _block);
            break;
        default:
            assert(false && "create_eq_beta: type mismatch!");
    }
}

Value* IRBuilder::cast_to_i1(Value* val) {
    Value* res = nullptr;
    if (not val->is_i1()) {
        if (val->is_i32()) {
            res = makeInst<ICmpInst>(ValueId::vINE, val,
                                     ir::Constant::gen_i32(0), _block);
        } else if (val->is_float()) {
            res = makeInst<FCmpInst>(ValueId::vFONE, val,
                                     ir::Constant::gen_f32(0.0), _block);
        }
    } else {
        res = val;
    }
    return res;
}
Value* IRBuilder::create_load(Value* ptr) {
    auto inst = LoadInst::gen(ptr, _block);
    block()->emplace_back_inst(inst);
    if (not ptr->comment().empty()) {
        inst->set_comment(ptr->comment());
    }
    return inst;
}

Value* IRBuilder::create_alloca(Type* base_type,
                                     bool is_const,
                                     std::vector<int> dims,
                                     const_str_ref comment,
                                     int capacity) {
    AllocaInst* inst = nullptr;
    auto entryBlock = block()->parent()->entry();
    if (dims.size() == 0) {
        inst = new AllocaInst(base_type, entryBlock, "", is_const);
    } else
        inst =
            new AllocaInst(base_type, dims, entryBlock, "", is_const, capacity);
    /* hhw, add alloca to function entry block*/
    block()->parent()->entry()->emplace_back_inst(inst);
    inst->set_comment(comment);
    return inst;
}

Value* IRBuilder::create_getelementptr(Type* base_type,
                                                   Value* value,
                                                   Value* idx,
                                                   std::vector<int> dims,
                                                   std::vector<int> cur_dims) {
    GetElementPtrInst* inst = nullptr;
    if (dims.size() == 0 && cur_dims.size() == 0)
        inst = new GetElementPtrInst(base_type, value, _block, idx);
    else if (dims.size() == 0 && cur_dims.size() != 0)
        inst = new GetElementPtrInst(base_type, value, _block, idx, cur_dims);
    else
        inst = new GetElementPtrInst(base_type, value, _block, idx, dims,
                                     cur_dims);
    block()->emplace_back_inst(inst);
    return inst;
}
};  // namespace ir