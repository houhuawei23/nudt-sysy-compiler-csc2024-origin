#include "include/instructions.hpp"
#include "include/utils.hpp"

namespace ir {
//! Value <- User <- Instruction <- XxxInst

/*
 * @brief AllocaInst::print
 */
void AllocaInst::print(std::ostream& os) const {
    os << name() << " = ";
    os << "alloca ";

    if (is_scalar() > 0) {  //! 1. scalar
        os << *(base_type());
    } else {  //! 2. array (未考虑变量定义数组维度)
        int dims = dims_cnt();
        for (int i = 0; i < dims; i++) {
            auto value = operand(i);
            if (auto cvalue = ir::dyn_cast<ir::Constant>(value)) {
                os << "[" << *value << " x ";
            } else {
                assert(false);
            }
        }
        os << *(base_type());
        for (int i = 0; i < dims; i++)
            os << "]";
    }
}

/*
 * @brief StoreInst::print
 * @details:
 *      store <ty> <value>, ptr <pointer>
 */
void StoreInst::print(std::ostream& os) const {
    os << "store ";
    os << *(value()->type()) << " ";
    os << value()->name() << ", ";
    os << *ptr()->type() << " ";
    os << ptr()->name();
}

void LoadInst::print(std::ostream& os) const {
    os << name() << " = ";
    os << "load ";
    os << *dyn_cast<PointerType>(ptr()->type())->base_type() << ", "
       << *ptr()->type() << " ";
    os << ptr()->name();
}

void ReturnInst::print(std::ostream& os) const {
    os << "ret ";
    auto ret = return_value();
    if (ret) {
        os << *ret->type() << " ";
        if (ir::isa<ir::Constant>(ret))
            os << *(ret);
        else
            os << ret->name();
    } else {
        os << "void";
    }
}

/*
 * @brief Binary Instruction Output
 *      <result> = add <ty> <op1>, <op2>
 */
void BinaryInst::print(std::ostream& os) const {
    os << name() << " = ";
    switch (scid()) {
        case vADD:
            os << "add ";
            break;
        case vFADD:
            os << "fadd ";
            break;

        case vSUB:
            os << "sub ";
            break;
        case vFSUB:
            os << "fsub ";
            break;

        case vMUL:
            os << "mul ";
            break;
        case vFMUL:
            os << "fmul ";
            break;

        case vSDIV:
            os << "sdiv ";
            break;
        case vFDIV:
            os << "fdiv ";
            break;

        case vSREM:
            os << "srem ";
            break;
        case vFREM:
            os << "frem ";
            break;

        default:
            break;
    }
    // <type>
    os << *type() << " ";
    // <op1>
    if (ir::isa<ir::Constant>(get_lvalue()))
        os << *(get_lvalue()) << ", ";
    else
        os << get_lvalue()->name() << ", ";
    // <op2>
    if (ir::isa<ir::Constant>(get_rvalue()))
        os << *(get_rvalue());
    else
        os << get_rvalue()->name();
}

/*
 * @brief Unary Instruction Output
 *      <result> = sitofp <ty> <value> to <ty2>
 *      <result> = fptosi <ty> <value> to <ty2>
 *
 *      <result> = fneg [fast-math flags]* <ty> <op1>
 */
void UnaryInst::print(std::ostream& os) const {
    os << name() << " = ";
    switch (scid()) {
        case vFTOI:
            os << "fptosi ";

            if (is_i32())
                os << "float ";
            else
                os << "i32 ";
            os << get_value()->name() << " to " << *type();
            break;
        case vITOF:
            os << "sitofp ";

            if (is_i32())
                os << "float ";
            else
                os << "i32 ";
            os << get_value()->name() << " to " << *type();
            break;
        case vFNEG:
            os << "fneg " << *type() << " " << get_value()->name();
            break;
        default:
            break;
    }
}

void ICmpInst::print(std::ostream& os) const {
    // <result> = icmp <cond> <ty> <op1>, <op2>   ; yields i1 or <N x i1>:result
    // %res = icmp eq i32, 1, 2
    os << name() << " = ";

    os << "icmp ";
    // cond code
    switch (scid()) {
        case vIEQ:
            os << "eq ";
            break;
        case vINE:
            os << "ne ";
            break;
        case vISGT:
            os << "sgt ";
            break;
        case vISGE:
            os << "sge ";
            break;
        case vISLT:
            os << "slt ";
            break;
        case vISLE:
            os << "sle ";
            break;
        default:
            // assert(false && "unimplemented");
            std::cerr << "Error from ICmpInst::print(), wrong Inst Type!"
                      << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
    // type
    os << *lhs()->type() << " ";
    // op1
    os << lhs()->name() << ", ";
    // op2
    os << rhs()->name();
}

void FCmpInst::print(std::ostream& os) const {
    // <result> = icmp <cond> <ty> <op1>, <op2>   ; yields i1 or <N x i1>:result
    // %res = icmp eq i32, 1, 2
    os << name() << " = ";

    os << "fcmp ";
    // cond code
    switch (scid()) {
        case vFOEQ:
            os << "oeq ";
            break;
        case vFONE:
            os << "one ";
            break;
        case vFOGT:
            os << "ogt ";
            break;
        case vFOGE:
            os << "oge ";
            break;
        case vFOLT:
            os << "olt ";
            break;
        case vFOLE:
            os << "ole ";
            break;
        default:
            // assert(false && "unimplemented");
            std::cerr << "Error from FCmpInst::print(), wrong Inst Type!"
                      << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
    // type
    os << *lhs()->type() << " ";
    // op1
    os << lhs()->name() << ", ";
    // op2
    os << rhs()->name();
}

void BranchInst::print(std::ostream& os) const {
    // br i1 <cond>, label <iftrue>, label <iffalse>
    // br label <dest>
    os << "br ";
    //
    if (is_cond()) {
        os << "i1 ";
        os << cond()->name() << ", ";
        os << "label "
           << "%" << iftrue()->name() << ", ";
        os << "label "
           << "%" << iffalse()->name();
    } else {
        os << "label "
           << "%" << dest()->name();
    }
}

/*
 * @brief GetElementPtrInst::print
 *      数组: <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32
 * <idx> 指针: <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
 */
void GetElementPtrInst::print(std::ostream& os) const {
    if (is_arrayInst()) {
        // 确定数组指针地址
        // <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32 idx
        int dimensions = dims_cnt();
        os << name() << " = "
           << "getelementptr ";

        for (int cur = current_dimension(); cur < dimensions + 1; cur++) {
            auto value = operand(cur);
            if (auto cvalue = ir::dyn_cast<ir::Constant>(value)) {
                os << "[" << *value << " x ";
            } else {
                assert(false);
            }
        }
        os << *(base_type());
        for (int cur = current_dimension(); cur < dimensions + 1; cur++)
            os << "]";
        os << ", ";

        for (int cur = current_dimension(); cur < dimensions + 1; cur++) {
            auto value = operand(cur);
            if (auto cvalue = ir::dyn_cast<ir::Constant>(value)) {
                os << "[" << *value << " x ";
            } else {
                assert(false);
            }
        }
        os << *(base_type());
        for (int cur = current_dimension(); cur < dimensions + 1; cur++)
            os << "]";
        os << "* ";

        os << get_value()->name() << ", ";
        os << *(base_type()) << " 0, " << *(base_type()) << " "
           << get_index()->name();
    } else {
        // <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
    }
}

void CallInst::print(std::ostream& os) const {
    if (name().size() == 0) {
        os << "call ";
    } else {
        os << name() << " = call ";
    }
    // ret_type
    os << *type() << " ";
    // func name
    os << "@" << callee()->name() << "(";

    if (_rargs.size() > 0) {
        auto last = _rargs.end() - 1;  // Iterator pointing to the last element
        for (auto it = _rargs.begin(); it != last; ++it) {
            // it is a iterator, *it get the element in _rargs,
            // which is the Value* ptr
            os << *((*it)->type()) << " ";
            os << (*it)->name();
            os << ", ";
        }
        os << *((*last)->type()) << " ";
        os << (*last)->name();
    }

    os << ")";
}
void CallInstBeta::print(std::ostream& os) const {
    os << "call ";
    // ret_type
    os << *type() << " ";
    // func name
    // os << "@" << callee()->name() << "(";
    // for (auto rarg : _rargs) {
    //     os << rarg->type() << " ";
    //     os << rarg->name();
    //     if (rarg != _rargs.back())
    //         os << ", ";
    // }
    os << ")";
}
}  // namespace ir
