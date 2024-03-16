#include "include/instructions.hpp"
#include "include/utils.hpp"

namespace ir {
// Value <- User <- Instruction <- XxxInst
/**
 * @brief AllocaInst::print
 *
 * @param os
 *
 * %1 = alloca i32
 */
void AllocaInst::print(std::ostream& os) const {
    os << name() << " = ";
    os << "alloca ";
    // just for int scalar

    os << *(base_type());
}

void StoreInst::print(std::ostream& os) const {
    // store i32 5, i32* %1
    os << "store ";
    os << *(value()->type()) << " ";
    os << value()->name() << ", ";  // constant worked
    os << *ptr()->type() << " ";
    os << ptr()->name();
}

void LoadInst::print(std::ostream& os) const {
    // %2 = load i32, i32* %1
    //! to do
    os << name() << " = ";
    os << "load ";
    os << *dyn_cast<PointerType>(ptr()->type())->base_type() << ", "
       << *ptr()->type() << " " << ptr()->name();
}

void ReturnInst::print(std::ostream& os) const {
    // ret i32 %2
    //! to do
    os << "ret ";
    // if (auto value = return_value()) {
    //     os <<
    // }

    // Type* ty
    auto ret = return_value();
    if (ret) {
        os << *ret->type() << " ";
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
    switch (scid())
    {
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
    os << get_lvalue()->name() << ", ";
    // <op2>
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
    switch (scid())
    {
    case vFTOI:
        os << "fptosi ";
        
        if (is_int()) os << "float ";
        else os << "i32 ";
        os << get_value()->name() << " to " << *type();
        break;
    case vITOF:
        os << "sitofp ";

        if (is_int()) os << "float ";
        else os << "i32 ";
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
        default:
            assert(false && "unimplemented");
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
        os << "label " << iftrue()->name() << ", ";
        os << "label " << iffalse()->name();
    } else {
        os << "label " << dest()->name();
    }
}
}  // namespace ir
