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
    // print var name
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
