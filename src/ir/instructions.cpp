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
void AllocaInst::print(std::ostream &os) const {
    // print var name
    os << "%" << name() << " = ";
    os << "alloca ";
    // just for int scalar
    Type *base_type = dynamic_cast<PointerType *>(type())->base_type();
    os << *base_type;
}

void StoreInst::print(std::ostream &os) const {
    // store i32 5, i32* %1
    os << "store ";
    os << *(value()->type()) << " ";
    os << *value() << ", "; // constant worked
    os << *ptr()->type() << " ";
    os << "%" << ptr()->name();
}

void LoadInst::print(std::ostream &os) const {
    // %2 = load i32, i32* %1
    //! to do
}

void ReturnInst::print(std::ostream &os) const {
    // ret i32 %2
    //! to do
    os << "ret ";
    // if (auto value = return_value()) {
    //     os <<
    // }

    // Type* ty
    os << *return_value()->type() << " ";
    os << *return_value();
}
} // namespace ir
