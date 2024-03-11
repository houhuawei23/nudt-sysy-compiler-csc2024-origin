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
    os << "%" << get_name() << " = ";
    os << "alloca ";
    // just for int scalar
    Type *base_type = dynamic_cast<PointerType *>(get_type())->get_base_type();
    os << *base_type;
}

void StoreInst::print(std::ostream &os) const {
    // store i32 5, i32* %1
    os << "store ";
    os << *(get_value()->get_type()) << " ";
    os << *get_value() << ", "; // constant worked
    os << *get_ptr()->get_type() << " ";
    os << "%" << get_ptr()->get_name();
}

void LoadInst::print(std::ostream &os) const {
    // %2 = load i32, i32* %1
    //! to do
}

void ReturnInst::print(std::ostream &os) const {
    // ret i32 %2
    //! to do
    os << "ret ";
    // if (auto value = get_return_value()) {
    //     os << 
    // }

    // just for simplest, return 0
    os << "i32 0";
}
} // namespace ir
