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
    os << name() << " = ";
    os << "alloca ";
    // just for int scalar

    os << *(base_type());
}

void StoreInst::print(std::ostream &os) const {
    // store i32 5, i32* %1
    os << "store ";
    os << *(value()->type()) << " ";
    os << value()->name() << ", "; // constant worked
    os << *ptr()->type() << " ";
    os << ptr()->name();
}

void LoadInst::print(std::ostream &os) const {
    // %2 = load i32, i32* %1
    //! to do
    os<<name()<<" = ";
    os<<"load ";
    os<<* dyn_cast<PointerType>(ptr()->type())->base_type()<<", "
    <<*ptr()->type()<<" "<<ptr()->name();
}

void ReturnInst::print(std::ostream &os) const {
    // ret i32 %2
    //! to do
    os << "ret ";
    // if (auto value = return_value()) {
    //     os <<
    // }

    // Type* ty
    auto ret=return_value();
    if(ret){
        os << *ret->type() << " ";
        os << ret->name();
    }
    else{
        os<<"void";
    }
}
} // namespace ir
