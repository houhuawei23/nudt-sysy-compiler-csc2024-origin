#include "ir/global.hpp"
#include "ir/constant.hpp"

namespace ir {
/*
 * @brief: GlobalVariable::print
 * @example: 
 *      scalar: @a = global i32 0
 *      array:  @a = global [4 x [4 x i32]] zeroinitializer 
 */
void GlobalVariable::print(std::ostream& os) {
    os << name();
    if (is_const()) os << " = constant ";
    else os << " = global ";
    os << *(base_type()) << " ";
    
    if (is_array()) {
        int idx = 0, dimensions = dims_cnt();
        print_ArrayInit(os, dimensions, 0, &idx);
    } else {
        os << *scalar_value();
    }

    os << "\n";
}

void GlobalVariable::print_ArrayInit(std::ostream& os, const int dimension, const int begin, int* idx) const {
    auto atype = dyn_cast<ArrayType>(base_type());
    auto baseType = atype->base_type();
    bool is_zero = true;
    if (begin + 1 == dimension) {
        int num = atype->dim(begin);

        is_zero = true;
        for (int i = 0; i < num; i++) {
            auto cinit = dyn_cast<ir::Constant>(init(*idx + i));
            if (!cinit->is_zero()) {
                is_zero = false;
                break;
            }
        }
        
        if (is_zero) os << "zeroinitializer";
        else {
            os << "[";
            for (int i = 0; i < num - 1; i++) {
                os << *(baseType) << " " << *init(*idx + i) << ", ";
            }
            os << *(baseType) << " " << *init(*idx + num - 1);
            os << "]";
        }
        *idx = *idx + num;
    } else if (dimension == 2 + begin) {
        os << "[";

        int num1 = atype->dim(begin), num2 = atype->dim(begin + 1);
        for (int i = 0; i < num1 - 1; i++) {
            os << "[" << num2 << " x " << *(baseType) << "] ";

            is_zero = true;
            for (int j = 0; j < num2; j++) {
                auto cinit = dyn_cast<ir::Constant>(init(*idx + i * num2 + j));
                if (!cinit->is_zero()) {
                    is_zero = false;
                    break;
                }
            }
            if (is_zero) os << "zeroinitializer, ";
            else {
                os << "[";
                for (int j = 0; j < num2 - 1; j++) {
                    os << *(baseType) << " " << *init(*idx + i * num2 + j) << ", ";
                }
                os << *(baseType) << " " << *init(*idx + i * num2 + num2 - 1);
                os << "], ";
            }
        }
        os << "[" << num2 << " x " << *(baseType) << "] ";

        is_zero = true;
        for (int j = 0; j < num2; j++) {
            auto cinit = dyn_cast<ir::Constant>(init(*idx + (num1 - 1) * num2 + j));
            if (!cinit->is_zero()) {
                is_zero = false;
                break;
            }
        }
        if (is_zero) os << "zeroinitializer";
        else {
            os << "[";
            for (int j = 0; j < num2 - 1; j++) {
                os << *(baseType) << " " << *init(*idx + (num1 - 1) * num2 + j) << ", ";
            }
            os << *(baseType) << " " << *init(*idx + (num1 - 1) * num2 + num2 - 1);
            os << "]";
        }
        
        *idx = *idx + (num1 - 1) * num2 + num2;
        os << "]";
    } else {
        os << "[";

        int num = atype->dim(begin), num1 = atype->dim(begin + 1), num2 = atype->dim(begin + 2);
        for  (int i = 1; i < num; i++) {
            for (int cur = begin + 1; cur < dimension; cur++) {
                auto value = atype->dim(cur);
                os << "[" << value << " x ";
            }
            os << *(baseType);
            for (int cur = begin + 1; cur < dimension; cur++) os << "]";
            os << " ";
            print_ArrayInit(os, dimension, begin + 1, idx);
            os << ", ";
        }
        for (int cur = begin + 1; cur < dimension; cur++) {
            auto value = atype->dim(cur);
            os << "[" << value << " x ";
        }
        os << *(baseType);
        for (int cur = begin + 1; cur < dimension; cur++) os << "]";
        os << " ";
        print_ArrayInit(os, dimension, begin + 1, idx);

        os << "]";
    }
}
}  // namespace ir