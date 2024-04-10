#include "ir/global.hpp"
#include "ir/constant.hpp"

namespace ir {
/*
 * @brief: GlobalVariable::print
 * @example: 
 *      @a = global [4 x [4 x i32]] zeroinitializer 
 */
void GlobalVariable::print(std::ostream& os) {
    os << name();
    if (is_const()) os << " = constant ";
    else os << " = global ";
    
    if (is_array()) {
        os << *type() << " ";
        if (init_cnt()) {
            int idx = 0, dimensions = dims_cnt();
            print_ArrayInit(os, dimensions, 0, &idx);
        } else {  // default initialization
            os << "zeroinitializer";
        }
    } else {
        os << *(base_type()) << " " << *scalar_value();
    }

    os << "\n";
}

void GlobalVariable::print_ArrayInit(std::ostream& os, const int dimension, const int begin, int* idx) const {
    auto atype = dyn_cast<ArrayType>(type());
    if (begin + 1 == dimension) {
        os << "[";

        int num = atype->dim(begin);
        for (int i = 0; i < num - 1; i++) {
            os << *(base_type()) << " " << *init(*idx + i) << ", ";
        }
        os << *(base_type()) << " " << *init(*idx + num - 1);
        
        os << "]";
        
        *idx = *idx + num;
    } else if (dimension == 2 + begin) {
        os << "[";

        int num1 = atype->dim(begin), num2 = atype->dim(begin + 1);
        for (int i = 0; i < num1 - 1; i++) {
            os << "[" << num2 << " x " << *(base_type()) << "] ";

            os << "[";
            for (int j = 0; j < num2 - 1; j++) {
                os << *(base_type()) << " " << *init(*idx + i * num2 + j) << ", ";
            }
            os << *(base_type()) << " " << *init(*idx + i * num2 + num2 - 1);
            os << "], ";
        }
        os << "[" << num2 << " x " << *(base_type()) << "] ";

        os << "[";
        for (int j = 0; j < num2 - 1; j++) {
            os << *(base_type()) << " " << *init(*idx + (num1 - 1) * num2 + j) << ", ";
        }
        os << *(base_type()) << " " << *init(*idx + (num1 - 1) * num2 + num2 - 1);
        os << "]";
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
            os << *(base_type());
            for (int cur = begin + 1; cur < dimension; cur++) os << "]";
            os << " ";
            print_ArrayInit(os, dimension, begin + 1, idx);
            os << ", ";
        }
        for (int cur = begin + 1; cur < dimension; cur++) {
            auto value = atype->dim(cur);
            os << "[" << value << " x ";
        }
        os << *(base_type());
        for (int cur = begin + 1; cur < dimension; cur++) os << "]";
        os << " ";
        print_ArrayInit(os, dimension, begin + 1, idx);

        os << "]";
    }
}
}  // namespace ir