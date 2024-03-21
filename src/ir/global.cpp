#pragma onece
#include "include/global.hpp"

namespace ir {
/*
 * @brief GlobalVariable::print
 * @example: 
 *      @a = global [4 x [4 x i32]] zeroinitializer 
 */
void GlobalVariable::print(std::ostream& os) {
    os << name() << " = global ";
    if (is_array()) {
        int dimensions = dims_cnt();
        for (int cur = 0; cur < dimensions; cur++) {
            auto value = operand(cur);
            if (auto cvalue = ir::dyn_cast<ir::Constant>(value)) {
                os << "[" << *value << " x ";
            } else {
                assert(false);
            }
        }
        os << *(base_type());
        for (int cur = 0; cur < dimensions; cur++) os << "]";
        os << " ";

        if (init_cnt()) {
            os << "[";
            print_ArrayInit(os, dimensions);
            os << "]";
        } else {  // default initialization
            os << "zeroinitializer";
        }
    } else {
        os << *(base_type()) << " " << *scalar_value();
    }
    os << "\n";
}

void GlobalVariable::print_ArrayInit(std::ostream& os, const int dimension) const {
    if (dimension == 1) {
        int num = ir::dyn_cast<ir::Constant>(operand(0))->i32();
        for (int i = 0; i < num - 1; i++) {
            os << *(base_type()) << " " << *init(i) << ", ";
        }
        os << *(base_type()) << " " << *init(num - 1);
    } else if (dimension == 2) {
        int num1 = ir::dyn_cast<ir::Constant>(operand(0))->i32();
        int num2 = ir::dyn_cast<ir::Constant>(operand(1))->i32();
        for (int i = 0; i < num1 - 1; i++) {
            os << "[" << num1 << " x " << *(base_type()) << "] ";

            os << "[";
            for (int j = 0; j < num2 - 1; j++) {
                os << *(base_type()) << " " << *init( i * num2 + j) << ", ";
            }
            os << *(base_type()) << " " << *init(i * num2 + num2 - 1);
            os << "], ";
        }
        os << "[" << num1 << " x " << *(base_type()) << "] ";

        os << "[";
        for (int j = 0; j < num2 - 1; j++) {
            os << *(base_type()) << " " << *init((num1 - 1) * num2 + j) << ", ";
        }
        os << *(base_type()) << " " << *init((num1 - 1) * num2 + num2 - 1);
        os << "]";
    } else {
        // TODO: 多维数组的IR输出
    }
}
}  // namespace ir