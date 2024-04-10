#include "ir/type.hpp"
#include <assert.h>

#include <variant>  // dynamic_cast

namespace ir {
/// Type instance construct functions BEGIN
Type* Type::void_type() {
    static Type voidType(VOID);
    return &voidType;
}

Type* Type::i1_type() {
    static Type i1Type(INT1);
    return &i1Type;
}

// return static Type instance of int
Type* Type::i32_type() {
    static Type intType(INT32);
    return &intType;
}
Type* Type::float_type() {
    static Type floatType(FLOAT);
    return &floatType;
}
Type* Type::double_type() {
    static Type doubleType(DOUBLE);
    return &doubleType;
}
Type* Type::label_type() {
    static Type labelType(LABEL);
    return &labelType;
}

Type* Type::pointer_type(Type* baseType) {
    return PointerType::gen(baseType);
}
Type* Type::array_type(Type* baseType, std::vector<int> dims) {
    return ArrayType::gen(baseType, dims);
}
Type* Type::func_type(Type* ret_type, const type_ptr_vector& arg_types) {
    return FunctionType::gen(ret_type, arg_types);
}

//! type check
bool Type::is(Type* type) {
    return this == type;
}
bool Type::isnot(Type* type) {
    return this != type;
}

bool Type::is_void() {
    return _btype == VOID;
}

bool Type::is_i1() {
    return _btype == INT1;
}
bool Type::is_i32() {
    return _btype == INT32;
}

bool Type::is_float32() {
    return _btype == FLOAT;
}
bool Type::is_double() {
    return _btype == DOUBLE;
}

bool Type::is_label() {
    return _btype == LABEL;
}
bool Type::is_pointer() {
    return _btype == POINTER;
}
bool Type::is_function() {
    return _btype == FUNCTION;
}
bool Type::is_array() {
    return _btype == ARRAY;
}

//! print
void Type::print(std::ostream& os){
    auto basetype = btype();
    switch (basetype) {
        case INT1:
            os << "i1";
            break;
        case INT32:
            os << "i32";
            break;
        case FLOAT:
            os << "float";
            break;
        case DOUBLE:
            os << "float";
            break;
        case VOID:
            os << "void";
            break;
        case FUNCTION:
            os << "function";
            break;
        case POINTER:
            static_cast<const PointerType*>(this)->base_type()->print(os);
            os << "*";
            break;
        case LABEL:
            break;
        case ARRAY:
            if (auto atype = static_cast<ArrayType*>(this)) {
                int dims = atype->dims_cnt();
                for (int i = 0; i < dims; i++) {
                    int value = atype->dim(i);
                    os << "[" << value << " x ";
                }
                atype->base_type()->print(os);
                for (int i = 0; i < dims; i++) os << "]";
            } else {
                assert(false);
            }
            break;
        default:
            break;
    }
}


PointerType* PointerType::gen(Type* base_type) {
    PointerType* ptype = new PointerType(base_type);
    return ptype;
}
ArrayType* ArrayType::gen(Type* baseType, std::vector<int> dims) {
    ArrayType* ptype = new ArrayType(baseType, dims);
    return ptype;
}
FunctionType* FunctionType::gen(Type* ret_type, const type_ptr_vector& arg_types) {
    FunctionType* ftype = new FunctionType(ret_type, arg_types);
    return ftype;
}

}  // namespace ir