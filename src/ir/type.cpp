#include "ir/type.hpp"
#include <assert.h>
#include <variant>

namespace ir {
Type* Type::void_type() {
    static Type voidType(VOID);
    return &voidType;
}

Type* Type::TypeBool() {
    static Type i1Type(INT1);
    return &i1Type;
}

// return static Type instance of size_t
Type* Type::TypeInt32() {
    static Type intType(INT32);
    return &intType;
}
Type* Type::TypeFloat32() {
    static Type floatType(FLOAT);
    return &floatType;
}
Type* Type::TypeDouble() {
    static Type doubleType(DOUBLE);
    return &doubleType;
}
Type* Type::TypeLabel() {
    static Type labelType(LABEL);
    return &labelType;
}
Type* Type::TypeUndefine() {
    static Type undefineType(UNDEFINE);
    return &undefineType;
}
Type* Type::TypePointer(Type* baseType) {
    return PointerType::gen(baseType);
}
Type* Type::TypeArray(Type* baseType, std::vector<size_t> dims, size_t capacity) {
    return ArrayType::gen(baseType, dims, capacity);
}
Type* Type::TypeFunction(Type* ret_type, const type_ptr_vector& arg_types) {
    return FunctionType::gen(ret_type, arg_types);
}

//! type check
bool Type::is(Type* type) {
    return this == type;
}
bool Type::isnot(Type* type) {
    return this != type;
}

bool Type::isVoid() {
    return mBtype == VOID;
}

bool Type::isBool() {
    return mBtype == INT1;
}
bool Type::isInt32() {
    return mBtype == INT32;
}

bool Type::isFloat32() {
    return mBtype == FLOAT;
}
bool Type::isDouble() {
    return mBtype == DOUBLE;
}

bool Type::isUnder() {
    return mBtype == UNDEFINE;
}

bool Type::isLabel() {
    return mBtype == LABEL;
}
bool Type::isPointer() {
    return mBtype == POINTER;
}
bool Type::isFunction() {
    return mBtype == FUNCTION;
}
bool Type::isArray() {
    return mBtype == ARRAY;
}

//! print
void Type::print(std::ostream& os) {
    auto basetype = btype();
    switch (basetype) {
        case INT1: os << "i1";
            break;
        case INT32: os << "i32";
            break;
        case FLOAT: os << "float";
            break;
        case DOUBLE: os << "float";
            break;
        case VOID: os << "void";
            break;
        case FUNCTION: os << "function";
            break;
        case POINTER:
            static_cast<const PointerType*>(this)->baseType()->print(os);
            os << "*";
            break;
        case LABEL: break;
        case ARRAY:
            if (auto atype = static_cast<ArrayType*>(this)) {
                size_t dims = atype->dims_cnt();
                for (size_t i = 0; i < dims; i++) {
                    size_t value = atype->dim(i);
                    os << "[" << value << " x ";
                }
                atype->baseType()->print(os);
                for (size_t i = 0; i < dims; i++)
                    os << "]";
            } else {
                assert(false);
            }
            break;
        default: break;
    }
}

PointerType* PointerType::gen(Type* base_type) {
    PointerType* ptype = new PointerType(base_type);
    return ptype;
}
ArrayType* ArrayType::gen(Type* baseType, std::vector<size_t> dims, size_t capacity) {
    ArrayType* ptype = new ArrayType(baseType, dims, capacity);
    return ptype;
}
FunctionType* FunctionType::gen(Type* ret_type,
                                const type_ptr_vector& arg_types) {
    FunctionType* ftype = new FunctionType(ret_type, arg_types);
    return ftype;
}
}