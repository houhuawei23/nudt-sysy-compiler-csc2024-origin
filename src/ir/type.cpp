#include "include/type.hpp"

namespace ir {
/// Type instance construct functions BEGIN
Type *Type::void_type() {
    static Type voidType(VOID);
    return &voidType;
}
// return static Type instance of int
Type *Type::int_type() {
    static Type intType(INT);
    return &intType;
}
Type *Type::float_type() {
    static Type floatType(FLOAT);
    return &floatType;
}

Type *Type::label_type() {
    static Type labelType(LABEL);
    return &labelType;
}
Type *Type::pointer_type(Type *baseType) { return PointerType::gen(baseType); }
Type *Type::function_type(Type *ret_type,
                          const std::vector<Type *> &arg_types) {
    return FunctionType::gen(ret_type, arg_types);
}
/// Type instance construct functions END
/// Type check functions BEGIN
bool Type::is(Type *type) {
    // directly compare pointer?
    // static Type, only create once
    return this == type;
}
bool Type::isnot(Type *type) { return this != type; }
bool Type::is_void() { return _btype == VOID; }
bool Type::is_int() { return _btype == INT; }
bool Type::is_float() { return _btype == FLOAT; }
bool Type::is_label() { return _btype == LABEL; }
bool Type::is_pointer() { return _btype == POINTER; }
bool Type::is_function() { return _btype == FUNCTION; }

/// Type check functions END
/// PointerType
PointerType *PointerType::gen(Type *base_type) {     // to be complete
    PointerType *ptype = new PointerType(base_type); // PointerType*
    return ptype;
}

/// FunctionType
FunctionType *FunctionType::gen(Type *ret_type,
                                const std::vector<Type *> &arg_types) {
    // to be complete
    FunctionType *ftype =
        new FunctionType(ret_type, arg_types); // FunctionType*
    return ftype;
}

} // namespace ir