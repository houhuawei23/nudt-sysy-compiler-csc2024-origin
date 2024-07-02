#pragma once
#include <stdlib.h>
#include <iostream>
#include <vector>

namespace ir {
class DataLayout;
class Type;
class PointerType;
class FunctionType;
using type_ptr_vector = std::vector<Type*>;
typedef enum : size_t {
    VOID,
    INT1,
    INT32,
    FLOAT,
    DOUBLE,
    LABEL,
    POINTER,
    FUNCTION,
    ARRAY, 
    UNDEFINE
} BType;

/* Type */
class Type {
protected:
    BType _btype;
    size_t _size;
public:
    Type(BType btype, size_t size=4) : _btype(btype), _size(size) {}
    virtual ~Type() = default;
public:  // static method for construct Type instance
    static Type* void_type();
    static Type* i1_type();
    static Type* i32_type();
    static Type* float_type();
    static Type* double_type();
    static Type* label_type();
    static Type* undefine_type();
    static Type* pointer_type(Type* baseType);
    static Type* array_type(Type* baseType, std::vector<int> dims, int capacity=1);
    static Type* func_type(Type* ret_type, const type_ptr_vector& param_types);
public:  // check function
    bool is(Type* type);
    bool isnot(Type* type);
    bool is_void();
    bool is_i1();
    bool is_i32();
    bool is_int() { return is_i1() || is_i32(); }
    bool is_float32();
    bool is_double();
    bool is_float() { return is_float32() || is_double(); }
    bool is_undef();
    bool is_label();
    bool is_pointer();
    bool is_array();
    bool is_function();
public:  // get function
    BType btype() const { return _btype; }
    size_t size() const { return _size; }
public:
    void print(std::ostream& os);
};
/* PointerType */
class PointerType : public Type {
protected:
    Type* _base_type;
    PointerType(Type* baseType) : Type(POINTER, 8), _base_type(baseType) {}
public:
    static PointerType* gen(Type* baseType);
    Type* base_type() const { return _base_type; }
};
/* ArrayType */
class ArrayType : public Type {
protected:
    std::vector<int> _dims;  // dimensions
    Type* _base_type;        // int or float
    ArrayType(Type* baseType, std::vector<int> dims, int capacity=1) 
        : Type(ARRAY, capacity * 4), _base_type(baseType), _dims(dims) {}
public:  // generate function
    static ArrayType* gen(Type* baseType, std::vector<int> dims, int capacity=1);
public:  // get function
    int dims_cnt() const { return _dims.size(); }
    int dim(int index) const { return _dims[index]; }
    std::vector<int> dims() const { return _dims; }
    Type* base_type() const { return _base_type; }
};
/* FunctionType */
class FunctionType : public Type {
protected:
    Type* _ret_type;
    std::vector<Type*> _arg_types;
    FunctionType(Type* ret_type, const type_ptr_vector& arg_types={})
        : Type(FUNCTION, 8), _ret_type(ret_type), _arg_types(arg_types) {}
public:  // generate function
    static FunctionType* gen(Type* ret_type, const type_ptr_vector& arg_types);
public:  // get function
    Type* ret_type() const { return _ret_type; }
    type_ptr_vector& arg_types() { return _arg_types; }
};
}