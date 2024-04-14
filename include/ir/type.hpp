#pragma once

#include <stdlib.h>
#include <iostream>

#include <vector>

namespace ir {
// mir
class DataLayout;

class Type;
class PointerType;
class FunctionType;

using type_ptr_vector = std::vector<Type*>;

// using const_vector_type = const std::vector<Type*>;
// ir base type
typedef enum : size_t {
    VOID,
    INT1,
    INT32,
    FLOAT,   // represent f32 in C
    DOUBLE,  // represent f64

    LABEL,  // BasicBlock
    POINTER,
    FUNCTION,

    ARRAY, 

    UNDEFINE
} BType;

class Type {
    protected:
        BType _btype;

    public:
        Type(BType btype) : _btype(btype) {}
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
        static Type* array_type(Type* baseType, std::vector<int> dims);
        static Type* func_type(Type* ret_type, const type_ptr_vector& param_types);

    public:  // check
        bool is(Type* type);
        bool isnot(Type* type);
        bool is_void();

        bool is_i1();
        bool is_i32();
        bool is_int() { return is_i1() || is_i32(); }

        bool is_float32();
        bool is_double();
        bool is_float() { return is_float32() || is_double(); }
        
        bool is_label();
        bool is_pointer();
        bool is_array();
        bool is_function();

    public:  // get
        BType btype() const { return _btype; };
        size_t size() const { 
            switch (_btype) {
                case INT32:
                    return 4;
                    break;
                case FLOAT:
                    return 4;
                    break;
                case DOUBLE:
                case LABEL:
                case POINTER:
                case FUNCTION:
                    return 8;
                    break;
                case VOID:
                    return 0;
                default:
                    break;
            }
            return -1;
        };

    public:
        void print(std::ostream& os);
};

class PointerType : public Type {
    protected:
        Type* _base_type;
        PointerType(Type* baseType) : Type(POINTER), _base_type(baseType) {}

    public:
        static PointerType* gen(Type* baseType);

        Type* base_type() const { return _base_type; }
};

class ArrayType : public Type {
    protected:
        std::vector<int> _dims;  // dimensions
        Type* _base_type;        // int or float
        ArrayType(Type* baseType, std::vector<int> dims) 
            : Type(ARRAY), _base_type(baseType), _dims(dims) {}

    public:
        static ArrayType* gen(Type* baseType, std::vector<int> dims);

        int dims_cnt() const { return _dims.size(); }
        int dim(int index) const { return _dims[index]; }
        std::vector<int> dims() const { return _dims; }
        Type* base_type() const { return _base_type; }
};

class FunctionType : public Type {
    protected:
        Type* _ret_type;
        std::vector<Type*> _arg_types;

    FunctionType(Type* ret_type, const type_ptr_vector& arg_types={})
        : Type(FUNCTION), _ret_type(ret_type), _arg_types(arg_types) {}

    public:  // Gen
        static FunctionType* gen(Type* ret_type, const type_ptr_vector& arg_types);

    //! get the return type of the function
    Type* ret_type() const { return _ret_type; }

    type_ptr_vector& arg_types() { return _arg_types; }
};
}  // namespace ir