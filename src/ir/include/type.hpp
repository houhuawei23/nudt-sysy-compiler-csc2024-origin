#pragma once

#include <stdlib.h>
#include <iostream>

#include <vector>

namespace ir {
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
    LABEL,   // BasicBlock
    POINTER,
    FUNCTION,
    UNDEFINE
} BType;

class Type {
   protected:
    BType _btype;

   public:
    Type(BType btype) : _btype(btype) {}
    virtual ~Type() = default;  // default deconstructor

    // static method for construct Type instance
    static Type* void_type();

    static Type* i1_type();
    static Type* i32_type();

    static Type* float_type();
    static Type* double_type();

    static Type* label_type();
    static Type* pointer_type(Type* baseType);
    // static Type *array
    static Type* function_type(Type* ret_type,
                               const type_ptr_vector& param_types);

    // type check
    bool is(Type* type);
    bool isnot(Type* type);
    bool is_void();

    bool is_i1();
    bool is_i32();

    bool is_float32();  // only check f32
    bool is_double();   // only check f64
    bool is_float() { return is_float32() || is_double(); }
    bool is_label();
    bool is_pointer();
    bool is_function();

    // get attribute
    // Type* btype();
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
    void print(std::ostream& os);
};


class PointerType : public Type {
    //* inherit from Type
    // _btype = POINTER
   protected:
    Type* _base_type;
    PointerType(Type* baseType) : Type(POINTER), _base_type(baseType) {}

   public:
    // Generate a pointer type from a given base type
    static PointerType* gen(Type* baseType);

    //! Get the base type of this pointer
    Type* base_type() const { return _base_type; }

    // void print(std::ostream &os) const
};

class FunctionType : public Type {
    //*inherit from Type
    // BType _btype = FUNCTION
   protected:

    // the return type of the function
    Type* _ret_type;

    // the argument types of the function
    std::vector<Type*> _arg_types;
    
    // the constructor for FunctionType
    FunctionType(Type* ret_type, const type_ptr_vector& arg_types = {})
        : Type(FUNCTION), _ret_type(ret_type), _arg_types(arg_types) {}

   public:
    //! Gen
    static FunctionType* gen(Type* ret_type, const type_ptr_vector& arg_types);
    
    //! get the return type of the function
    Type* ret_type() const { return _ret_type; }

    type_ptr_vector& arg_types() {
        return _arg_types;
    }
};
}  // namespace ir