#pragma once

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>
// #include <memory>
#include <any>

namespace ir {
class Type;
class PointerType;
class FunctionType;

typedef enum : size_t {
    VOID,
    INT,
    FLOAT,
    LABEL,
    POINTER,
    FUNCTION,
    UNDEFINE
} BType;
class Type {
  protected:
    BType _btype;

  public:
    Type(BType btype) : _btype(btype) {}
    virtual ~Type() = default; // default deconstructor
    // static method for construct Type instance
    static Type *void_type();
    static Type *int_type();
    static Type *float_type();
    static Type *label_type();
    static Type *pointer_type(Type *baseType);
    // static Type *array
    static Type *function_type(Type *ret_type,
                               const std::vector<Type *> &param_types);

    // type check
    bool is(Type *type);
    bool isnot(Type *type);
    bool is_void();
    bool is_int();
    bool is_float();
    bool is_label();
    bool is_pointer();
    bool is_function();

    // get attribute
    // Type* btype();
    BType btype() const { return _btype; };
    size_t size() const {
        switch (_btype) {
        case INT:
            return 4;
            break;
        case FLOAT:
            return 4;
        case LABEL:
        case POINTER:
        case FUNCTION:
            return 8;
        case VOID:
            return 0;
        default:
            break;
        }
    };
    void print(std::ostream &os) const;
};
/**
 * @brief
 *
 */
class PointerType : public Type {
    //! inherit from Type
    // BType _btype;
  protected:
    Type *_base_type;
    PointerType(Type *baseType) : Type(POINTER), _base_type(baseType) {}

  public:
    //! Generate a pointer type from a given base type
    static PointerType *gen(Type *baseType);

    //! Get the base type of this pointer
    Type *base_type() const { return _base_type; }

    // void print(std::ostream &os) const
};

class FunctionType : public Type {
    //! inherit from Type
    // BType _btype;
  protected:
    //! the return type of the function
    Type *_ret_type;
    //! the argument types of the function
    std::vector<Type *> _arg_types;
    //! the constructor for FunctionType
    FunctionType(Type *ret_type, const std::vector<Type *> &arg_types = {})
        : Type(FUNCTION), _ret_type(ret_type), _arg_types(arg_types) {}

  public:
    //! Gen
    static FunctionType *gen(Type *ret_type,
                             const std::vector<Type *> &arg_types);
    //! get the return type of the function
    Type *ret_type() const { return _ret_type; }
    std::vector<Type *> param_type() const { // ?? return what type?
        // make_range(paramTypes)
        return _arg_types;
    }
};
} // namespace ir