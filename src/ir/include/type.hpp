#pragma once

#include <stdlib.h>
#include <string.h>
#include <vector>
// #include <memory>

namespace ir {
class Type;
class PointerType;
class FunctionType;

typedef enum : size_t {
    VOID,
    // INT1,
    INT,
    FLOAT,
    LABEL,
    POINTER,
    // ARRAY,
    FUNCTION,
    UNDEFINE
} BType;
typedef enum : size_t {
    RET,
    JMP,
    BR,
    FNEG,
    ADD,
    FADD,
    SUB,
    FSUB,
    MUL,
    FMUL,
    SDIV,
    FDIV,
    SREM,
    SHL,
    LSHR,
    ASHR,
    AND,
    OR,
    XOR,
    ALLOCA,
    LOAD,
    STORE,
    ADDRADD,
    ADDRDEREFADD,
    TRUNC,
    ZEXT,
    SEXT,
    FPTRUNC,
    FPEXT,
    FPTOSI,
    SITOFP,
    PTRTOINT,
    INTTOPTR,
    IEQ,
    INE,
    ISGT,
    ISGE,
    ISLT,
    ISLE,
    FOEQ,
    FONE,
    FOGT,
    FOGE,
    FOLT,
    FOLE,
    PHI,
    CALL,
} IType;

class Type {
  protected:
    BType _btype;

  public:
    Type(BType btype) : _btype(btype) {}
    virtual ~Type();
    // static method for construct Type instance
    static Type *void_type();
    static Type *int_type();
    static Type *float_type();
    static Type *label_type();
    static Type *pointer_type(Type *baseType);
    // static Type *array
    static Type *function_type(Type *returnType,
                               const std::vector<Type *> &paramTypes);

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
    BType get_btype();
    size_t get_size();
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
    Type *get_base_type() const { return _base_type; }
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
    Type *get_ret_type() const { return _ret_type; }
};
} // namespace ir