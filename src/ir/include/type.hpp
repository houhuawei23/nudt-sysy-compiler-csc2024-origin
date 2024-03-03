#pragma once

#include <stdlib.h>
#include <vector>
// #include <string>
// #include <memory>

namespace ir {
class Type;
class PointerType;
class FunctionType;

typedef enum
  : size_t
{
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
typedef enum
  : size_t
{
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

class Type
{
  protected:
    BType _btype;

  public:
    Type(BType btype)
      : _btype(btype)
    {
    }
    virtual ~Type();
    // static method for construct Type instance
    static Type* void_type();
    static Type* int_type();
    static Type* float_type();
    static Type* label_type();
    static Type* pointer_type();
    // static Type *array
    static Type* function_type();

    // type check
    bool is(Type* type);
    bool isnot(Type* type);
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

class PointerType : public Type
{
  protected:
    Type* _base;
    PointerType(Type* base);

  public:
    static PointerType* get(Type* base);
};

class FunctionType : public Type
{
  protected:
    Type* _ret_type;
    std::vector<Type*> _args_types;

  public:
    FunctionType(Type* ret, const std::vector<Type*>& agrs_types);
};
}