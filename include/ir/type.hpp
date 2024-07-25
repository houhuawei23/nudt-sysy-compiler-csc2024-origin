#pragma once
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cassert>
#include "support/arena.hpp"
namespace ir {
class DataLayout;
class Type;
class PointerType;
class FunctionType;
using type_ptr_vector = std::vector<Type*>;

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
  ARRAY,
  UNDEFINE
} BType; // FIXME: BType -> BasicTypeRank

/* Type */
class Type {
 protected:
  BType mBtype;
  size_t mSize;

 public:
  static constexpr auto arenaSource = utils::Arena::Source::IR;
  Type(BType btype, size_t size = 4) : mBtype(btype), mSize(size) {}
  virtual ~Type() = default;

 public:  // static method for construct Type instance
  static Type* void_type();

  static Type* TypeBool();
  static Type* TypeInt32();

  static Type* TypeFloat32();
  static Type* TypeDouble();

  static Type* TypeLabel();
  static Type* TypeUndefine();
  static Type* TypePointer(Type* baseType);
  static Type* TypeArray(Type* baseType,
                         std::vector<size_t> dims,
                         size_t capacity = 1);
  static Type* TypeFunction(Type* ret_type, const type_ptr_vector& param_types);

 public:  // check
  bool is(Type* type);
  bool isnot(Type* type);
  bool isVoid();

  bool isBool();
  bool isInt32();
  bool isInt() { return isBool() || isInt32(); }

  bool isFloat32();
  bool isDouble();
  bool isFloatPoint() { return isFloat32() || isDouble(); }
  bool isUndef();

  bool isLabel();
  bool isPointer();
  bool isArray();
  bool isFunction();

 public:  // get
  auto btype() const { return mBtype; }
  auto size() const { return mSize; }

 public:
  void print(std::ostream& os);

  template <typename T>
  T* as() {
    static_assert(std::is_base_of_v<Type, T>);
    auto ptr = dynamic_cast<T*>(this);
    assert(ptr);
    return ptr;
  }
  template <typename T>
  T* dynCast() {
    static_assert(std::is_base_of_v<Type, T>);
    return dynamic_cast<T*>(this);
  }
};

SYSYC_ARENA_TRAIT(Type, IR);
/* PointerType */
class PointerType : public Type {
  Type* mBaseType;

 public:
  // fix: pointer size is 8 bytes
  PointerType(Type* baseType) : Type(POINTER, 8), mBaseType(baseType) {}
  static PointerType* gen(Type* baseType);

  auto baseType() const { return mBaseType; }
};
/* ArrayType */
class ArrayType : public Type {
 protected:
  std::vector<size_t> mDims;  // dimensions
  Type* mBaseType;            // size_t or float

 public:
  ArrayType(Type* baseType, std::vector<size_t> dims, size_t capacity = 1)
      : Type(ARRAY, capacity * 4), mBaseType(baseType), mDims(dims) {}

  static ArrayType* gen(Type* baseType,
                        std::vector<size_t> dims,
                        size_t capacity = 1);

 public:
  auto dims_cnt() const { return mDims.size(); }
  auto dim(size_t index) const { return mDims[index]; }
  auto& dims() const { return mDims; }
  auto baseType() const { return mBaseType; }
};
/* FunctionType */
class FunctionType : public Type {
 protected:
  Type* mRetType;
  std::vector<Type*> mArgTypes;

 public:  // Gen
  FunctionType(Type* ret_type, const type_ptr_vector& arg_types = {})
      : Type(FUNCTION, 8), mRetType(ret_type), mArgTypes(arg_types) {}
  static FunctionType* gen(Type* ret_type, const type_ptr_vector& arg_types);

  //! get the return type of the function
  auto retType() const { return mRetType; }

  auto& argTypes() const { return mArgTypes; }
};
}