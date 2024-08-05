#include <assert.h>

#include <variant>  // dynamic_cast

#include <string_view>
using namespace std::string_view_literals;

#include "ir/type.hpp"
#include "support/arena.hpp"
#include "support/StaticReflection.hpp"

namespace ir {
Type* Type::void_type() {
  static Type voidType(BasicTypeRank::VOID);
  return &voidType;
}

Type* Type::TypeBool() {
  static Type i1Type(BasicTypeRank::INT1);
  return &i1Type;
}

// return static Type instance of size_t
Type* Type::TypeInt8() {
  static Type intType(BasicTypeRank::INT8);
  return &intType;
}
Type* Type::TypeInt32() {
  static Type intType(BasicTypeRank::INT32);
  return &intType;
}
Type* Type::TypeFloat32() {
  static Type floatType(BasicTypeRank::FLOAT);
  return &floatType;
}
Type* Type::TypeDouble() {
  static Type doubleType(BasicTypeRank::DOUBLE);
  return &doubleType;
}
Type* Type::TypeLabel() {
  static Type labelType(BasicTypeRank::LABEL);
  return &labelType;
}
Type* Type::TypeUndefine() {
  static Type undefineType(BasicTypeRank::UNDEFINE);
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
  return mBtype == BasicTypeRank::VOID;
}

bool Type::isBool() {
  return mBtype == BasicTypeRank::INT1;
}
bool Type::isInt32() {
  return mBtype == BasicTypeRank::INT32;
}

bool Type::isFloat32() {
  return mBtype == BasicTypeRank::FLOAT;
}
bool Type::isDouble() {
  return mBtype == BasicTypeRank::DOUBLE;
}

bool Type::isUndef() {
  return mBtype == BasicTypeRank::UNDEFINE;
}

bool Type::isLabel() {
  return mBtype == BasicTypeRank::LABEL;
}
bool Type::isPointer() {
  return mBtype == BasicTypeRank::POINTER;
}
bool Type::isFunction() {
  return mBtype == BasicTypeRank::FUNCTION;
}
bool Type::isArray() {
  return mBtype == BasicTypeRank::ARRAY;
}

static std::string_view getTypeName(BasicTypeRank btype) {
  switch (btype) {
    case BasicTypeRank::INT1:
      return "i1"sv;
    case BasicTypeRank::INT8:
      return "i8"sv;
    case BasicTypeRank::INT32:
      return "i32"sv;
    case BasicTypeRank::FLOAT:
      return "float"sv;
    case BasicTypeRank::DOUBLE:
      return "double"sv;
    case BasicTypeRank::VOID:
      return "void"sv;
    case BasicTypeRank::LABEL:
      return "label"sv;
    case BasicTypeRank::POINTER:
      return "pointer"sv;
    case BasicTypeRank::FUNCTION:
      return "function"sv;
    case BasicTypeRank::ARRAY:
      return "array"sv;
    case BasicTypeRank::UNDEFINE:
      return "undefine"sv;
    default:
      std::cerr << "unknown BasicTypeRank: " << utils::enumName(btype) << std::endl;
      assert(false);
      return "";
  }
}

//! print
void Type::print(std::ostream& os) const {
  os << getTypeName(mBtype);
}

PointerType* PointerType::gen(Type* base_type) {
  return utils::make<PointerType>(base_type);
}

void PointerType::print(std::ostream& os) const {
  mBaseType->print(os);
  os << "*";
}

ArrayType* ArrayType::gen(Type* baseType, std::vector<size_t> dims, size_t capacity) {
  return utils::make<ArrayType>(baseType, dims, capacity);
}

void ArrayType::print(std::ostream& os) const {
  for (size_t i = 0; i < mDims.size(); i++) {
    size_t value = mDims[i];
    os << "[" << value << " x ";
  }
  mBaseType->print(os);
  for (size_t i = 0; i < mDims.size(); i++)
    os << "]";
}

FunctionType* FunctionType::gen(Type* ret_type, const type_ptr_vector& arg_types) {
  return utils::make<FunctionType>(ret_type, arg_types);
}
/** void (i32, i32) */
void FunctionType::print(std::ostream& os) const {
  mRetType->print(os);
  os << " (";
  bool isFirst = true;
  for (auto arg_type : mArgTypes) {
    if (isFirst)
      isFirst = false;
    else
      os << ", ";
    arg_type->print(os);
  }
  os << ")";
}

}  // namespace ir