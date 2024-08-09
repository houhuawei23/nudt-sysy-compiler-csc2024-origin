#include "ir/ConstantValue.hpp"
#include "support/arena.hpp"

using namespace ir;

std::unordered_map<size_t, ConstantValue*> constantPool;

ConstantValue* ConstantValue::findCacheByHash(size_t hash) {
  if (const auto iter = constantPool.find(hash); iter != constantPool.cend()) {
    return iter->second;
  }
  return nullptr;
}

int64_t ConstantValue::i64() const {
  const auto val = getValue();
  if (std::holds_alternative<intmax_t>(val)) {
    return static_cast<int64_t>(std::get<intmax_t>(val));
  } else if (std::holds_alternative<double>(val)) {
    return static_cast<int64_t>(std::get<double>(val));
  }
  assert(false);
}
int32_t ConstantValue::i32() const {
  const auto val = getValue();
  if (std::holds_alternative<intmax_t>(val)) {
    return static_cast<int32_t>(std::get<intmax_t>(val));
  } else if (std::holds_alternative<double>(val)) {
    return static_cast<int32_t>(std::get<double>(val));
  }
  assert(false);
}
float ConstantValue::f32() const {
  const auto val = getValue();
  if (std::holds_alternative<double>(val)) {
    return static_cast<float>(std::get<double>(val));
  } else if (std::holds_alternative<intmax_t>(val)) {
    return static_cast<float>(std::get<intmax_t>(val));
  }
  assert(false);
}
bool ConstantValue::i1() const {
  const auto val = getValue();
  if (std::holds_alternative<intmax_t>(val)) {
    return static_cast<int32_t>(std::get<intmax_t>(val));
  }
  assert(false);
}

bool ConstantValue::isZero() {
  if (auto cint = dynCast<ConstantInteger>()) {
    return cint->isZero();
  } else if (auto cfloat = dynCast<ConstantFloating>()) {
    return cfloat->isZero();
  }
  return false;
}
bool ConstantValue::isOne() {
  if (auto cint = dynCast<ConstantInteger>()) {
    return cint->isOne();
  } else if (auto cfloat = dynCast<ConstantFloating>()) {
    return cfloat->isOne();
  }
  return false;
}
size_t ConstantInteger::hash() const {
  return std::hash<intmax_t>{}(mVal);
}

ConstantInteger* ConstantInteger::gen_i64(intmax_t val) {
  return ConstantInteger::get(Type::TypeInt64(), val);
}
ConstantInteger* ConstantInteger::gen_i32(intmax_t val) {
  return ConstantInteger::get(Type::TypeInt32(), val);
}
ConstantInteger* ConstantInteger::gen_i1(bool val) {
  return ConstantInteger::get(Type::TypeBool(), val);
}

ConstantInteger* ConstantInteger::get(Type* type, intmax_t val) {
  if (type->isBool()) {
    return (val & 1) ? getTrue() : getFalse();
  }
  return utils::make<ConstantInteger>(type, val);
}

ConstantInteger* ConstantInteger::getTrue() {
  static auto i1True = utils::make<ConstantInteger>(Type::TypeBool(), 1);
  return i1True;
}

ConstantInteger* ConstantInteger::getFalse() {
  static auto i1False = utils::make<ConstantInteger>(Type::TypeBool(), 0);
  return i1False;
}

ConstantInteger* ConstantInteger::getNeg() const {
  return ConstantInteger::get(mType, -mVal);
}

void ConstantInteger::print(std::ostream& os) const {
  os << mVal;
}
void ConstantInteger::dumpAsOpernd(std::ostream& os) const {
  os << mVal;
}

ConstantFloating* ConstantFloating::gen_f32(double val) {
  return ConstantFloating::get(Type::TypeFloat32(), val);
}

ConstantFloating* ConstantFloating::get(Type* type, double val) {
  return utils::make<ConstantFloating>(type, val);
}
ConstantFloating* ConstantFloating::getNeg() const {
  return ConstantFloating::get(mType, -mVal);
}

void ConstantFloating::print(std::ostream& os) const {
  os << getMC(mVal);  // implicit conversion to float
}
void ConstantFloating::dumpAsOpernd(std::ostream& os) const {
  os << getMC(mVal);  // implicit conversion to float
}

size_t ConstantFloating::hash() const {
  return std::hash<double>{}(mVal);
}

UndefinedValue* UndefinedValue::get(Type* type) {
  return utils::make<UndefinedValue>(type);
}
void UndefinedValue::dumpAsOpernd(std::ostream& os) const {
  print(os);
}
