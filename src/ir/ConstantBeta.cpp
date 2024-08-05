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
bool ConstantValue::isZero() {
  if (auto cint = dynCast<ConstantInteger>()) {
    return cint->isZero();
  } else if (auto cfloat = dynCast<ConstantFloating>()) {
    return cfloat->isZero();
  }
}
bool ConstantValue::isOne() {
  if (auto cint = dynCast<ConstantInteger>()) {
    return cint->isOne();
  } else if (auto cfloat = dynCast<ConstantFloating>()) {
    return cfloat->isOne();
  }
}
size_t ConstantInteger::hash() const {
  return std::hash<intmax_t>{}(mVal);
}

ConstantInteger* ConstantInteger::gen_i32(intmax_t val) {
  return ConstantInteger::get(Type::TypeInt32(), val);
}
ConstantInteger* ConstantInteger::gen_i1(bool val) {
  return ConstantInteger::get(Type::TypeBool(), val);
}

ConstantInteger* ConstantInteger::get(Type* type, intmax_t val) {
  // ConstantInteger* ret = nullptr;
  // find in constant pool
  // size_t hashVal = std::hash<intmax_t>{}(val);
  // if(auto cache = ConstantValue::findCacheByHash(hashVal)) {
  //   ret = cache->dynCast<ConstantInteger>();
  // }

  if (type->isBool()) {
    return (val & 1) ? getTrue() : getFalse();
  }

  if (val == 0) {
    assert(type->isInt32());
    static auto i32Zero = utils::make<ConstantInteger>(Type::TypeInt32(), 0);
    return i32Zero;
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
