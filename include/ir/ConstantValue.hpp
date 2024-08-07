#pragma once
#include "ir/type.hpp"
#include "ir/utils_ir.hpp"
#include "ir/value.hpp"
#include "support/arena.hpp"
#include <variant>
namespace ir {

class ConstantValue : public Value {
  static std::unordered_map<size_t, ConstantValue*> mConstantPool;

public:
  ConstantValue(Type* type) : Value(type, ValueId::vCONSTANT) {}
  virtual size_t hash() const = 0;
  static ConstantValue* findCacheByHash(size_t hash);

  virtual std::variant<std::monostate, intmax_t, double> getValue() const = 0;

  int32_t i32() const;
  float f32() const;
  bool i1() const;

  virtual bool isZero();
  virtual bool isOne();

  static ConstantValue* get(Type* type, std::variant<std::monostate, intmax_t, double> val);
};

class ConstantInteger : public ConstantValue {
  intmax_t mVal;

public:
  ConstantInteger(Type* type, intmax_t val) : ConstantValue(type), mVal(val) {}

  size_t hash() const override;

  intmax_t getVal() const { return mVal; }

  std::variant<std::monostate, intmax_t, double> getValue() const override { return mVal; }

  static ConstantInteger* get(Type* type, intmax_t val);
  static ConstantInteger* getTrue();
  static ConstantInteger* getFalse();

  static ConstantInteger* gen_i32(intmax_t val);
  static ConstantInteger* gen_i1(bool val);

  ConstantInteger* getNeg() const;

  void print(std::ostream& os) const override;
  void dumpAsOpernd(std::ostream& os) const override;
  bool isZero() override { return mVal == 0; }
  bool isOne() override { return mVal == 1; }
};

class ConstantFloating : public ConstantValue {
  double mVal;

public:
  ConstantFloating(Type* type, double val) : ConstantValue(type), mVal(val) {}

  size_t hash() const override;

  double getVal() const { return mVal; }
  std::variant<std::monostate, intmax_t, double> getValue() const override { return mVal; }

  static ConstantFloating* get(Type* type, double val);
  static ConstantFloating* gen_f32(double val);

  ConstantFloating* getNeg() const;

  void print(std::ostream& os) const override;
  void dumpAsOpernd(std::ostream& os) const override;
  bool isZero() override { return mVal == 0.0; }
  bool isOne() override { return mVal == 1.0; }
};

class UndefinedValue : public ConstantValue {
public:
  explicit UndefinedValue(Type* type) : ConstantValue(type) { assert(not type->isVoid()); }

  static UndefinedValue* get(Type* type);

  void print(std::ostream& os) const override { os << "undef"; }
  void dumpAsOpernd(std::ostream& os) const override;
  size_t hash() const override { return 0; }
  std::variant<std::monostate, intmax_t, double> getValue() const override {
    return std::monostate{};
  }
};

}  // namespace ir