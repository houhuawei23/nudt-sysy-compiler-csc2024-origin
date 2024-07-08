#pragma once
#include "ir/type.hpp"
#include "ir/utils_ir.hpp"
#include "ir/value.hpp"

namespace ir {
/* Constant Class */
class Constant : public User {
  static std::map<std::string, Constant*> cache;

 private:
  bool _is_zero = false;

 protected:
  union {
    bool mInt1;
    int64_t mInt32;
    int64_t mInt64;
    float mFloat32;
    double mDouble;
  };

 public:
  Constant(bool val, const_str_ref name)
      : User(Type::TypeBool(), vCONSTANT, name), mInt1(val) {}

  Constant(int32_t i32, const_str_ref name)
      : User(Type::TypeInt32(), vCONSTANT, name),
        mInt32(i32),
        _is_zero(i32 == 0) {}

  Constant(int64_t i64, const_str_ref name)
      : User(Type::TypeInt32(), vCONSTANT, name),
        mInt32(i64),
        _is_zero(i64 == 0) {}

  Constant(float f32, const_str_ref name)
      : User(Type::TypeFloat32(), vCONSTANT, name),
        mFloat32(f32),
        _is_zero(f32 == 0.0) {}

  Constant(double f64, const_str_ref name)
      : User(Type::TypeDouble(), vCONSTANT, name),
        mDouble(f64),
        _is_zero(f64 == 0.0) {}

  Constant() : User(Type::void_type(), vCONSTANT, "VOID") {}
  Constant(const_str_ref name)
      : User(Type::TypeUndefine(), vCONSTANT, "undef") {}

 public:
  //* add constant to cache
  template <typename T>
  static Constant* cache_add(T val, const std::string& name) {
    auto iter = cache.find(name);
    if (iter != cache.end()) {
      return iter->second;
    }

    Constant* c;
    c = new Constant(val, name);
    auto res = cache.emplace(name, c);
    return c;
  }

 public:  // generate function
  template <typename T>
  static Constant* gen_i1(T v) {
    assert(std::is_integral_v<T> ||
           std::is_floating_point_v<T> && "not int or float!");

    bool num = (bool)v;
    std::string name;
    if (num) {
      name = "true";
    } else {
      name = "false";
    }
    return cache_add(num, name);
  }

  template <typename T>
  static Constant* gen_i32(T v) {
    assert(std::is_integral_v<T> ||
           std::is_floating_point_v<T> && "not int or float!");

    int64_t num = (int64_t)v;
    std::string name = std::to_string(num);
    return cache_add(num, name);
  }
  template <typename T>
  static Constant* gen_i32(T v, std::string name) {
    assert(std::is_integral_v<T> ||
           std::is_floating_point_v<T> && "not int or float!");

    int64_t num = (int64_t)v;
    return cache_add(num, name);
  }

  template <typename T>
  static Constant* gen_f32(T val) {
    assert(std::is_integral_v<T> ||
           std::is_floating_point_v<T> && "not int or float!");

    auto f32 = (float)val;
    auto f64 = (double)val;
    auto name = getMC(f64);
    return cache_add(f32, name);
  }
  template <typename T>
  static Constant* gen_f32(T val, std::string name) {
    assert(std::is_integral_v<T> ||
           std::is_floating_point_v<T> && "not int or float!");

    auto f32 = (float)val;
    auto f64 = (double)val;
    return cache_add(f32, name);
  }
  template <typename T>
  static Constant* gen_f64(T val) {
    assert(std::is_integral_v<T> ||
           std::is_floating_point_v<T> && "not int or float!");

    auto f64 = (double)val;
    auto name = getMC(f64);
    return cache_add(f64, name);
  }
  template <typename T>
  static Constant* gen_f64(T val, std::string name) {
    assert(std::is_integral_v<T> ||
           std::is_floating_point_v<T> && "not int or float!");

    auto f64 = (double)val;
    return cache_add(f64, name);
  }

  static Constant* gen_void() {
    std::string name = "VOID";
    auto iter = cache.find(name);
    if (iter != cache.end()) {
      return iter->second;
    }

    Constant* c;
    c = new Constant();
    auto res = cache.emplace(name, c);
    return c;
  }

  static Constant* gen_undefine() {
    std::string name = "UNDEFINE";
    auto iter = cache.find(name);
    if (iter != cache.end()) {
      return iter->second;
    }

    auto c = new Constant(name);
    auto res = cache.emplace(name, c);
    return c;
  }

 public:
  std::string comment() const override {
    return name();
  }
  // get
  bool i1() const {
    if (not isBool()) {
      std::cerr << "Implicit type conversion!" << std::endl;
      return (bool)mInt1;
    }
    assert(isBool() && "not i1!");
    return mInt1;
  }
  int32_t i32() const {
    if (not isInt32()) {
      std::cerr << "Implicit type conversion!" << std::endl;
      return (int32_t)mFloat32;
    }
    return mInt32;
  }
  float f32() const {
    if (not isFloat32()) {
      std::cerr << "Implicit type conversion!" << std::endl;
      return (float)mInt32;
    }
    return mFloat32;
  }
  double f64() const {
    assert(isFloatPoint() && "not f64!");
    return mDouble;
  }

  template <typename T>
  T f() const {
    if (isFloat32()) {
      return mFloat32;
    } else if (isDouble()) {
      return mDouble;
    }
  }

  //    public:  // check
  bool is_zero() const { return _is_zero; }

 public:
  static bool classof(const Value* v) { return v->valueId() == vCONSTANT; }
  void print(std::ostream& os) const override;

  bool isequal(Constant* c) {
    if (c->valueId() != valueId())
      return false;
    if (c->i32())
      return c->i32() == i32();
    if (c->f32())
      return c->f32() == f32();
    return false;
  }
};
}  // namespace ir