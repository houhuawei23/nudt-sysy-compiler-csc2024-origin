#pragma once

#include "ir/type.hpp"
#include "ir/utils_ir.hpp"
#include "ir/value.hpp"

namespace ir {
/**
 * @brief Constant Value in IR
 * Same Constant val is ther same Object, managed by the cache
 */
class Constant : public User {
    static std::map<std::string, Constant*> cache;

   protected:
    union {
        bool _i1;
        int32_t _i32;
        float _f32;
        double _f64;
    };

   public:
    Constant(bool i1, const_str_ref name)
        : User(Type::i1_type(), vCONSTANT, name), _i1(i1) {}
    Constant(int32_t i32, const_str_ref name)
        : User(Type::i32_type(), vCONSTANT, name), _i32(i32) {}
    Constant(float f32, const_str_ref name)
        : User(Type::float_type(), vCONSTANT, name), _f32(f32) {}
    Constant(double f64, const_str_ref name)
        : User(Type::double_type(), vCONSTANT, name), _f64(f64) {}
    Constant() : User(Type::void_type(), vCONSTANT, "VOID") {}

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

    //* generate function: gen_xx(val), gen_xx(val, name)
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

        int32_t num = (int32_t)v;
        std::string name = std::to_string(num);
        return cache_add(num, name);
    }
    template <typename T>
    static Constant* gen_i32(T v, std::string name) {
        assert(std::is_integral_v<T> ||
               std::is_floating_point_v<T> && "not int or float!");

        int32_t num = (int32_t)v;
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
    // static Constant* 

   public:

    // get function
    int32_t i32() const {
        if (not is_i32()) {
            std::cerr << "Implicit type conversion!" << std::endl;
            return (int32_t)_i32;
        }
        assert(is_i32() && "not i32!");
        return _i32;
    }
    float f32() const {
        if (not is_float32()) {
            std::cerr << "Implicit type conversion!" << std::endl;
            return (float)_f32;
        }
        // assert(is_float32() && "not f32!");
        return _f32;
    }
    double f64() const {
        assert(is_float() && "not f64!");
        return _f64;
    }
    // get f by catural type
    template <typename T>
    T f() const {
        if (is_float32()) {
            return _f32;
        } else if (is_double()) {
            return _f64;
        }
    }

   public:
    static bool classof(const Value* v) { return v->scid() == vCONSTANT; }

    void print(std::ostream& os) override;
};
}