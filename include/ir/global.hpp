#pragma once
#include "ir/infrast.hpp"
#include "ir/type.hpp"
#include "ir/value.hpp"
#include "support/utils.hpp"
namespace ir {
/* GlobalVariable */
/*
 * @brief: GlobalVariable Class
 * @note: 
 *      1. Array (全局矢量)
 *      2. Scalar (全局标量)
 * @var: 
 *      1. _parent: 当前
 */
class GlobalVariable : public User {
protected:
    Module* _parent = nullptr;
    bool _is_array = false;
    bool _is_const = false;
    std::vector<Value*> _init;
public:
    //! 1. Array
    GlobalVariable(Type* base_type, std::vector<Value*> init, std::vector<int> dims,
                   Module* parent=nullptr, const_str_ref name="", bool is_const=false)
        : User(ir::Type::pointer_type(ir::Type::array_type(base_type, dims)), vGLOBAL_VAR, name),
          _parent(parent), _init(init), _is_const(is_const), _is_array(true) {}
    //! 2. Scalar
    GlobalVariable(Type* base_type, std::vector<Value*> init, Module* parent=nullptr,
                   const_str_ref name="", bool is_const=false)
        : User(ir::Type::pointer_type(base_type), vGLOBAL_VAR, name),
          _parent(parent), _init(init), _is_const(is_const), _is_array(false) {}
public:  // generate function
    static GlobalVariable* gen(Type* base_type, std::vector<Value*> init, Module* parent=nullptr,
                               const_str_ref name="", bool is_const=false, std::vector<int> dims={}) {
        GlobalVariable* var = nullptr;
        if (dims.size() == 0) var = new GlobalVariable(base_type, init, parent, name, is_const);
        else var = new GlobalVariable(base_type, init, dims, parent, name, is_const);
        return var;
    }
public:  // check function
    bool is_array() const { return _is_array; }
    bool is_const() const { return _is_const; }
    bool is_init() const { return _init.size() > 0; }
public:  // get function
    Module* parent() const { return _parent; }
    int dims_cnt() const {
        if (is_array()) return dyn_cast<ArrayType>(base_type())->dims_cnt();
        else return 0;
    }
    int init_cnt() const { return _init.size(); }
    Value* init(int index) const { return _init[index]; }
    Type* base_type() const {
        assert(dyn_cast<PointerType>(type()) && "type error");
        return dyn_cast<PointerType>(type())->base_type();
    }
    Value* scalar_value() const { return _init[0]; }
public:
    void print_ArrayInit(std::ostream& os, const int dimension,
                         const int begin, int* idx) const;
public:
    static bool classof(const Value* v) { return v->scid() == vGLOBAL_VAR; }
    std::string name() const override { return "@" + _name; }
    void print(std::ostream& os) override;
};
}  // namespace ir