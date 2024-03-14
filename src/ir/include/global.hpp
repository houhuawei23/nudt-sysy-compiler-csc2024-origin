#pragma once
#include "infrast.hpp"
#include "type.hpp"
#include "value.hpp"
namespace ir {
using const_str = const std::string;
using const_value_vector = const std::vector<Value*>;
/*
全局变量类型

*/
class GlobalVariable : public Constant {
    // _type, _uses, _name
    // _operands
   protected:
    Module* _parent;
    bool _is_const;
    bool _is_init;

   public:
    // ValueId scid,
    // where the global value store?
    // operand
    GlobalVariable(Type* base_type,
                   const_value_vector& dims = {},
                   Value* init = nullptr,
                   Module* parent = nullptr,
                   bool is_const = false,
                   const_str& name = "")
        : Constant(base_type, vGLOBAL_VAR, name),
          _parent(parent),
          _is_const(is_const),
          _is_init(init != nullptr) {
        add_operands(dims);
        if (init) {
            add_operand(init);
        }
        //! TODO: if array
    }

   public:
    // get data attributes
    Module* parent() const { return _parent; }
    bool is_const() const { return _is_const; }
    bool is_init() const { return _is_init; }

    int dims_num() const {
        // int a[2][3]; return 2
        return _operands.size() - (_is_init ? 1 : 0);
    }
    VAlue* dim(int i) { return operand(i); }
    //

   public:
    static bool classof(const Value* v) { return v->scid() == scid(); }

    void print(std::ostream& os) const override;
};
}  // namespace ir
