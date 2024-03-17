#pragma once
#include "infrast.hpp"
#include "type.hpp"
#include "utils.hpp"
#include "value.hpp"
namespace ir {

/*
全局变量类型

*/
class GlobalVariable : public User {
    // _type, _uses, _name
    // _operands
   protected:
    Module* _parent;
    bool _is_init;

   public:
    // ValueId scid,
    // where the global value store?
    // operand
    GlobalVariable(Type* type,
                   const_value_ptr_vector& dims = {},
                   Value* init = nullptr,
                   Module* parent = nullptr,
                   const_str_ref name = "")
        : User(type, vGLOBAL_VAR, name),
          _parent(parent),
          _is_init(init != nullptr) {
        add_operands(dims);
        // TODO:
        // if (ir:)
        // if(dyn_cast<Constant>(init))
        assert(isa<Constant>(init) && "init must be constant");
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
