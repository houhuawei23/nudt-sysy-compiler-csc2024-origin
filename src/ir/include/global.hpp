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
    bool _is_decl_const;

   public:
    // ValueId scid,
    // where the global value store?
    // operand
    GlobalVariable(Type* type,  // PointerType
                   const_value_ptr_vector& dims = {},
                   Constant* init = nullptr,
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
        assert(dyn_cast<PointerType>(type)->base_type() == init->type() &&
               "type must be equal to init type");

        if (init) {
            add_operand(init);
        }
        //! TODO: if array
    }

   public:
    static GlobalVariable* gen(Type* base_type,
                               const_value_ptr_vector& dims = {},
                               Constant* cinit = nullptr,
                               Module* parent = nullptr,
                               const_str_ref name = "") {
        if (cinit && base_type != cinit->type()) {
            if (base_type->is_i32() && cinit->is_float()) {
                cinit = ir::Constant::gen_i32(cinit->f64());
            } else if (base_type->is_float() && cinit->is_i32()) {
                cinit = ir::Constant::gen_f64(cinit->i32());
            } else {
                assert(false && "type mismatch");
            }
        }
        auto ptr_type = Type::pointer_type(base_type);
        auto ngv = new GlobalVariable(ptr_type, dims, cinit, parent, name);
        return ngv;
    }

    // get data attributes
    Module* parent() const { return _parent; }
    // bool is_const() const { return _is_const; }
    bool is_init() const { return _is_init; }
    bool is_decl_const() const { return _is_decl_const; }

    Value* init_value() const {
        assert(_is_init && "not init");
        return operand(_operands.size() - 1);
    }
    int dims_num() const {
        // int a[2][3]; return 2
        return _operands.size() - (_is_init ? 1 : 0);
    }
    Value* dim(int i) { return operand(i); }
    //

   public:
    static bool classof(const Value* v) { return v->scid() == vGLOBAL_VAR; }

    void print(std::ostream& os) const override;
};
}  // namespace ir
