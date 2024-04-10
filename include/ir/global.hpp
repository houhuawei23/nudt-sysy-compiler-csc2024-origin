#pragma once
#include "ir/infrast.hpp"
#include "ir/type.hpp"
#include "ir/value.hpp"
#include "support/utils.hpp"
namespace ir {
/**
 * @brief Class GlobalVariable
 * 1 Init Value must be Constant
 * 2
 */
class GlobalVariable : public User {
   protected:
    Module* _parent = nullptr;
    bool _is_array = false;
    bool _is_constant = false;
    int _dimensions = 0;
    std::vector<Value*> _init;

   public:
    GlobalVariable(Type* base_type,
                   const std::vector<Value*>& init,
                   const_value_ptr_vector& dims = {},
                   Module* parent = nullptr,
                   const_str_ref name = "",
                   bool is_constant = false)
        : User(ir::Type::pointer_type(base_type), vGLOBAL_VAR, name),
          _parent(parent),
          _init(init),
          _is_constant(is_constant) {
        _dimensions = dims.size();

        if (dims.size() == 0) {  //! 1. scalar
            _is_array = false;
        } else {  //! 2. array
            _is_array = true;
            add_operands(dims);
        }
    }

   public:  
   // generate function
    static GlobalVariable* gen(Type* base_type,
                               const std::vector<Value*>& init,
                               const_value_ptr_vector& dims = {},
                               Module* parent = nullptr,
                               const_str_ref name = "",
                               bool is_constant = false) {
        auto var = new GlobalVariable(base_type, init, dims, parent, name,
                                      is_constant);
        return var;
    }

   public:  // check function
    bool is_array() const { return _is_array; }
    bool is_constant() const { return _is_constant; }
    bool is_init() const { return _init.size() > 0; }

   public:  // get function
    Module* parent() const { return _parent; }

    int dims_cnt() const {
        if (is_array())
            return _operands.size();
        else
            return 0;
    }

    std::vector<Value*> dims() const {
        std::vector<Value*> ans;
        int dimensions = dims_cnt();
        for (int i = 0; i < dimensions; i++) {
            ans.push_back(operand(i));
        }
        return ans;
    }

    int init_cnt() const { return _init.size(); }

    Value* init(int index) const { return _init[index]; }

    Type* base_type() const {
        return dyn_cast<PointerType>(type())->base_type();
    }

    Value* scalar_value() const { return _init[0]; }

   public:
    void print_ArrayInit(std::ostream& os,
                         const int dimension,
                         const int begin,
                         int* idx) const;

   public:
    static bool classof(const Value* v) { return v->scid() == vGLOBAL_VAR; }

    void print(std::ostream& os) override;
};
}  // namespace ir
