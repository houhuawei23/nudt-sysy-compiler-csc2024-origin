#include "infrast.hpp"
#include "value.hpp"

#include <vector>
namespace ir {
class AllocaInst : public Instruction {
    // using vect
    friend class IRBuilder;

  protected:
    bool _is_const;

  public:
    AllocaInst(Type *type, BasicBlock *parent = nullptr,
               const std::vector<Value *> &dims = {},
               const std::string &name = "", bool is_const = false)
        : Instruction(ALLOCA, type, name, parent), _is_const(is_const) {
        add_operands(dims);
        // more if (is_const and arr)
    }

  public:
    void print(std::ostream &os) const override;
};

class StoreInst : public Instruction {
    friend class IRBuilder;

  public:
    /**
     * @brief Construct a new Store Inst object
     * @details `store [volatile] <ty> <value>, ptr <pointer>`
     *
     * @param value
     * @param ptr
     * @param parent
     * @param indices
     * @param name
     */
    StoreInst(Value *value, Value *ptr, BasicBlock *parent = nullptr,
              const std::vector<Value *> &indices = {},
              const std::string &name = "")
        : Instruction(STORE, Type::void_type(), name, parent) {
        add_operand(value);
        add_operand(ptr);
        add_operands(indices);
    }

    Value *value() const { return operand(0); }
    Value *ptr() const { return operand(1); }

  public:
    void print(std::ostream &os) const override;
};

class LoadInst : public Instruction {
    friend class IRBuilder;

  public:
    //<result> = load [volatile] <ty>, ptr <pointer>
    LoadInst(Value *ptr, BasicBlock *parent,
             const std::vector<Value *> &indices = {},
             const std::string &name = "")
        : Instruction(LOAD, ptr->type()->as<PointerType>()->base_type(),
                      name, parent) {
        // Instruction type?? should be what?
        add_operand(ptr);
        add_operands(indices);
    }

  public:
    void print(std::ostream &os) const override;
};

class ReturnInst : public Instruction {
    friend class IRBuilder;

  public:
    // ret <type> <value>
    // ret void
    ReturnInst(Value *value = nullptr, BasicBlock *parent = nullptr)
        : Instruction(RET, Type::void_type(), "ret", parent) {
        add_operand(value);
    }

    bool has_return_value() const { return not _operands.empty(); }
    Value *return_value() const {
        return has_return_value() ? operand(0) : nullptr;
    }

  public:
    void print(std::ostream &os) const override;
};

} // namespace ir