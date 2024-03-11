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
        : Instruction(ALLOCA, type, parent, name), _is_const(is_const) {
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
        : Instruction(STORE, Type::void_type(), parent, name) {
        add_operand(value);
        add_operand(ptr);
        add_operands(indices);
    }

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
        : Instruction(
              LOAD,
              dynamic_cast<PointerType *>(ptr->get_type())->get_base_type(),
              parent, name) {
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
        : Instruction(RET, Type::void_type(), parent, "ret") {
        add_operand(value);
    }

    bool has_ReturnValue() const { return not _operands.empty(); }
    Value* get_ReturnValue() {
      return has_ReturnValue() ? get_operand(0) : nullptr;
    }

  public:
    void print(std::ostream &os) const override;
};

} // namespace ir