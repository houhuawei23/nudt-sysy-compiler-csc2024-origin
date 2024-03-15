#pragma once
#include "infrast.hpp"
#include "utils.hpp"
#include "value.hpp"

#include <vector>
namespace ir {

using const_value_ptr_vector = const std::vector<Value*>;
using value_ptr_vector = std::vector<Value*>;
using const_str = const std::string;

class AllocaInst;
class StoreInst;
class LoadInst;
class ReturnInst;

class CallInst;
class UnaryInst;
class BinaryInst;
class BranchInst;

class CmpInst;
class CastInst;

class AllocaInst : public Instruction {
    // using
    friend class IRBuilder;

   protected:
    bool _is_const;

   public:
    // AllocaInst 的类型是 PointerType 实例
    // 其 base_type() 表征指针的基类型
    // alloca i32
    AllocaInst(Type* base_type,  // i32 type
               BasicBlock* parent = nullptr,
               const_value_ptr_vector& dims = {},
               const_str& name = "",
               bool is_const = false)
        : Instruction(vALLOCA, ir::Type::pointer_type(base_type), parent, name),
          _is_const(is_const) {
        add_operands(dims);
        // more if (is_const and arr)
    }

    Type* base_type() const {
        return dyn_cast<PointerType>(type())->base_type();
    }
    bool is_const() const { return _is_const; }
    int dims_cnt() const { return operands_cnt(); }
    bool is_scalar() const { return dims_cnt() == 0; }

   public:
    static bool classof(const Value* v) { return v->scid() == vALLOCA; }
    void print(std::ostream& os) const override;
};

class StoreInst : public Instruction {
    friend class IRBuilder;

   public:
    /**
     * @brief Construct a new Store Inst object
     * @details `store [volatile] <ty> <value>, ptr <pointer>`
     *
     */
    StoreInst(Value* value,
              Value* ptr,
              BasicBlock* parent = nullptr,
              const_value_ptr_vector& indices = {},
              const_str& name = "")
        : Instruction(vSTORE, Type::void_type(), parent, name) {
        add_operand(value);
        add_operand(ptr);
        add_operands(indices);
    }

    Value* value() const { return operand(0); }
    Value* ptr() const { return operand(1); }

   public:
    static bool classof(const Value* v) { return v->scid() == vSTORE; }

    void print(std::ostream& os) const override;
};

class LoadInst : public Instruction {
    friend class IRBuilder;

   public:
    //<result> = load [volatile] <ty>, ptr <pointer>
    LoadInst(Value* ptr,
             BasicBlock* parent,
             const_value_ptr_vector& indices = {},
             const_str& name = "")
        : Instruction(vLOAD,
                      dyn_cast<PointerType>(ptr->type())->base_type(),
                      parent,
                      name) {
        // Instruction type?? should be what?
        add_operand(ptr);
        add_operands(indices);
    }

    Value* ptr() const { return operand(0); }
    // int dims_cnt() const { return operands_cnt(); }

   public:
    static bool classof(const Value* v) { return v->scid() == vLOAD; }
    void print(std::ostream& os) const override;
};

class ReturnInst : public Instruction {
    friend class IRBuilder;

   public:
    // ret <type> <value>
    // ret void
    ReturnInst(Value* value = nullptr,
               BasicBlock* parent = nullptr,
               const_str& name = "")
        : Instruction(vRETURN, Type::void_type(), parent, name) {
        add_operand(value);
    }

    bool has_return_value() const { return not _operands.empty(); }
    Value* return_value() const {
        return has_return_value() ? operand(0) : nullptr;
    }

   public:
    static bool classof(const Value* v) { return v->scid() == vRETURN; }
    void print(std::ostream& os) const override;
};

//! Unary instruction, includes '!', '-' and type conversion.
class UnaryInst : public Instruction {
    //! TODO
   protected:
    // Constructor
    UnaryInst(ValueId itype,
              Value* operand,
              BasicBlock* parent = nullptr,
              const_str& name = "")
        : Instruction(itype,
                      Type::void_type(),  //! TODO: modified ret_type
                      parent,
                      name) {
        add_operand(operand);
    }

   public:
    static bool classof(const Value* v) {
        //! TODO
        assert(false && "not implemented");
    }
    void print(std::ostream& os) const override;  //! TODO
};

/*
 * @brief Binary Instruction
 * @details: 
 *      1. exp (MUL | DIV | MODULO) exp
 *      2. exp (ADD | SUB) exp
 *      3. exp (LT | GT | LE | GE) exp
 *      4. exp (EQ | NE) exp
 *      5. exp AND exp
 *      6. exp OR exp
 */
class BinaryInst : public Instruction {
   public:
    BinaryInst(ValueId kind, Type* type, Value* lvalue, Value* rvalue, BasicBlock* parent, const std::string name="") 
     : Instruction(kind, type, parent, name) {
        add_operand(lvalue);
        add_operand(rvalue);
    }

   public:
    static bool classof(const Value* v) {
        // TODO
        assert(false && "not implemented");
    }

    public:
    Value* get_lvalue() const { return operand(0); }
    Value* get_rvalue() const { return operand(1); }


    void print(std::ostream& os) const override;
};

class CallInst : public Instruction {
    //! TODO
   protected:
    CallInst() {
        //! TODO
        assert(false && "not implemented");
    }

   public:
    static bool classof(const Value* v) {
        //! TODO
        assert(false && "not implemented");
    }
    void print(std::ostream& os) const override;  //! TODO
};

//! Conditional or Unconditional Branch instruction.
class BranchInst : public Instruction {
    // `br i1 <cond>, label <iftrue>, label <iffalse>`
    // br label <dest>
    bool _is_cond;
    //! TODO
   public:
    // Condition Branch
    BranchInst(Value* cond,
               BasicBlock* iftrue,
               BasicBlock* iffalse,
               BasicBlock* parent = nullptr,
               const_str& name = "")
        : Instruction(vBR, Type::void_type(), parent, name), _is_cond(true) {
        //! TODO
        // assert(false && "not implemented");
        add_operand(cond);
        add_operand(iftrue);
        add_operand(iffalse);
    }
    // UnCondition Branch
    BranchInst(BasicBlock* dest, BasicBlock* parent = nullptr, const_str& name="")
        : Instruction(vBR, Type::void_type(), parent, name), _is_cond(false) {
        add_operand(dest);
    }

    // get
    bool is_cond() const { return _is_cond; }
    Value* cond() const {
        assert(_is_cond && "not a conditional branch");
        return operand(0);
    }
    BasicBlock* iftrue() const {
        assert(_is_cond && "not a conditional branch");
        return dyn_cast<BasicBlock>(operand(1));
    }
    BasicBlock* iffalse() const {
        assert(_is_cond && "not a conditional branch");
        return dyn_cast<BasicBlock>(operand(2));
    }
    BasicBlock* dest() const {
        assert(!_is_cond && "not an unconditional branch");
        return dyn_cast<BasicBlock>(operand(0));
    }

   public:
    static bool classof(const Value* v) {
        //! TODO
        // assert(false && "not implemented");
        return v->scid() == vBR;
    }
    void print(std::ostream& os) const override;  //! TODO
};

/// This class is the base class for the comparison instructions.
/// Abstract base class of comparison instructions.
//! CmpInst
// class CmpInst : public Instruction {
//     //! TODO
// };

//! ICmpInst
//! <result> = icmp <cond> <ty> <op1>, <op2>
// icmp ne i32 1, 2
class ICmpInst : public Instruction {
    //! TODO
   public:
    ICmpInst(ValueId itype,
             Value* lhs,
             Value* rhs,
             BasicBlock* parent,
             const_str& name = "")
        : Instruction(itype, Type::int_type(), parent, name) {
        add_operand(lhs);
        add_operand(rhs);
    }

   public:
    Value* lhs() const { return operand(0); }
    Value* rhs() const { return operand(1); }

   public:
    static bool classof(const Value* v) {
        //! TODO
        // assert(false && "not implemented");
        return v->scid() != vICMP;
    }
    void print(std::ostream& os) const override;  //! TODO
};

//! FCmpInst
class FCmpInst : public Instruction {
    //! TODO
};

//! CastInst
/// This is the base class for all instructions that perform data
/// casts. It is simply provided so that instruction category testing
/// can be performed with code like:
///
/// if (isa<CastInst>(Instr)) { ... }
/// Base class of casting instructions.
class CastInst : public Instruction {
    //! TODO
};

}  // namespace ir