#pragma once
#include "function.hpp"
#include "global.hpp"
#include "infrast.hpp"
#include "utils.hpp"
#include "value.hpp"

namespace ir {

class AllocaInst;
class StoreInst;
class LoadInst;
class ReturnInst;

class CallInst;
class UnaryInst;
class BinaryInst;
class BranchInst;

class ICmpInst;
class FCmpInst;
class CastInst;

class AllocaInst : public Instruction {
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
               const_str_ref name = "",
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
              const_str_ref name = "")
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
             Type* type,
             BasicBlock* parent,
             const_value_ptr_vector& indices = {},
             const_str_ref name = "")
        : Instruction(vLOAD, type, parent, name) {
        // Instruction type?? should be what?
        add_operand(ptr);
        add_operands(indices);
    }

    static LoadInst* gen(Value* ptr,
                         BasicBlock* parent,
                         const_value_ptr_vector& indices = {},
                         const_str_ref name = "") {
        // assert()
        Type* type = nullptr;
        type = dyn_cast<PointerType>(ptr->type())->base_type();
        auto inst = new LoadInst(ptr, type, parent, indices, name);
        return inst;
    }

    Value* ptr() const { return operand(0); }

   public:
    static bool classof(const Value* v) { return v->scid() == vLOAD; }
    void print(std::ostream& os) const override;
};

/*
 * @brief Return Instruction
 * @details:
 *      ret <type> <value>
 *      ret void
 */
class ReturnInst : public Instruction {
    friend class IRBuilder;

   public:
    ReturnInst(Value* value = nullptr,
               BasicBlock* parent = nullptr,
               const_str_ref name = "")
        : Instruction(vRETURN, Type::void_type(), parent, name) {
        add_operand(value);
        // std::cout << _operands.size() << std::endl;
    }

   public:
    bool has_return_value() const { return not _operands.empty(); }
    Value* return_value() const {
        return has_return_value() ? operand(0) : nullptr;
    }

   public:
    static bool classof(const Value* v) { return v->scid() == vRETURN; }
    void print(std::ostream& os) const override;
};

/*
 * @brief Unary Instruction
 * @details:
 *      <result> = sitofp <ty> <value> to <ty2>
 *      <result> = fptosi <ty> <value> to <ty2>
 *
 *      <result> = fneg [fast-math flags]* <ty> <op1>
 */
class UnaryInst : public Instruction {
    friend class IRBuilder;

   protected:
    UnaryInst(Value::ValueId kind,
              Type* type,
              Value* operand,
              BasicBlock* parent = nullptr,
              const_str_ref name = "")
        : Instruction(kind, type, parent, name) {
        add_operand(operand);
    }

   public:
    static bool classof(const Value* v) {
        return v->scid() == vFTOI || v->scid() == vITOF || v->scid() == vFNEG;
    }

   public:
    Value* get_value() const { return operand(0); }

   public:
    void print(std::ostream& os) const override;
};

/*
 * @brief Binary Instruction
 * @details:
 *      1. exp (MUL | DIV | MODULO) exp
 *      2. exp (ADD | SUB) exp
 */
class BinaryInst : public Instruction {
    friend class IRBuilder;

   public:
    BinaryInst(ValueId kind,
               Type* type,
               Value* lvalue,
               Value* rvalue,
               BasicBlock* parent,
               const std::string name = "")
        : Instruction(kind, type, parent, name) {
        add_operand(lvalue);
        add_operand(rvalue);
    }

   public:
    static bool classof(const Value* v) {
        return v->scid() == vADD || v->scid() == vFADD || v->scid() == vSUB ||
               v->scid() == vFSUB || v->scid() == vMUL || v->scid() == vFMUL ||
               v->scid() == vSDIV || v->scid() == vFDIV || v->scid() == vSREM ||
               v->scid() == vFREM;
    }

   public:
    Value* get_lvalue() const { return operand(0); }
    Value* get_rvalue() const { return operand(1); }

   public:
    void print(std::ostream& os) const override;
};
class CallInstBeta : public Instruction {
    //! TODO
    // Function* _callee;
    // const_value_ptr_vector _rargs;

   public:
    CallInstBeta(Function* callee,
                 const_value_ptr_vector rargs = {},
                 BasicBlock* parent = nullptr,
                 const_str_ref name = "")
        : Instruction(vCALL, callee->ret_type(), parent, name) {}

   public:
    // Function* callee() const { return dyn_cast<Function>(operand(0)); }

    static bool classof(const Value* v) {
        //! TODO
        // assert(false && "not implemented");
        return v->scid() == vCALL;
    }
    void print(std::ostream& os) const override;  //! TODO
};

class CallInst : public Instruction {
    //! TODO
    Function* _callee;
    const_value_ptr_vector _rargs;

   public:
    CallInst(Function* callee,
             const_value_ptr_vector rargs = {},
             BasicBlock* parent = nullptr,
             const_str_ref name = "")
        : Instruction(vCALL, callee->ret_type(), parent, name),
          _callee(callee),
          _rargs(rargs) {
        //! TODO
        // add_operands(args);
        // for (auto arg : args) {
        //     add_operand(arg);
        // }
    }

   public:
    Function* callee() const { 
        return _callee;
        // return dyn_cast<Function>(operand(0)); 
        }

    // use_ptr_vector& args() {
    //     // return _operands;
    //     return _args;
    // }
    // std::vector<Use*>::iterator args_begin() { return _operands.begin() + 1;
    // } std::vector<Use*>::iterator args_end() { return _operands.end(); }
    // std::vector<Value*>& args() {
    //     // std::vector<Value*> _operands, 1:end
    //     //TODO
    // }
    static bool classof(const Value* v) {
        //! TODO
        // assert(false && "not implemented");
        return v->scid() == vCALL;
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
               const_str_ref name = "")
        : Instruction(vBR, Type::void_type(), parent, name), _is_cond(true) {
        //! TODO
        // assert(false && "not implemented");
        add_operand(cond);
        add_operand(iftrue);
        add_operand(iffalse);
    }
    // UnCondition Branch
    BranchInst(BasicBlock* dest,
               BasicBlock* parent = nullptr,
               const_str_ref name = "")
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
             const_str_ref name = "")
        : Instruction(itype, Type::i1_type(), parent, name) {  // cmp return i1
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
        return v->scid() >= vICMP && v->scid() <= vICMP_END;
        // return v->scid() == vICMP;
    }
    void print(std::ostream& os) const override;  //! TODO
};

//! FCmpInst
class FCmpInst : public Instruction {
    //! TODO
    //! TODO
   public:
    FCmpInst(ValueId itype,
             Value* lhs,
             Value* rhs,
             BasicBlock* parent,
             const_str_ref name = "")
        : Instruction(itype,
                      Type::i1_type(),
                      parent,
                      name) {  //! return float type?
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
        // return v->scid() == vFCMP;
        return v->scid() >= vFCMP && v->scid() <= vFCMP_END;
    }
    void print(std::ostream& os) const override;  //! TODO
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