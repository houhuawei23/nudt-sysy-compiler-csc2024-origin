#pragma once
#include "ir/function.hpp"
#include "ir/global.hpp"
#include "ir/infrast.hpp"
#include "ir/value.hpp"
#include "support/utils.hpp"

namespace ir {

class AllocaInst;
class LoadInst;
class StoreInst;
class GetElementPtrInst;

class ReturnInst;
class BranchInst;

class UnaryInst;
class BinaryInst;

class ICmpInst;
class FCmpInst;
class CallInst;

class CastInst;

/*
 * @brief Class AllocaInst
 * @maninclude
 *      1. Variable
 *      2. Constant Array (Global OR Local)
 */
class AllocaInst : public Instruction {
    friend class IRBuilder;

   protected:
    bool _is_const = false;

   public:
    AllocaInst(Type* base_type,
               BasicBlock* parent = nullptr,
               const_value_ptr_vector& dims = {},
               const_str_ref name = "",
               bool is_const = false)
        : Instruction(vALLOCA, ir::Type::pointer_type(base_type), parent, name),
          _is_const(is_const) {
        add_operands(dims);
    }

   public:  // get function
    Type* base_type() const {
        return dyn_cast<PointerType>(type())->base_type();
    }

    int dims_cnt() const { return operands_cnt(); }

    std::vector<Value*> dims() const {
        std::vector<Value*> ans;
        int dimensions = dims_cnt();
        for (int i = 0; i < dimensions; i++) {
            ans.push_back(operand(i));
        }
        return ans;
    }

   public:  // check function
    bool is_scalar() const { return dims_cnt() == 0; }
    bool is_const() const { return _is_const; }

   public:
    static bool classof(const Value* v) { return v->scid() == vALLOCA; }
    void print(std::ostream& os) override;
};

class StoreInst : public Instruction {
    friend class IRBuilder;

   public:
    StoreInst(Value* value,
              Value* ptr,
              BasicBlock* parent = nullptr,
              const_str_ref name = "")
        : Instruction(vSTORE, Type::void_type(), parent, name) {
        add_operand(value);
        add_operand(ptr);
    }

    Value* value() const { return operand(0); }
    Value* ptr() const { return operand(1); }

   public:
    static bool classof(const Value* v) { return v->scid() == vSTORE; }

    void print(std::ostream& os) override;
};

/*
 * @brief Load Instruction
 * @details:
 *      <result> = load <ty>, ptr <pointer>
 */
class LoadInst : public Instruction {
    friend class IRBuilder;

   public:
    LoadInst(Value* ptr,
             Type* type,
             BasicBlock* parent,
             const_value_ptr_vector& indices = {},
             const_str_ref name = "")
        : Instruction(vLOAD, type, parent, name) {
        add_operand(ptr);
        add_operands(indices);
    }

    static LoadInst* gen(Value* ptr,
                         BasicBlock* parent,
                         const_value_ptr_vector& indices = {},
                         const_str_ref name = "") {
        Type* type = nullptr;
        type = dyn_cast<PointerType>(ptr->type())->base_type();
        auto inst = new LoadInst(ptr, type, parent, indices, name);
        return inst;
    }
    // int dims_cnt() const { return operands_cnt(); }
    // bool is_scalar() const { return dims_cnt() == 0; }

    Value* ptr() const { return operand(0); }

   public:
    static bool classof(const Value* v) { return v->scid() == vLOAD; }
    void print(std::ostream& os) override;
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
        if (value) {
            add_operand(value);
        }
    }

   public:
    bool has_return_value() const { return not _operands.empty(); }
    Value* return_value() const {
        return has_return_value() ? operand(0) : nullptr;
    }

   public:
    static bool classof(const Value* v) { return v->scid() == vRETURN; }
    void print(std::ostream& os) override;
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
        return v->scid() >= vUNARY_BEGIN && v->scid() <= vUNARY_END;
    }

   public:
    Value* get_value() const { return operand(0); }

   public:
    void print(std::ostream& os) override;
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
    void print(std::ostream& os) override;
};

class CallInst : public Instruction {
    Function* _callee = nullptr;
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
    Function* callee() const { return _callee; }

    static bool classof(const Value* v) { return v->scid() == vCALL; }
    void print(std::ostream& os) override;
};

//! Conditional or Unconditional Branch instruction.
class BranchInst : public Instruction {
    // br i1 <cond>, label <iftrue>, label <iffalse>
    // br label <dest>
    bool _is_cond = false;
    //! TODO
   public:
    //* Condition Branch
    BranchInst(Value* cond,
               BasicBlock* iftrue,
               BasicBlock* iffalse,
               BasicBlock* parent = nullptr,
               const_str_ref name = "")
        : Instruction(vBR, Type::void_type(), parent, name), _is_cond(true) {
        add_operand(cond);
        add_operand(iftrue);
        add_operand(iffalse);
    }
    //* UnCondition Branch
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
    static bool classof(const Value* v) { return v->scid() == vBR; }
    void print(std::ostream& os) override;
};

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
        return v->scid() >= vICMP_BEGIN && v->scid() <= vICMP_END;
    }
    void print(std::ostream& os) override;
};

//! FCmpInst
class FCmpInst : public Instruction {
   public:
    FCmpInst(ValueId itype,
             Value* lhs,
             Value* rhs,
             BasicBlock* parent,
             const_str_ref name = "")
        : Instruction(itype,
                      Type::i1_type(),  // also return i1
                      parent,
                      name) {
        add_operand(lhs);
        add_operand(rhs);
    }

   public:
    Value* lhs() const { return operand(0); }
    Value* rhs() const { return operand(1); }

   public:
    static bool classof(const Value* v) {
        return v->scid() >= vFCMP_BEGIN && v->scid() <= vFCMP_END;
    }
    void print(std::ostream& os) override;
};

//! CastInst
/// This is the base class for all instructions that perform data
/// casts. It is simply provided so that instruction category testing
/// can be performed with code like:
///
/// if (isa<CastInst>(Instr)) { ... }
/// Base class of casting instructions.
// class CastInst : public Instruction {
//     CastInst(ValueId itype,
//               Type* type,
//              Value* src_val,
//              BasicBlock* parent = nullptr)
//         : Instruction(itype, type, parent) {
//         add_operand(src_val);
//     }

//     void print(std::ostream& os) override;
// };

/*
 * @brief GetElementPtr Instruction
 * @details:
 *      数组: <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32
 * <idx> 指针: <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
 * @param:
 *      1. _current_dimension: 当前数组维度
 *      2. _idx: 数组各个维度的下标索引
 *      3. _id : calculate array address OR pointer address
 */
class GetElementPtrInst : public Instruction {
    friend class IRBuilder;

   protected:
    int _id = 0, _current_dimension = 0;
    Value* _idx = nullptr;

   public:
    //! 1. Array GetElementPtr Instruction
    GetElementPtrInst(Type* base_type,
                      Value* value,
                      BasicBlock* parent,
                      Value* idx,
                      const_value_ptr_vector& dims = {},
                      const int current_dimension = 1,
                      const std::string name = "",
                      int id = 1)
        : Instruction(vGETELEMENTPTR,
                      ir::Type::pointer_type(base_type),
                      parent,
                      name),
          _current_dimension(current_dimension),
          _idx(idx),
          _id(id) {
        add_operand(value);
        add_operands(dims);
    }

   public:
    static bool classof(const Value* v) { return v->scid() == vGETELEMENTPTR; }

   public:  // get function
    int dims_cnt() const { return operands_cnt() - 1; }
    int current_dimension() const { return _current_dimension; }
    Value* get_value() const { return operand(0); }
    Value* get_index() const { return _idx; }
    Type* base_type() const {
        return dyn_cast<PointerType>(type())->base_type();
    }

   public:  // check function
    bool is_arrayInst() const { return _id == 1; }

   public:
    void print(std::ostream& os) override;
};

}  // namespace ir