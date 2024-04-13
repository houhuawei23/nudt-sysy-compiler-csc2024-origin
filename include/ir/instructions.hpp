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

class PhiInst;

/*
 * @brief: AllocaInst
 */
class AllocaInst : public Instruction {
    friend class IRBuilder;

   protected:
    bool _is_const = false;

   public:  // 构造函数
    //! 1. Alloca Scalar
    AllocaInst(Type* base_type,
               BasicBlock* parent = nullptr,
               const_str_ref name = "",
               bool is_const = false)
        : Instruction(vALLOCA, ir::Type::pointer_type(base_type), parent, name),
          _is_const(is_const) {}

    //! 2. Alloca Array
    AllocaInst(Type* base_type,
               std::vector<int> dims,
               BasicBlock* parent = nullptr,
               const_str_ref name = "",
               bool is_const = false)
        : Instruction(
              vALLOCA,
              ir::Type::pointer_type(ir::Type::array_type(base_type, dims)),
              parent,
              name),
          _is_const(is_const) {}

   public:  // get function
    Type* base_type() const {
        assert(dyn_cast<PointerType>(type()) && "type error");
        return dyn_cast<PointerType>(type())->base_type();
    }
    int dims_cnt() const {
        if (base_type()->is_array())
            return dyn_cast<ArrayType>(base_type())->dims_cnt();
        else
            return 0;
    }

   public:  // check function
    bool is_scalar() const { return !base_type()->is_array(); }
    bool is_const() const { return _is_const; }

   public:
    static bool classof(const Value* v) { return v->scid() == vALLOCA; }
    void print(std::ostream& os) override;
};

class StoreInst : public Instruction {
    friend class IRBuilder;

   public:
    StoreInst(Value* value, Value* ptr, BasicBlock* parent = nullptr)
        : Instruction(vSTORE, Type::void_type(), parent) {
        add_operand(value);
        add_operand(ptr);
    }

   public:
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
    LoadInst(Value* ptr, Type* type, BasicBlock* parent)
        : Instruction(vLOAD, type, parent) {
        add_operand(ptr);
    }

    static LoadInst* gen(Value* ptr, BasicBlock* parent) {
        Type* type = nullptr;
        if (ptr->type()->is_pointer())
            type = dyn_cast<PointerType>(ptr->type())->base_type();
        else
            type = dyn_cast<ArrayType>(ptr->type())->base_type();
        auto inst = new LoadInst(ptr, type, parent);
        return inst;
    }

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
    // const_value_ptr_vector _rargs;

   public:
    CallInst(Function* callee,
             const_value_ptr_vector rargs = {},
             BasicBlock* parent = nullptr,
             const_str_ref name = "")
        : Instruction(vCALL, callee->ret_type(), parent, name),
          _callee(callee) {
        add_operands(rargs);
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

/*
 * @brief GetElementPtr Instruction
 * @details:
 *      数组: <result> = getelementptr <type>, <type>* <ptrval>, i32 0, i32
 * <idx> 指针: <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
 * @param:
 *      1. _idx: 数组各个维度的下标索引
 *      2. _id : calculate array address OR pointer address
 */
class GetElementPtrInst : public Instruction {
    friend class IRBuilder;

   protected:
    int _id = 0;
    // Value* _idx = nullptr;
    std::vector<int> _cur_dims = {};

   public:
    //! 1. Pointer <result> = getelementptr <type>, <type>* <ptrval>, i32 <idx>
    GetElementPtrInst(Type* base_type,
                      Value* value,
                      BasicBlock* parent,
                      Value* idx)
        : Instruction(vGETELEMENTPTR,
                      ir::Type::pointer_type(base_type),
                      parent) {
        _id = 0;
        add_operand(value);
        add_operand(idx);
    }

    //! 2. 高维 Array <result> = getelementptr <type>, <type>* <ptrval>, i32 0,
    //! i32 <idx>
    GetElementPtrInst(Type* base_type,
                      Value* value,
                      BasicBlock* parent,
                      Value* idx,
                      std::vector<int> dims,
                      std::vector<int> cur_dims)
        : Instruction(
              vGETELEMENTPTR,
              ir::Type::pointer_type(ir::Type::array_type(base_type, dims)),
              parent),
          _cur_dims(cur_dims) {
        _id = 1;
        add_operand(value);
        add_operand(idx);
    }

    //! 3. 一维 Array <result> = getelementptr <type>, <type>* <ptrval>, i32 0,
    //! i32 <idx>
    GetElementPtrInst(Type* base_type,
                      Value* value,
                      BasicBlock* parent,
                      Value* idx,
                      std::vector<int> cur_dims)
        : Instruction(vGETELEMENTPTR,
                      ir::Type::pointer_type(base_type),
                      parent),
          _cur_dims(cur_dims) {
        _id = 2;
        add_operand(value);
        add_operand(idx);
    }

   public:
    // get function
    Value* get_value() const { return operand(0); }
    Value* get_index() const { return operand(1); }

    Type* base_type() const {
        assert(dyn_cast<PointerType>(type()) && "type error");
        return dyn_cast<PointerType>(type())->base_type();
    }
    int cur_dims_cnt() const { return _cur_dims.size(); }
    std::vector<int> cur_dims() const { return _cur_dims; }

   public:  // check function
    bool is_arrayInst() const { return _id != 0; }

   public:
    static bool classof(const Value* v) { return v->scid() == vGETELEMENTPTR; }
    void print(std::ostream& os) override;
};

class PhiInst : public Instruction {
   protected:
    size_t size;
    std::vector<Value*> _vals;
    std::vector<BasicBlock*> _bbs;

   public:
    PhiInst(BasicBlock* bb,
            Type* type,
            const std::vector<Value*>& vals = {},
            const std::vector<BasicBlock*>& bbs = {})
        : Instruction(vPHI, type, bb), size(vals.size()) {
        add_operands(vals);  // _operands;
        add_operands(bbs);
        _vals = vals;
        _bbs = bbs;
        // f
    }
    auto vals() {
        // return Util::range(_operands.begin(), _operands.begin() + size);
        return _vals;
    }
    auto bbs() {
        // return Util::range(_operands.begin() + size, _operands.begin() + 2 *
        // size);
        return _bbs;
    }
    Value* getvals(size_t k) { return operand(k); }
    Value* getblos(size_t k) { return operand(k + size); }
    void addIncoming(Value* val, BasicBlock* bb) {
        add_operand(val);
        add_operand(bb);
        // parent()->parent()->print(std::cout);
        _vals.push_back(val);
        _bbs.push_back(bb);
        // 更新操作数的数量
        size++;
    }
    void print(std::ostream& os) override;
};

}  // namespace ir
