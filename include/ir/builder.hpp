/**
 * @file builder.hpp
 *
 */
#pragma once

#include <any>
#include "ir/infrast.hpp"
#include "ir/instructions.hpp"
namespace ir {

/**
 * @brief IR Builder for Module.
 *
 */
class IRBuilder {
   private:
    BasicBlock* _block = nullptr;  // current basic block for insert instruction
    inst_iterator _pos;            // insert pos for cur block
    block_ptr_stack _headers, _exits;
    int _if_cnt, _while_cnt, _rhs_cnt, _func_cnt, _var_cnt;

    // true/false br targets stack for if-else Short-circuit evaluation
    // the top the deeper nest
    block_ptr_stack _true_targets, _false_targets;
    int _bb_cnt;

   public:
    IRBuilder() {
        _if_cnt = 0;
        _while_cnt = 0;
        _rhs_cnt = 0;
        _func_cnt = 0;
        _var_cnt = 0;
        _bb_cnt = 0;
    }

    void reset() {
        _var_cnt = 0;
        _bb_cnt = 0;
    }

    Value* cast_to_i1(Value* val) {
        Value* res = nullptr;
        if (not val->is_i1()) {
            if (val->is_i32()) {
                res = create_icmp(Value::vINE, val, ir::Constant::gen_i32(0));
            } else if (val->is_float()) {
                res =
                    create_fcmp(Value::vFONE, val, ir::Constant::gen_f32(0.0));
            }
        } else {
            res = val;
        }
        return res;
    }
    // using pair?
    // defalut promote to i32
    Value* type_promote(Value* val,
                        Type* target_tpye,
                        Type* base_type = Type::i32_type()) {
        Value* res = val;
        if (val->type()->btype() < target_tpye->btype()) {
            // need to promote
            if (val->type()->is_i1()) {
                if (target_tpye->is_i32()) {
                    res = create_unary_beta(Value::vZEXT, res, Type::i32_type());

                } else if (target_tpye->is_float()) {
                    res = create_unary_beta(Value::vZEXT, res, Type::i32_type());
                    res = create_unary_beta(Value::vSITOFP, res, Type::float_type());
                }
            } else if (val->type()->is_i32()) {
                if (target_tpye->is_float()) {
                    res = create_unary_beta(Value::vSITOFP, res, Type::float_type());
                }
            }
        }
        return res;
    }

    auto create_cmp(Value::CmpOp op, Value* lhs, Value* rhs) {
        if (lhs->type() != rhs->type()) {
            assert(false && "create_eq_beta: type mismatch!");
        }

        switch (lhs->type()->btype()) {
            case INT32: {
                switch (op) {
                    case Value::EQ:
                        return create_icmp(Value::vIEQ, lhs, rhs);
                    case Value::NE:
                        return create_icmp(Value::vINE, lhs, rhs);
                    case Value::GT:
                        return create_icmp(Value::vISGT, lhs, rhs);
                    case Value::GE:
                        return create_icmp(Value::vISGE, lhs, rhs);
                    case Value::LT:
                        return create_icmp(Value::vISLT, lhs, rhs);
                    case Value::LE:
                        return create_icmp(Value::vISLE, lhs, rhs);
                    default:
                        assert(false && "create_cmp: invalid op!");
                }
            } break;
            case FLOAT:
            case DOUBLE: {
                switch (op) {
                    case Value::EQ:
                        return create_fcmp(Value::vFOEQ, lhs, rhs);
                    case Value::NE:
                        return create_fcmp(Value::vFONE, lhs, rhs);
                    case Value::GT:
                        return create_fcmp(Value::vFOGT, lhs, rhs);
                    case Value::GE:
                        return create_fcmp(Value::vFOGE, lhs, rhs);
                    case Value::LT:
                        return create_fcmp(Value::vFOLT, lhs, rhs);
                    case Value::LE:
                        return create_fcmp(Value::vFOLE, lhs, rhs);
                    default:
                        assert(false && "create_cmp: invalid op!");
                }
            } break;
            default:
                assert(false && "create_eq_beta: type mismatch!");
        }
    }

    auto create_binary_beta(Value::BinaryOp op, Value* lhs, Value* rhs) {
        Type* ltype = lhs->type(), *rtype = rhs->type();
        if (ltype != rtype) {
            assert(false && "create_eq_beta: type mismatch!");
        }
        Value* res = nullptr;
        Value::ValueId vid;
        switch (ltype->btype()) {
            case INT32: {
                switch (op) {
                    case Value::ADD:
                        vid = Value::vADD;
                        break;
                    case Value::SUB:
                        vid = Value::vSUB;
                        break;
                    case Value::MUL:
                        vid = Value::vMUL;
                        break;
                    case Value::DIV:
                        vid = Value::vSDIV;
                        break;
                    case Value::REM:
                        vid = Value::vSREM;
                        break;
                    default:
                        assert(false && "create_binary_beta: invalid op!");
                }
                res = create_binary(vid, Type::i32_type(), lhs, rhs);
            } break;
            case FLOAT: {
                switch (op) {
                    case Value::ADD:
                        vid = Value::vFADD;
                        break;
                    case Value::SUB:
                        vid = Value::vFSUB;
                        break;
                    case Value::MUL:
                        vid = Value::vFMUL;
                        break;
                    case Value::DIV:
                        vid = Value::vFDIV;
                        break;
                    default:
                        assert(false && "create_binary_beta: invalid op!");
                }
                res = create_binary(vid, Type::float_type(), lhs, rhs);
            } break;
            case DOUBLE: {
                assert(false && "create_binary_beta: invalid type!");
            }
        }
        return res;
    }

    Value* create_unary_beta(Value::ValueId vid, Value* val, Type* ty = nullptr) {
        //! check vid
        Value* res = nullptr;

        if (vid == Value::vFNEG) {
            assert(val->type()->is_float() && "fneg must have float operand");
            res = create_unary(Value::vFNEG, Type::float_type(), val);
            return dyn_cast_Value(res);
        }
        //! else
        assert(ty != nullptr && "must have target type");

        switch (vid) {
            case Value::vSITOFP:
                assert(val->type()->is_int() && "sitofp must have int operand");
                assert(ty->is_float() && "sitofp must have float type");
                break;
            case Value::vFPTOSI:
                assert(val->type()->is_float() && "fptosi must have float operand");
                assert(ty->is_int() && "fptosi must have int type");
                break;
            case Value::vTRUNC:
            case Value::vZEXT:
            case Value::vSEXT:
                assert(val->type()->is_int() && ty->is_int());
                break;
            case Value::vFPTRUNC:
                assert(val->type()->is_float() && ty->is_float());
                break;
        }
        res = create_unary(vid, ty, val);
        return dyn_cast_Value(res);
    }
    //! get
    std::string get_bbname() { return "bb" + std::to_string(_bb_cnt++); }
    BasicBlock* block() const { return _block; }
    inst_iterator position() const { return _pos; }

    BasicBlock* header() {
        if (not _headers.empty())
            return _headers.top();
        else
            return nullptr;
    }
    BasicBlock* exit() {
        if (not _exits.empty())
            return _exits.top();
        else
            return nullptr;
    }

    //! manage attributes
    void set_pos(BasicBlock* block, inst_iterator pos) {
        _block = block;
        _pos = pos;  // _pos 与 ->end() 绑定?
    }

    void push_header(BasicBlock* block) { _headers.push(block); }
    void push_exit(BasicBlock* block) { _exits.push(block); }

    void push_loop(BasicBlock* header_block, BasicBlock* exit_block) {
        push_header(header_block);
        push_exit(exit_block);
    }

    void pop_loop() {
        _headers.pop();
        _exits.pop();
    }

    void if_inc() { _if_cnt++; }
    void while_inc() { _while_cnt++; }
    void rhs_inc() { _rhs_cnt++; }
    void func_inc() { _func_cnt++; }

    int if_cnt() const { return _if_cnt; }
    int while_cnt() const { return _while_cnt; }
    int rhs_cnt() const { return _rhs_cnt; }
    int func_cnt() const { return _func_cnt; }

    void push_true_target(BasicBlock* block) { _true_targets.push(block); }
    void push_false_target(BasicBlock* block) { _false_targets.push(block); }
    void push_tf(BasicBlock* true_block, BasicBlock* false_block) {
        _true_targets.push(true_block);
        _false_targets.push(false_block);
    }

    // current stmt or exp 's true/false_target
    BasicBlock* true_target() { return _true_targets.top(); }
    BasicBlock* false_target() { return _false_targets.top(); }

    void pop_tf() {
        _true_targets.pop();
        _false_targets.pop();
    }

    //! Create Alloca Instruction
    AllocaInst* create_alloca(Type* base_type, bool is_const=false, 
                              std::vector<int> dims={}, 
                              const_str_ref name="") {
        AllocaInst* inst = nullptr;
        auto entryBlock=block()->parent()->entry();
        if (dims.size() == 0) inst = new AllocaInst(base_type, entryBlock, name, is_const);
        else inst = new AllocaInst(base_type, dims, entryBlock, name, is_const);
        /* hhw, add alloca to function entry block*/
        entryBlock->emplace_back_inst(inst);
        return inst;
    }

    StoreInst* create_store(Value* value, Value* pointer) {
        auto inst = new StoreInst(value, pointer, _block);
        block()->emplace_back_inst(inst);
        return inst;
    }

    ReturnInst* create_return(Value* value = nullptr, const_str_ref name = "") {
        auto inst = new ReturnInst(value, _block);
        block()->emplace_back_inst(inst);
        return inst;
    }

    LoadInst* create_load(Value* ptr) {
        auto inst = LoadInst::gen(ptr, _block);
        block()->emplace_back_inst(inst);
        return inst;
    }

    UnaryInst* create_unary(Value::ValueId kind,
                            Type* type,
                            Value* value,
                            const_str_ref name = "") {
        auto inst = new UnaryInst(kind, type, value, _block, name);
        block()->emplace_back_inst(inst);
        return inst;
    }

    BinaryInst* create_binary(Value::ValueId kind,
                              Type* type,
                              Value* lvalue,
                              Value* rvalue,
                              const_str_ref name = "") {
        auto inst = new BinaryInst(kind, type, lvalue, rvalue, _block, name);
        block()->emplace_back_inst(inst);
        return inst;
    }

    CallInst* create_call(Function* func,
                          const_value_ptr_vector& rargs,
                          const_str_ref name = "") {
        auto call = new CallInst(func, rargs, _block, name);
        block()->emplace_back_inst(call);
        return call;
    }
    BranchInst* create_br(Value* cond,
                          BasicBlock* true_block,
                          BasicBlock* false_block) {
        auto inst = new BranchInst(cond, true_block, false_block, _block);
        block()->emplace_back_inst(inst);  // _pos++
        return inst;
    }

    BranchInst* create_br(BasicBlock* dest) {
        auto inst = new BranchInst(dest, _block);
        block()->emplace_back_inst(inst);  // _pos++
        return inst;
    }
    //! ICMP inst family
    // (itype, lhs, rhs, parent, name)
    Instruction* create_icmp(Value::ValueId itype,
                             Value* lhs,
                             Value* rhs,
                             const_str_ref name = "") {
        auto inst = new ICmpInst(itype, lhs, rhs, _block, name);
        block()->emplace_back_inst(inst);  // _pos++
        return inst;
    }

    //! FCMP inst family
    Instruction* create_fcmp(Value::ValueId itype,
                             Value* lhs,
                             Value* rhs,
                             const_str_ref name = "") {
        auto inst = new FCmpInst(itype, lhs, rhs, _block, name);
        block()->emplace_back_inst(inst);  // _pos++
        return inst;
    }

    //! Create GetElementPtr Instruction
    GetElementPtrInst* create_getelementptr(Type* base_type, Value* value, Value* idx, 
                                            std::vector<int> dims={}, std::vector<int> cur_dims={}) {
        GetElementPtrInst* inst = nullptr;
        if (dims.size() == 0 && cur_dims.size() == 0) inst = new GetElementPtrInst(base_type, value, _block, idx);
        else if (dims.size() == 0 && cur_dims.size() != 0) inst = new GetElementPtrInst(base_type, value, _block, idx, cur_dims);
        else inst = new GetElementPtrInst(base_type, value, _block, idx, dims, cur_dims);
        block()->emplace_back_inst(inst);
        return inst;
    }

    PhiInst* create_phi(Type *type, const std::vector<Value *> &vals, const std::vector<BasicBlock *> &bbs){
        auto inst = new PhiInst(_block, type, vals, bbs);
        block()->emplace_back_inst(inst);
        return inst;
    }
};

}  // namespace ir
