#pragma once

#include "ir/type.hpp"
#include "ir/utils_ir.hpp"
#include "ir/value.hpp"
#include <algorithm>
namespace ir {
/**
 * @brief Argument represents an incoming formal argument to a Function.
 * 形式参数，因为它是“形式的”，所以不包含实际值，而是表示特定函数的参数的类型、参数编号和属性。
 * 当在所述函数体中使用时，参数当然代表调用该函数的实际参数的值。
 */
class Argument : public Value {
   protected:
    Function* _parent;
    int _index;
    std::vector<int> _dims;  // 维数信息

   public:
    Argument(Type* type,
             size_t index,
             Function* parent = nullptr,
             const_str_ref name = "")
        : Value(type, vARGUMENT, name), _index(index), _parent(parent) {}

    Function* parent() const { return _parent; }

    int index() const { return _index; }

    std::vector<int>& dims() { return _dims; }  // get ref?

    int dim_num() const { return _dims.size(); }
    int dim(int i) const { return _dims[i]; }

    // for isa<>
    static bool classof(const Value* v) { return v->scid() == vARGUMENT; }

    // ir print
    void print(std::ostream& os) override;
};

/**
 * @brief The container for `Instruction` sequence.
 * `BasicBlock` maintains a list of `Instruction`s, with the last one being a
 * terminator (branch or return). Besides, `BasicBlock` stores its arguments and
 * records its predecessor and successor `BasicBlock`s.
 */
class BasicBlock : public Value {
    // _type: label_type()

    // dom info for anlaysis
   public:
    BasicBlock* idom;
    BasicBlock* sdom;
    BasicBlock* ipdom;
    BasicBlock* spdom;
    std::vector<BasicBlock*> domTree;      // sons in dom Tree
    std::vector<BasicBlock*> domFrontier;  // dom frontier
    std::vector<BasicBlock*> pdomTree;      
    std::vector<BasicBlock*> pdomFrontier;  
    // std::set<BasicBlock*> dom;//those bb was dominated by self
    int domLevel;
    int pdomLevel;
    int looplevel;

   protected:
    Function* _parent;
    inst_list _insts;

    // for CFG
    block_ptr_list _next_blocks;
    block_ptr_list _pre_blocks;
    // specially for Phi
    inst_list _phi_insts;

    int _depth = 0;
    bool _is_terminal = false;

    uint32_t _idx = 0;

   public:
    BasicBlock(const_str_ref name = "", Function* parent = nullptr)
        : Value(Type::label_type(), vBASIC_BLOCK, name),
          _parent(parent){

          };
    uint32_t idx() const { return _idx; }
    void set_idx(uint32_t idx) { _idx = idx; }
    /* must override */
    std::string name() const override { return "bb" + std::to_string(_idx); }
    // get
    int depth() const { return _depth; }

    bool empty() const { return _insts.empty(); }

    //* get Data Attributes
    Function* parent() const { return _parent; }

    inst_list& insts() { return _insts; }

   inst_list& phi_insts(){ return _phi_insts; }

    block_ptr_list& next_blocks() { return _next_blocks; }
    block_ptr_list& pre_blocks() { return _pre_blocks; }

    void set_depth(int d) { _depth = d; }  // ?

    void emplace_back_inst(Instruction* i);

    void emplace_inst(inst_iterator pos,
                      Instruction* i);  // didn't check is_terminal

    void emplace_first_inst(Instruction* i);  // didn't check is_terminal

    void delete_inst(Instruction* inst);

    void force_delete_inst(Instruction* inst);

    void replaceinst(Instruction* old, Value* new_);

    // for CFG
    static void block_link(ir::BasicBlock* pre, ir::BasicBlock* next) {
        pre->next_blocks().emplace_back(next);
        next->pre_blocks().emplace_back(pre);
    }

    static void delete_block_link(ir::BasicBlock* pre, ir::BasicBlock* next) {
        pre->next_blocks().remove(next);
        next->pre_blocks().remove(pre);
    }

    bool dominate(BasicBlock* bb) {
        if (this == bb)
            return true;
        for (auto bbnext : domTree) {
            if (bbnext->dominate(bb))
                return true;
        }
        return false;
    }

    // for isa<>
    static bool classof(const Value* v) { return v->scid() == vBASIC_BLOCK; }
    // ir print
    void print(std::ostream& os) override;
};

/**
 * @brief Base class for all instructions in IR
 *
 */
class Instruction : public User {
    // Instuction 的类型也通过 _scid
   protected:
    BasicBlock* _parent;

   public:
    // Construct a new Instruction object
    Instruction(ValueId itype = vINSTRUCTION,
                Type* ret_type = Type::void_type(),
                BasicBlock* pblock = nullptr,
                const_str_ref name = "")
        : User(ret_type, itype, name), _parent(pblock) {}
    // get
    BasicBlock* parent() { return _parent; };

    // set
    void set_parent(BasicBlock* parent) { _parent = parent; }

    // inst type check
    bool is_terminator() { return _scid == vRETURN || _scid == vBR; }
    bool is_unary() { return _scid > vUNARY_BEGIN && _scid < vUNARY_END; };
    bool is_binary() { return _scid > vBINARY_BEGIN && _scid < vBINARY_END; };
    bool is_bitwise() { return false; }
    bool is_memory() {
        return _scid == vALLOCA || _scid == vLOAD || _scid == vSTORE ||
               _scid == vGETELEMENTPTR;
    };
    bool is_conversion();
    bool is_compare();
    bool is_other();
    bool is_icmp();
    bool is_fcmp();
    bool is_math();
    bool is_noname() { return is_terminator() or _scid == vSTORE or _scid == vMEMSET; }

    // for isa, cast and dyn_cast
    static bool classof(const Value* v) { return v->scid() >= vINSTRUCTION; }

    void setvarname();  // change varname to pass lli

    void virtual print(std::ostream& os) = 0;

    virtual Value* getConstantRepl()=0;
};

}  // namespace ir