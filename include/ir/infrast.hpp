#pragma once

#include "ir/type.hpp"
#include "ir/utils_ir.hpp"
#include "ir/value.hpp"

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

    void setname(std::string name) { _name = name; }
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
    std::vector<BasicBlock*> domTree;//sons in dom Tree
    std::vector<BasicBlock*> domFrontier;//dom frontier
    // std::set<BasicBlock*> dom;//those bb was dominated by self
    int domLevel;

   protected:
    Function* _parent;
    inst_list _insts;

    // for CFG
    block_ptr_list _next_blocks;
    block_ptr_list _pre_blocks;

    int _depth = 0;
    bool _is_terminal = false;

    std::string _comment;

   public:
    BasicBlock(const_str_ref name = "", Function* parent = nullptr)
        : Value(Type::label_type(), vBASIC_BLOCK, name),
          _parent(parent){

          };

    // get
    int depth() const { return _depth; }

    int insts_num() const { return _insts.size(); }

    int next_num() const { return _next_blocks.size(); }
    int pre_num() const { return _pre_blocks.size(); }

    bool empty() const { return _insts.empty(); }

    //* get Data Attributes
    Function* parent() const { return _parent; }

    inst_list& insts() { return _insts; }

    block_ptr_list& next_blocks() { return _next_blocks; }
    block_ptr_list& pre_blocks() { return _pre_blocks; }

    // inst iter of the block
    inst_iterator begin() { return _insts.begin(); }
    inst_iterator end() { return _insts.end(); }

    // manage
    void set_comment(const_str_ref comment) {
        if (!_comment.empty()) {
            std::cerr << "re-set basicblock comment!" << std::endl;
        }
        _comment = comment;
    }
    void append_comment(const_str_ref comment) {
        if (_comment.empty()) {
            _comment += comment;
        } else {
            _comment = _comment + ", " + comment;
        }
    }
    void set_depth(int d) { _depth = d; }  // ?

    void emplace_back_inst(Instruction* i);

    void emplace_inst(inst_iterator pos, Instruction* i);// didn't check is_terminal 

    void emplace_first_inst(Instruction* i);// didn't check is_terminal

    void delete_inst(Instruction* inst);

    // void delete_inst(Instruction* inst);

    // for CFG
    void add_next_block(BasicBlock* b) { _next_blocks.push_back(b); }
    void add_pre_block(BasicBlock* b) { _pre_blocks.push_back(b); }

    static void block_link(ir::BasicBlock* pre, ir::BasicBlock* next) {
        pre->add_next_block(next);
        next->add_pre_block(pre);
    }

    bool dominate(BasicBlock *bb){
        if(this==bb)return true;
        for(auto bbnext:domTree){
            if(bbnext->dominate(bb)) return true;
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
    Instruction(ValueId itype=vINSTRUCTION,
                Type* ret_type=Type::void_type(),
                BasicBlock* pblock=nullptr,
                const_str_ref name="")
        : User(ret_type, itype, name), _parent(pblock) {}
    // get
    BasicBlock* parent() { return _parent; };

    // set
    void set_parent(BasicBlock* parent) { _parent = parent; }

    // inst type check
    bool is_terminator() { return scid() == vRETURN || scid() == vBR; }
    bool is_unary() { return scid() == vFNEG; };
    bool is_binary() { return scid() > vBINARY_BEGIN && scid() < vBINARY_END; };
    bool is_bitwise();
    bool is_memory() {
        return scid() == vALLOCA || scid() == vLOAD || scid() == vSTORE;
    };
    bool is_conversion();
    bool is_compare();
    bool is_other();
    bool is_icmp();
    bool is_fcmp();
    bool is_math();
    bool is_noname(){return is_terminator() or scid()==vSTORE;}

    // for isa, cast and dyn_cast
    static bool classof(const Value* v) { return v->scid() >= vINSTRUCTION; }

    void setvarname();  // change varname to pass lli

    void virtual print(std::ostream& os) = 0;
};

}  // namespace ir