#pragma once

// #include <functional>
#include <cassert>
#include <map>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <variant>
// #include <unordered_set>
// #include "function.hpp"
#include "type.hpp"
#include "value.hpp"

// #include "module.hpp"
namespace ir {

using inst_list = std::list<Instruction*>;  // list
using inst_iterator = inst_list::iterator;
using reverse_iterator = inst_list::reverse_iterator;

using arg_list = std::list<Argument*>;      // vector -> list
using block_list = std::list<BasicBlock*>;  // vector -> list
using const_str_ref = const std::string&;

//  _type, _name, _uses
class Constant : public User {
   protected:
    // std::variant<bool, int32_t, float> _field;
    union {
        /* data */
        // bool _b;
        int _i;
        float _f;
        // double _d;
    };

   public:
    Constant();
    // Constant(Type* type, const_str_ref name = "")
    //     : User(type, vCONSTANT, name) {}
    Constant(Type* type, ValueId scid = vCONSTANT, const_str_ref name = "")
        : User(type, scid, name) {}

    Constant(int i)
        : User(Type::int_type(), vCONSTANT, std::to_string(i)), _i(i) {}

    //! TODO: float to string? is that ok?
    Constant(float f)
        : User(Type::float_type(), vCONSTANT, std::to_string(f)), _f(f) {}

    Constant(double f)
        : User(Type::float_type(), vCONSTANT, std::to_string(f)), _f(f) {}

    Constant(int i, const_str_ref name)
        : User(Type::int_type(), vCONSTANT, name), _i(i) {}

    Constant(float f, const_str_ref name)
        : User(Type::float_type(), vCONSTANT, name), _f(f) {}

    Constant(double f, const_str_ref name)
        : User(Type::float_type(), vCONSTANT, std::to_string(f)), _f(f) {}

    // gen Const from int or float

    template <typename T>
    static Constant* gen(T v, const_str_ref name = "") {
        static std::map<T, Constant*> cache;
        assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
        auto iter = cache.find(v);
        if (iter != cache.end()) {
            return iter->second;
        }
        Constant* c;
        if (name == "")
            c = new Constant(v);
        else
            c = new Constant(v, name);
        auto res = cache.emplace(v, c);
        return c;  // res.first->second; ??
    }

    int i() const {
        assert(is_int());  // assert
        return _i;
    }
    float f() const {
        assert(is_float());
        return _f;
    }

    // isa
    static bool classof(const Value* v) { return v->scid() == vCONSTANT; }

    // ir print
    void print(std::ostream& os) const override;
};

class ConstantBeta : public User {
   protected:
    // std::variant<bool, int32_t, float> _field;
    union {
        /* data */
        // bool _b;
        int _i;
        float _f;
        // double _d;
    };

   public:
    ConstantBeta();
    ConstantBeta(Type* type, const_str_ref name = "")
        : User(type, vCONSTANT, name) {}
    ConstantBeta(Type* type, ValueId scid = vCONSTANT, const_str_ref name = "")
        : User(type, scid, name) {}

    ConstantBeta(int i)
        : User(Type::int_type(), vCONSTANT, std::to_string(i)), _i(i) {}
    //! TODO: float to string? is that ok?
    ConstantBeta(float f)
        : User(Type::float_type(), vCONSTANT, std::to_string(f)), _f(f) {}

    ConstantBeta(int i, const_str_ref name)
        : User(Type::int_type(), vCONSTANT, name), _i(i) {}
    ConstantBeta(float f, const_str_ref name)
        : User(Type::float_type(), vCONSTANT, name), _f(f) {}

    // gen Const from int or float

    template <typename T>
    static ConstantBeta* gen(T v) {
        static std::map<T, ConstantBeta*> cache;
        assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
        auto iter = cache.find(v);
        if (iter != cache.end()) {
            return iter->second;
        }
        ConstantBeta* c = new ConstantBeta(v);
        auto res = cache.emplace(v, c);
        return c;  // res.first->second; ??
    }

    int i() const {
        assert(is_int());  // assert
        return _i;
    }
    float f() const {
        assert(is_float());
        return _f;
    }

    // isa
    static bool classof(const Value* v) { return v->scid() == vCONSTANT; }

    // ir print
    void print(std::ostream& os) const override;
};
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
             const std::string& name = "")
        : Value(type, vARGUMENT, name), _index(index), _parent(parent) {}

    BasicBlock* parent() const { return _parent; }
    int index() const { return _index; }
    std::vector<int>& dims() { return _dims; }  // get ref?
    int dim_num() const { return _dims.size(); }
    int dim(int i) const { return _dims[i]; }

    // for isa<>
    static bool classof(const Value* v) { return v->scid() == vARGUMENT; }

    // ir print
    void print(std::ostream& os) const override;
};
/**
 * @brief The container for `Instruction` sequence.
 * `BasicBlock` maintains a list of `Instruction`s, with the last one being a
 * terminator (branch or return). Besides, `BasicBlock` stores its arguments and
 * records its predecessor and successor `BasicBlock`s.
 */
class BasicBlock : public Value {
    // _type: label_type()

   protected:
    Function* _parent;
    inst_list _insts;
    arg_list _args;  // ?
    block_list _next_blocks;
    block_list _pre_blocks;
    int _depth = 0;

   public:
    BasicBlock(const std::string& name = "", Function* parent = nullptr)
        : Value(Type::label_type(), vBASIC_BLOCK, name),
          _parent(parent){

          };

    // get
    int depth() const { return _depth; }
    int insts_num() const { return _insts.size(); }
    int args_num() const { return _args.size(); }
    int next_num() const { return _next_blocks.size(); }
    int pre_num() const { return _pre_blocks.size(); }

    bool empty() const { return _insts.empty(); }

    Function* parent() const { return _parent; }
    inst_list& insts() { return _insts; }
    arg_list& args() { return _args; }
    block_list& next_blocks() { return _next_blocks; }
    block_list& pre_blocks() { return _pre_blocks; }

    // inst iter of the block
    inst_iterator begin() { return _insts.begin(); }
    inst_iterator end() { return _insts.end(); }

    // manage
    void set_depth(int d) { _depth = d; }  // ?

    void emplace_back_inst(Instruction* i) { _insts.emplace_back(i); }
    void emplace_inst(inst_iterator pos, Instruction* i) {
        _insts.emplace(pos, i);
    }

    // for CFG ?
    void add_next_block(BasicBlock* b) { _next_blocks.push_back(b); }
    void add_pre_block(BasicBlock* b) { _pre_blocks.push_back(b); }

    // for isa<>
    static bool classof(const Value* v) { return v->scid() == vBASIC_BLOCK; }

    // ir print
    void print(std::ostream& os) const override;

    static void block_link(ir::BasicBlock* pre, ir::BasicBlock* next) {
        pre->add_next_block(next);
        next->add_pre_block(pre);
    }

    //! for pq
    // bool operator<(const BasicBlock& other) const {
    //     return std::stoi(name().substr(1)) >
    //     std::stoi(other.name().substr(1));
    // }
    // bool operator<(const BasicBlock* other) const {
    //     return std::stoi(name().substr(1)) >
    //     std::stoi(other->name().substr(1));
    // }
    friend bool operator<(BasicBlock& a, BasicBlock& other) {
        return std::stoi(a.name().substr(1)) >
               std::stoi(other.name().substr(1));
    }
};

inline bool compareBB(const BasicBlock* a1, const BasicBlock* a2) {
    // return a1->priority < a2->priority;
    if (a1->name().size() > 0 && a2->name().size() > 0)
        return std::stoi(a1->name().substr(1)) <
               std::stoi(a2->name().substr(1));
    else {
        // std::
        assert(false && "compareBB error");
    }
}

class Instruction : public User {
    // Instuction 的类型也通过 _scid
   protected:
    BasicBlock* _parent;

   public:
    /**
     * @brief Construct a new Instruction object
     *
     */
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
    bool is_terminator();
    bool is_unary();
    bool is_binary();
    bool is_bitwise();
    bool is_memory();
    bool is_conversion();
    bool is_compare();
    bool is_other();
    bool is_icmp();
    bool is_fcmp();
    bool is_math();

    // for isa, cast and dyn_cast
    static bool classof(const Value* v) {
        return v->scid() >= vINSTRUCTION;  // <= ?
    }
};

}  // namespace ir