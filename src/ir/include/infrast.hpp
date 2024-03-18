#pragma once

#include "type.hpp"
#include "utils.hpp"
#include "value.hpp"

// #include "module.hpp"
namespace ir {

//  _type, _name, _uses
class Constant : public User {
    static std::map<std::string, Constant*> cache;

   protected:
    // std::variant<bool, int32_t, float> _field;
    union {
        /* data */
        bool _i1;
        int32_t _i32;  // signed int
        float _f32;
        double _f64;
    };

   public:
    //* Constant(num, name)
    Constant(bool i1, const_str_ref name)
        : User(Type::i1_type(), vCONSTANT, name), _i1(i1) {
        // int a = 2;
    }

    Constant(int32_t i32, const_str_ref name)
        : User(Type::i32_type(), vCONSTANT, name), _i32(i32) {
        // int a = 2;
    }

    Constant(float f32, const_str_ref name)
        : User(Type::float_type(), vCONSTANT, name), _f32(f32) {}

    Constant(double f64, const_str_ref name)
        : User(Type::double_type(), vCONSTANT, name), _f64(f64) {
        // int a = 2;
    }

    // gen Const from int or float
    template <typename T>
    static Constant* cache_add(T val, const std::string& name) {
        auto iter = cache.find(name);
        if (iter != cache.end()) {
            return iter->second;
        }

        Constant* c;
        c = new Constant(val, name);
        auto res = cache.emplace(name, c);
        return c;
    }

    template <typename T>
    static Constant* gen_i1(T v) {
        assert(std::is_integral_v<T> ||
               std::is_floating_point_v<T> && "not int or float!");

        bool num = (bool)v;
        std::string name;
        // auto name = std::to_string(num);
        if (num) {
            name = "true";
        } else {
            name = "false";
        }
        return cache_add(num, name);
    }

    template <typename T>
    static Constant* gen_i32(T v) {
        assert(std::is_integral_v<T> ||
               std::is_floating_point_v<T> && "not int or float!");

        int32_t num = (int32_t)v;

        std::string name = std::to_string(num);
        return cache_add(num, name);
    }

    template <typename T>
    static Constant* gen_f64(T val) {
        assert(std::is_integral_v<T> ||
               std::is_floating_point_v<T> && "not int or float!");

        auto f64 = (double)val;

        auto name = getMC(f64);
        return cache_add(f64, name);
    }

    // static Constant* gen_i32(int32_t v) {
    //     auto name = std::to_string(v);
    //     return cache_add(name);
    // }
    // static Constant* gen_f32
    // float and double both gen f64
    // static Constant* gen_f64(float v) {
    //     auto name = getMC(v);
    //     return cache_add(name);
    // }
    // static Constant* gen_f64(double v) {
    //     auto name = getMC(v);
    //     return cache_add(name);
    // }

    int32_t i32() const {
        assert(is_i32());  // assert
        return _i32;
    }

    float f32() const {
        assert(is_float32());
        return _f32;
    }

    double f64() const {
        assert(is_float());
        return _f64;
    }

    template <typename T>
    T f() const {
        if (is_float32()) {
            return _f32;
        }
        else if (is_double()) {
            return _f64;
        } 
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
    // arg_ptr_list _args;  // ?
    block_ptr_list _next_blocks;
    block_ptr_list _pre_blocks;
    int _depth = 0;

   public:
    BasicBlock(const_str_ref name = "", Function* parent = nullptr)
        : Value(Type::label_type(), vBASIC_BLOCK, name),
          _parent(parent){

          };

    // get
    int depth() const { return _depth; }
    int insts_num() const { return _insts.size(); }
    // int args_num() const { return _args.size(); }
    int next_num() const { return _next_blocks.size(); }
    int pre_num() const { return _pre_blocks.size(); }

    bool empty() const { return _insts.empty(); }

    Function* parent() const { return _parent; }
    inst_list& insts() { return _insts; }
    // arg_ptr_list& args() { return _args; }
    block_ptr_list& next_blocks() { return _next_blocks; }
    block_ptr_list& pre_blocks() { return _pre_blocks; }

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