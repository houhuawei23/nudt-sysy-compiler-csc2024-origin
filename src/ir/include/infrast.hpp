#pragma once

#include "type.hpp"
#include "utils.hpp"
#include "value.hpp"

namespace ir {
/**
 * @brief Constant Value in IR
 * Same Constant val is ther same Object, managed by the cache
 */
class Constant : public User {
    static std::map<std::string, Constant*> cache;

   protected:
    union {
        bool _i1;
        int32_t _i32;
        float _f32;
        double _f64;
    };

   public:
    Constant(bool i1, const_str_ref name)
        : User(Type::i1_type(), vCONSTANT, name), _i1(i1) {}
    Constant(int32_t i32, const_str_ref name)
        : User(Type::i32_type(), vCONSTANT, name), _i32(i32) {}
    Constant(float f32, const_str_ref name)
        : User(Type::float_type(), vCONSTANT, name), _f32(f32) {}
    Constant(double f64, const_str_ref name)
        : User(Type::double_type(), vCONSTANT, name), _f64(f64) {}
    Constant() : User(Type::void_type(), vCONSTANT, "VOID") {}

   public:
    //* add constant to cache
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

    //* generate function: gen_xx(val), gen_xx(val, name)
    template <typename T>
    static Constant* gen_i1(T v) {
        assert(std::is_integral_v<T> ||
               std::is_floating_point_v<T> && "not int or float!");

        bool num = (bool)v;
        std::string name;
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
    static Constant* gen_i32(T v, std::string name) {
        assert(std::is_integral_v<T> ||
               std::is_floating_point_v<T> && "not int or float!");

        int32_t num = (int32_t)v;
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
    template <typename T>
    static Constant* gen_f64(T val, std::string name) {
        assert(std::is_integral_v<T> ||
               std::is_floating_point_v<T> && "not int or float!");

        auto f64 = (double)val;
        return cache_add(f64, name);
    }

    static Constant* gen_void() {
        std::string name = "VOID";
        auto iter = cache.find(name);
        if (iter != cache.end()) {
            return iter->second;
        }

        Constant* c;
        c = new Constant();
        auto res = cache.emplace(name, c);
        return c;
    }

   public:
    // get function
    int32_t i32() const {
        assert(is_i32());
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
    // get f by catural type
    template <typename T>
    T f() const {
        if (is_float32()) {
            return _f32;
        } else if (is_double()) {
            return _f64;
        }
    }

   public:
    static bool classof(const Value* v) { return v->scid() == vCONSTANT; }

    void print(std::ostream& os) override;
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

   protected:
    Function* _parent;
    inst_list _insts;

    // for CFG
    block_ptr_list _next_blocks;
    block_ptr_list _pre_blocks;

    int _depth = 0;
    bool _is_terminal = false;

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
    void set_depth(int d) { _depth = d; }  // ?

    void emplace_back_inst(Instruction* i);

    void emplace_inst(inst_iterator pos, Instruction* i);

    // for CFG
    void add_next_block(BasicBlock* b) { _next_blocks.push_back(b); }
    void add_pre_block(BasicBlock* b) { _pre_blocks.push_back(b); }

    static void block_link(ir::BasicBlock* pre, ir::BasicBlock* next) {
        pre->add_next_block(next);
        next->add_pre_block(pre);
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

    // for isa, cast and dyn_cast
    static bool classof(const Value* v) {
        return v->scid() >= vINSTRUCTION;  
    }

    void setvarname();  // change varname to pass lli

    void virtual print(std::ostream& os) = 0;
};

}  // namespace ir