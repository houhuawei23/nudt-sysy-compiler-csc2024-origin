#pragma once

// #include <functional>
#include <cassert>
#include <map>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <variant>
// #include <unordered_set>
#include "function.hpp"
#include "type.hpp"
#include "value.hpp"

// #include "module.hpp"
namespace ir {

using inst_list = std::list<std::unique_ptr<Instruction>>;  // list
using inst_iterator = inst_list::iterator;
using reverse_iterator = inst_list::reverse_iterator;

using arg_list = std::list<std::unique_ptr<Argument>>;      // vector -> list
using block_list = std::list<std::unique_ptr<BasicBlock>>;  // vector -> list
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
    Constant(Type* type, const_str_ref name = "")
        : User(type, vCONSTANT, name) {}
    Constant(Type* type, ValueId scid = vCONSTANT, const_str_ref name = "")
        : User(type, scid, name) {}

    Constant(int i)
        : User(Type::int_type(), vCONSTANT, std::to_string(i)), _i(i) {}
    //! TODO: float to string? is that ok?
    Constant(float f)
        : User(Type::float_type(), vCONSTANT, std::to_string(f)), _f(f) {}

    Constant(int i, const_str_ref name)
        : User(Type::int_type(), vCONSTANT, name), _i(i) {}
    Constant(float f, const_str_ref name)
        : User(Type::float_type(), vCONSTANT, name), _f(f) {}

    // gen Const from int or float

    template <typename T>
    static Constant* gen(T v) {
        static std::map<T, Constant*> cache;
        assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
        auto iter = cache.find(v);
        if (iter != cache.end()) {
            return iter->second;
        }
        Constant* c = new Constant(v);
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
 *
 */
class Argument : public Value {
   protected:
    BasicBlock* _parent;
    int _index;
    std::vector<int> _dims;  // 维数信息

   public:
    Argument(Type* type,
             const std::string& name,
             size_t index,
             BasicBlock* pblock = nullptr)
        : Value(type, vARGUMENT, name), _index(index), _parent(pblock) {}

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
    inst_list _instructions;
    arg_list _arguments;
    block_list _next_blocks;
    block_list _pre_blocks;
    int _depth = 0;

   public:
    BasicBlock(const std::string& name = "", Function* parent = nullptr)
        : Value(Type::label_type(), vBASIC_BLOCK, name),
          _parent(parent) {
            
          };

    // get
    int depth() const { return _depth; }
    int insts_num() const { return _instructions.size(); }
    int args_num() const { return _arguments.size(); }
    int next_num() const { return _next_blocks.size(); }
    int pre_num() const { return _pre_blocks.size(); }

    Function* parent() const { return _parent; }
    inst_list& insts() { return _instructions; }
    arg_list& args() { return _arguments; }
    block_list& next_blocks() { return _next_blocks; }
    block_list& pre_blocks() { return _pre_blocks; }

    // to be complete
    inst_iterator begin() { return _instructions.begin(); }
    inst_iterator end() { return _instructions.end(); }

    // manage
    void set_depth(int d) { _depth = d; }  // ?

    // for isa<>
    static bool classof(const Value* v) { return v->scid() == vBASIC_BLOCK; }

    // ir print
    void print(std::ostream& os) const override;
};
// Instuction 的类型也通过 _scid
class Instruction : public User {
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