#pragma once

// #include <functional>
#include <cassert>
#include <map>
#include <set>
#include <unordered_map>
#include <variant>
// #include <unordered_set>
#include "function.hpp"
#include "type.hpp"
#include "value.hpp"

// #include "module.hpp"
namespace ir {

using inst_list = std::list<std::unique_ptr<Instruction>>; // list
using inst_iterator = inst_list::iterator;
using reverse_iterator = inst_list::reverse_iterator;

using arg_list = std::list<std::unique_ptr<Argument>>;     // vector -> list
using block_list = std::list<std::unique_ptr<BasicBlock>>; // vector -> list
using const_str_ref = const std::string &;
//  _type, _name, _uses
class Constant : public Value {
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
    Constant(Type *type, const_str_ref name) : Value(type, name) {}
    // Constant(bool b) : Value(Type::)
    Constant(int i) : Value(Type::int_type(), "int"), _i(i) {}
    Constant(float f) : Value(Type::float_type(), "float"), _f(f) {}

    Constant(int i, const_str_ref name)
        : Value(Type::int_type(), name), _i(i) {}
    Constant(float f, const_str_ref name)
        : Value(Type::float_type(), name), _f(f) {}
    // gen Const from int or float
    static Constant *gen(int i) {
        static std::map<int, Constant *> cache;
        auto iter = cache.find(i);
        if (iter != cache.end()) {
            return iter->second;
        }

        Constant *c = new Constant(i);
        auto res = cache.emplace(i, c);
        return c; // res.first->second; ??
    }
    static Constant *gen(float f) {
        static std::map<float, Constant *> cache;
        auto iter = cache.find(f);
        if (iter != cache.end()) {
            return iter->second;
        }

        Constant *c = new Constant(f);
        auto res = cache.emplace(f, c);
        return c; // res.first->second; ??
    }

    int i() const {
        assert(is_int());
        return _i;
    } // assert
    float f() const { return _f; }

    // operator
    void print(std::ostream &os) const override;
};
/**
 * @brief Argument represents an incoming formal argument to a Function.
 *
 */
class Argument : public Value {
  protected:
    BasicBlock *_parent;
    int _index;
    std::vector<int> _dims; // 维数信息

  public:
    Argument(Type *type, const std::string &name, size_t index,
             BasicBlock *pblock = nullptr)
        : Value(type, name), _index(index), _parent(pblock) {}

    BasicBlock *parent() const { return _parent; }
    int index() const { return _index; }
    std::vector<int> &dims() { return _dims; } // get ref?
    int dim_num() const { return _dims.size(); }
    int dim(int i) const { return _dims[i]; }
    // ir print
    void print(std::ostream &os) const override;
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
    Function *_parent;
    inst_list _instructions;
    arg_list _arguments;
    block_list _next_blocks;
    block_list _pre_blocks;
    int _depth = 0;

  public:
    BasicBlock(Function *parent = nullptr, const std::string &name = "")
        : Value(Type::label_type(), name){

          };

    // get
    int depth() const { return _depth; }
    int insts_num() const { return _instructions.size(); }
    int args_num() const { return _arguments.size(); }
    int next_num() const { return _next_blocks.size(); }
    int pre_num() const { return _pre_blocks.size(); }

    Function *parent() const { return _parent; }
    inst_list &insts() { return _instructions; }
    arg_list &args() { return _arguments; }
    block_list &next_blocks() { return _next_blocks; }
    block_list &pre_blocks() { return _pre_blocks; }

    // to be complete
    inst_iterator begin() { return _instructions.begin(); }
    inst_iterator end() { return _instructions.end(); }

    // manage
    void set_depth(int d) { _depth = d; } // ?
    // ir print
    void print(std::ostream &os) const override;
};

class Instruction : public User {

  protected:
    BasicBlock *_parent;
    IType _itype;

  public:
    /**
     * @brief Construct a new Instruction object
     *
     * @param itype
     * @param ret_type
     * @param name
     * @param pblock
     */
    Instruction(IType itype, Type *ret_type = Type::void_type(),
                const_str_ref name = "", BasicBlock *pblock = nullptr)
        : User(ret_type, name), _parent(pblock), _itype(itype) {}
    // get
    BasicBlock *parent() { return _parent; };
    IType itype() { return _itype; };

    // set
    void set_parent(BasicBlock *parent) { _parent = parent; }
    void set_itype(IType itype) { _itype = itype; }

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
};

} // namespace ir