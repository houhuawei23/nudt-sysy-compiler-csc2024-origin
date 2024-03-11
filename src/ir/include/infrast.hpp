#pragma once

// #include <functional>
#include <map>
#include <set>
#include <unordered_map>
// #include <unordered_set>
#include <variant>

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

//  _type, _name, _uses
class Constant : public Value {
  protected:
    // std::variant<bool, int32_t, float> _field;
    // std::vector<int> hello;
    union {
        /* data */
        // bool _b;
        int _i;
        float _f;
        // double _d;
    };

  public:
    Constant();
    Constant(Type *type, const std::string &name) : Value(type, name) {}
    // Constant(bool b) : Value(Type::)
    Constant(int i) : Value(Type::int_type(), "int"), _i(i) {}
    Constant(float f) : Value(Type::float_type(), "float"), _f(f) {}

    // get
    static Constant *get(int i) {
        static std::map<int, Constant *> cache;
        auto iter = cache.find(i);
        if (iter != cache.end()) {
            return iter->second;
        }

        Constant *c = new Constant(i);
        auto res = cache.emplace(i, c);
        return c; // res.first->second; ??
    }
    static Constant *get(float f);

    int get_int() const { return _i; } // assert
    float get_float() const { return _f; }

    // operator
    void print(std::ostream &os) const override;
};
/**
 * @brief Argument represents an incoming formal argument to a Function.
 *
 */
class Argument : public Value {
  protected:
    int _index;

  public:
    Argument(Type *type, const std::string &name, size_t index)
        : Value(type, name) {
        this->_index = index;
    };
};

class BasicBlock : public Value {
  public:
  protected:
    Function *_parent;
    inst_list _instructions;
    arg_list _arguments;
    block_list _next_blocks;
    block_list _pre_blocks;

  public:
    int depth = 0;

    BasicBlock(Function *parent, const std::string &name = "")
        : Value(Type::label_type(), name){

          };

    // get
    int get_insts_num() const { return _instructions.size(); }
    int get_args_num() const { return _arguments.size(); }
    int get_next_num() const { return _next_blocks.size(); }
    int get_pre_num() const { return _pre_blocks.size(); }

    Function *get_parent() const { return _parent; }
    inst_list &get_insts() { return _instructions; }
    arg_list &get_args() { return _arguments; }
    block_list &get_next_blocks() { return _next_blocks; }
    block_list &get_pre_blocks() { return _pre_blocks; }

    // to be complete
    inst_iterator begin() { return _instructions.begin(); }
    inst_iterator end() { return _instructions.end(); }

    void print(std::ostream &os) const override;
};

class Instruction : public User {

  protected:
    BasicBlock *_parent;
    IType _itype;

  public:
    Instruction(IType itype, Type *type, BasicBlock *bb,
                const std::string &name)
        : User(type, name), _parent(bb), _itype(itype) {}
    // get
    BasicBlock *get_bb() { return _parent; };
    IType get_itype() { return _itype; };

    // set
    void set_bb(BasicBlock *bb) { _parent = bb; }
    void set_itype(IType itype) { _itype = itype; }

    // inst type check
    bool isTerminator();
    bool isUnary();
    bool isBinary();
    bool isBitwise();
    bool isMemory();
    bool isConversion();
    bool isCompare();
    bool isOther();
    bool isIcmp();
    bool isFcmp();
    bool isMath();
};

} // namespace ir