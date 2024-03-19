#pragma once
#include <cassert>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>
#include "type.hpp"
// #include <queue>      // for block list priority queue
// #include <algorithm>  // for block list sort
namespace ir {
class Use;
class User;
class Value;

class Constant;
class Instruction;
class BasicBlock;
class Argument;

class Function;
class Module;

//* string
// use as function formal param type for name
using const_str_ref = const std::string&;

//* Value
using value_ptr_vector = std::vector<Value*>;

// use as function formal param for dims or indices
using const_value_ptr_vector = const std::vector<Value*>;

// symbol table, look up value based on name
using str_value_map = std::map<std::string, Value*>;

//* Use
// Value _uses
using use_ptr_list = std::list<Use*>;
using use_ptr_vector = std::vector<Use*>;

//* BasicBlock
using block_ptr_list = std::list<BasicBlock*>;
using block_ptr_vector = std::vector<BasicBlock*>;
// true or false targets stack
using block_ptr_stack = std::stack<BasicBlock*>;

//* Argument
// function args
using arg_ptr_list = std::list<Argument*>;
using arg_ptr_vector = std::vector<Argument*>;

//* Instruction
// basicblock insts
using inst_list = std::list<Instruction*>;
// iterator for add/del/traverse inst list
using inst_iterator = inst_list::iterator;
using reverse_iterator = inst_list::reverse_iterator;

//* Function
// look up function in function table
using str_fun_map = std::map<std::string, Function*>;

// type, name
/**
 * @brief Base Class for all classes having 'value' to be used.?
 * @attention
 * - Value 是除了 Type， Module 之外几乎所有数据结构的基类。
 * - Value 表示一个“值”，它有名字 _name，有类型 _type，可以被使用 _uses。
 * - 派生类继承 Value，添加自己所需的 数据成员 和 方法。
 * - Value 的派生类可以重载 print() 方法，以打印出可读的 IR。
 */

class Value {
   public:
    enum ValueId {
        vValue,
        vFUNCTION,
        vCONSTANT,
        vARGUMENT,
        vBASIC_BLOCK,
        vGLOBAL_VAR,
        // instructions class id
        vINSTRUCTION,
        vALLOCA,
        vLOAD,
        vSTORE,
        vRETURN,
        vBR,
        vCALL,
        // icmp
        vICMP,
        vIEQ,
        vINE,
        vISGT,
        vISGE,
        vISLT,
        vISLE,
        vICMP_END,
        // fcmp
        vFCMP,
        vFOEQ,
        vFONE,
        vFOGT,
        vFOGE,
        vFOLT,
        vFOLE,
        vFCMP_END,
        // Unary Instruction
        vFNEG,

        vFTOI,
        vITOF,

        // Binary Instruction
        vADD,
        vFADD,
        vSUB,
        vFSUB,

        vMUL,
        vFMUL,

        vUDIV,
        vSDIV,
        vFDIV,

        vUREM,
        vSREM,
        vFREM,
    };
    // for isa<> type check
    // each subclass of Value has different _scid
    // compare _scid to check class
    // isa<Constant>: return _scid == CONSTANT

    // _type, _name, _uses
    // friend class User;
   protected:
    Type* _type;    // type of the value
    ValueId _scid;  // subclass id of Value
    std::string _name;
    use_ptr_list _uses;  // uses list

   public:
    Value(Type* type, ValueId scid = vValue, const_str_ref name = "")
        : _type(type), _scid(scid), _name(name), _uses() {}
    virtual ~Value() = default;
    // Value is all base, return true
    static bool classof(const Value* v) { return true; }

    //* get
    Type* type() const { return _type; }
    std::string name() const { return _name; }
    use_ptr_list& uses() { return _uses; }

    //* manage
    void add_use(Use* use);
    void del_use(Use* use);
    void replace_all_use_with(Value* _value);

    void set_name(const_str_ref name) { _name = name; }

    //* check
    bool is_i1() const { return _type->is_i1(); }
    bool is_i32() const { return _type->is_i32(); }
    bool is_float32() const { return _type->is_float32(); }
    bool is_double() const { return _type->is_double(); }
    bool is_float() const { return _type->is_float(); }

   public:
    ValueId scid() const { return _scid; }
    // each derived class must implement 'print' to print readable IR
    virtual void print(std::ostream& os) const {};
};

/**
 * @brief 表征操作数本身的信息, 连接 value 和 user
 * index in the _operands, _user, _value
 *
 */
class Use {
    // friend class Value;
    // friend class
   protected:
    size_t _index;
    User* _user;
    Value* _value;

   public:
    Use() = default;
    Use(size_t index, User* user, Value* value)
        : _index(index), _user(user), _value(value){};

    // get
    size_t index() const;
    User* user() const;
    Value* value() const;
    // set
    void set_index(size_t index);
    void set_value(Value* value);
    void set_user(User* user);
};

/**
 * @brief 使用“值”，既要使用值，又有返回值，所以继承自 Value
 * @attention _operands
 * 派生类： Instruction
 *
 * User is the abstract base type of `Value` types which use other `Value` as
 * operands. Currently, there are two kinds of `User`s, `Instruction` and
 * `GlobalValue`.
 *
 */
class User : public Value {
    // _type, _name, _uses
   protected:
    use_ptr_vector _operands;  // 操作数
   public:
    User(Type* type, ValueId scid, const_str_ref name = "")
        : Value(type, scid, name) {}
    // get
    use_ptr_vector& operands();
    Value* operand(size_t index) const;
    int operands_cnt() const { return _operands.size(); }
    // manage
    void add_operand(Value* value);
    void set_operand(size_t index, Value* value);

    template <typename Container>
    void add_operands(const Container& operands) {
        for (auto value : operands) {
            add_operand(value);
        }
    }

    void unuse_allvalue() {
        int a = 5;
    };
    void replace_operand_with(size_t index, Value* value){
        int a = 5;
    };
    // void print(std::ostream& os) const {};
};
}  // namespace ir
