#pragma once

#include <list>
#include <string>
// #include <cassert>
#include <memory>

#include "type.hpp"

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

using const_str_ref = const std::string &;
using use_ptr_list = std::list<std::shared_ptr<Use>>;
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
    // _type, _name, _uses
    // friend class User;
  protected:
    Type *_type;
    std::string _name;
    use_ptr_list _uses;

  public:
    Value(Type *type, const_str_ref name = "")
        : _type(type), _name(name), _uses() {}
    virtual ~Value() = default;
    // get
    Type *type() const { return _type; }
    std::string name() const { return _name; }
    use_ptr_list &uses() { return _uses; }

    // manage
    void add_use(std::shared_ptr<Use> use);
    void del_use(std::shared_ptr<Use> use);
    void replace_all_use_with(Value *_value);

    // check
    bool is_int() const { return _type->is_int(); }

  public:
    // each derived class must implement 'print' to print readable IR
    virtual void print(std::ostream &os) const {};
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
    User *_user;
    Value *_value;

  public:
    Use() = default;
    Use(size_t index, User *user, Value *value)
        : _index(index), _user(user), _value(value){};

    // get
    size_t index() const;
    User *user() const;
    Value *value() const;
    // set
    void set_index(size_t index);
    void set_value(Value *value);
    void set_user(User *user);
};
using use_ptr = std::shared_ptr<Use>;
using use_ptr_vector = std::vector<use_ptr>;
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
    use_ptr_vector _operands; // 操作数
  public:
    User(Type *type, const_str_ref name) : Value(type, name) {}
    // get
    use_ptr_vector &operands();
    Value *operand(size_t index) const;

    // manage
    void add_operand(Value *value);
    void set_operand(size_t index, Value *value);

    template <typename Container> void add_operands(const Container &operands) {
        for (auto value : operands) {
            add_operand(value);
        }
    }

    void unuse_allvalue();
    void replace_operand_with(size_t index, Value *value);
};
} // namespace ir
