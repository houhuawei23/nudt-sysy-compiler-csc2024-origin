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
    size_t get_index() const;
    User *get_user() const;
    Value *get_value() const;
    // set
    void set_index(size_t index);
    void set_value(Value *value);
    void set_user(User *user);
};
/**
 * @brief Base Class for all classes having 'value' to be used.?
 *
 *
 *
 */
class Value {
    // _type, _name, _uses
    // friend class User;
  protected:
    Type *_type;
    std::string _name;
    std::list<std::shared_ptr<Use>> _uses;

  public:
    Value(Type *type, const std::string &name)
        : _type(type), _name(name), _uses() {}
    virtual ~Value() = default;
    // get
    Type *get_type() const { return _type; }
    std::string get_name() const { return _name; }
    std::list<std::shared_ptr<Use>> &get_uses() { return _uses; }

    // manage
    void add_use(std::shared_ptr<Use> use);
    void del_use(std::shared_ptr<Use> use);
    void replace_all_use_with(Value *_value);
};

class User : public Value {
    // _type, _name, _uses
  protected:
    std::vector<std::shared_ptr<Use>> _operands; // 操作数
  public:
    User(Type *type, const std::string &name) : Value(type, name) {}
    // get
    std::vector<std::shared_ptr<Use>> &get_operands();
    Value *get_operand(size_t index);

    void add_operand(Value *value);
    void set_operand(size_t index, Value *value);

    template <typename Container> void add_operands(const Container &operands);

    void unuse_allvalue();
    void replace_operand_with(size_t index, Value *value);
};
} // namespace ir
