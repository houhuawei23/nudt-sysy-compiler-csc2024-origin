#include "include/value.hpp"

namespace ir {
/// Use
size_t Use::get_index() const { return _index; }

User *Use::get_user() const { return _user; }

Value *Use::get_value() const { return _value; }

void Use::set_index(size_t index) { _index = index; }
void Use::set_user(User *user) { _user = user; }
void Use::set_value(Value *value) { _value = value; }

// Value
/// std::list<std::shared_ptr<Use>> _uses;
void Value::add_use(std::shared_ptr<Use> use) { _uses.push_back(use); }
void Value::del_use(std::shared_ptr<Use> use) { _uses.remove(use); } // ?
// void Value::replace_all_use_with

/// User: public Value
std::vector<std::shared_ptr<Use>> &User::get_operands() { return _operands; }
Value *User::get_operand(size_t index) { return _operands[index]->get_value(); }

void User::add_operand(Value *value) {
    // call Use(index, user, value) to construct.
    auto nptr = std::make_shared<Use>(_operands.size(), this, value);
    _operands.emplace_back(nptr);
    value->add_use(nptr); //
}

void User::set_operand(size_t index, Value *value) {
    _operands[index]->set_value(value);
    value->add_use(_operands[index]);
}


} // namespace ir