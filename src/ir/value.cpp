#include "ir/value.hpp"

namespace ir {
//! Use
size_t Use::index() const {
    return _index;
}

User* Use::user() const {
    return _user;
}

Value* Use::value() const {
    return _value;
}

void Use::set_index(size_t index) {
    _index = index;
}
void Use::set_user(User* user) {
    _user = user;
}
void Use::set_value(Value* value) {
    _value = value;
}

//! Value

void Value::add_use(Use* use) {
    _uses.emplace_back(use);
}
void Value::del_use(Use* use) {
    _uses.remove(use);
}

void Value::replace_all_use_with(Value* _value) {
    for (auto puse : _uses) {
        puse->user()->set_operand(puse->index(), _value);
    }
    _uses.clear();
}

//! User: public Value
// TODO: unpack use to value!
use_ptr_vector& User::operands() {
    return _operands;
}

/* return as value */
Value* User::operand(size_t index) const {
    assert(index<_operands.size());
    return _operands[index]->value();
}

void User::add_operand(Value* value) {
    assert(value != nullptr && "value cannot be nullptr");

    auto new_use = new Use(_operands.size(), this, value);

    /* add use to user._operands*/
    _operands.emplace_back(new_use);
    /* add use to value._uses */
    value->add_use(new_use);
}

void User::set_operand(size_t index, Value* value) {
    assert(index<_operands.size());
    _operands[index]->set_value(value);
    value->add_use(_operands[index]);
}

}  // namespace ir