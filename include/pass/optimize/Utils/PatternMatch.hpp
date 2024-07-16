#pragma once
#include "ir/ir.hpp"
#include <cstdint>
#include <type_traits>
#include <cassert>
using namespace ir;
namespace pass {

template <typename ValueType>
struct MatchContext final {
  ValueType* value;

  explicit MatchContext(ValueType* val) : value{val} {};

  template <typename T = ValueType>
  MatchContext<Value> getOperand(uint32_t idx) const {
    return MatchContext<Value>{value->operand(idx)};
  }
};

template <typename T, typename Derived>
class GenericMatcher {
 public:
  bool operator()(const MatchContext<Value>& ctx) const noexcept {
    if (auto val = dynamic_cast<T*>(ctx.value)) {
      return (static_cast<const Derived*>(this))->handle(MatchContext<T>{val});
    }
    return false;
  }
};

class AnyMatcher {
  Value*& mValue;

 public:
  explicit AnyMatcher(Value*& value) noexcept : mValue{value} {}
  bool operator()(const MatchContext<Value>& ctx) const noexcept {
    mValue = ctx.value;
    return true;
  }
};

inline auto any(Value*& val) {
  return AnyMatcher{val};
}

class BooleanMatcher {
  Value*& mValue;

 public:
  explicit BooleanMatcher(Value*& value) noexcept : mValue{value} {}
  bool operator()(const MatchContext<Value>& ctx) const noexcept {
    mValue = ctx.value;
    return mValue->type()->isBool();
  }
};

inline auto boolean(Value*& val) {
  return BooleanMatcher{val};
}

class ConstantIntMatcher {
  int64_t& mVal;

 public:
  ConstantIntMatcher(int64_t& value) : mVal{value} {}
  bool operator()(const MatchContext<Value>& ctx) const {
    if (auto value = ctx.value->dynCast<Constant>()) {
      if (value->type()->isInt32()) {
        mVal = value->i32();
        return true;
      }
    }
    return false;
  }
};

class ConstatIntValMatcher {
  int64_t mVal;

 public:
  ConstatIntValMatcher(int64_t value) : mVal{value} {}
  bool operator()(const MatchContext<Value>& ctx) const {
    if (auto value = ctx.value->dynCast<Constant>()) {
      if (value->type()->isInt32()) {
        return mVal == value->i32();
      }
    }
    return false;
  }
};

inline auto int_(int64_t& val) noexcept {
  return ConstantIntMatcher{val};
}

inline auto intval_(int64_t val) {
  return ConstatIntValMatcher{val};
}

template <bool IsCommutative, typename LhsMatcher, typename RhsMatcher>
class BinaryInstMatcher final
    : public GenericMatcher<
          BinaryInst,
          BinaryInstMatcher<IsCommutative, LhsMatcher, RhsMatcher>> {
  //
  ValueId mValueId;
  LhsMatcher mLhsMatcher;
  RhsMatcher mRhsMatcher;

 public:
  BinaryInstMatcher(ValueId valueId,
                    LhsMatcher lhsMatcher,
                    RhsMatcher rhsMatcher)
      : mValueId{valueId}, mLhsMatcher{lhsMatcher}, mRhsMatcher{rhsMatcher} {}

  bool handle(const MatchContext<BinaryInst>& ctx) const {
    if (mValueId != ValueId::vInvalid and mValueId != ctx.value->valueId()) {
      return false;
    }
    if (mLhsMatcher(ctx.getOperand(0)) and mRhsMatcher(ctx.getOperand(1))) {
      return true;
    }
    if constexpr (IsCommutative) {
      return mLhsMatcher(ctx.getOperand(1)) and mRhsMatcher(ctx.getOperand(0));
    }
    return false;
  }
};
template <typename LhsMatcher, typename RhsMatcher>
auto add(LhsMatcher lhsMatcher, RhsMatcher rhsMatcher) {
  return BinaryInstMatcher<true, LhsMatcher, RhsMatcher>{
      ValueId::vADD, lhsMatcher, rhsMatcher};
}

}  // namespace pass
