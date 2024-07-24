#pragma once
#include "mir/mir.hpp"
namespace mir {
/*
 * @brief: TargetRegisterInfo Class (抽象基类)
 */
class TargetRegisterInfo {
public:
  virtual ~TargetRegisterInfo() = default;

public:  // get function
  virtual uint32_t get_alloca_class_cnt() = 0;
  virtual uint32_t getAllocationClass(OperandType type) = 0;
  virtual std::vector<uint32_t>& get_allocation_list(uint32_t classId) = 0;

  /** 获得合法化后的寄存器类型 */
  virtual OperandType getCanonicalizedRegisterType(OperandType type) = 0;
  virtual OperandType getCanonicalizedRegisterTypeForClass(
    uint32_t classId) = 0;
  virtual OperandType getCanonicalizedRegisterType(uint32_t reg) = 0;
  virtual MIROperand get_return_address_register() = 0;
  virtual MIROperand get_stack_pointer_register() = 0;

public:  // check function
  virtual bool is_legal_isa_reg_operand(MIROperand& op) = 0;
  virtual bool is_zero_reg(const uint32_t x) const = 0;
};

/*
 * @brief: RegisterSelector Class
 * @note:
 *      Register Selector (分为General-Purpose Register and Floating Point
 * Register)
 */
constexpr uint32_t maxRegisterCount = 60;
class RegisterSelector {
  int64_t mFree;  // 使用位集合的形式跟踪所有可用的寄存器: 如果位为1,
                  // 表示对应的寄存器是空闲的
  std::array<uint32_t, maxRegisterCount>
    mIdx2Reg;                      // 将数组索引映射到寄存器编号
  std::vector<uint32_t> mReg2Idx;  // 将寄存器编号映射到数组索引

public:
  RegisterSelector(const std::vector<uint32_t>& list) {
    assert(list.size() <= maxRegisterCount);
    mFree = (static_cast<int64_t>(1) << list.size()) - 1;
    for (uint32_t idx = 0; idx < list.size(); ++idx) {
      const auto reg = list[idx];
      mIdx2Reg[idx] = reg;
      if (reg >= mReg2Idx.size()) mReg2Idx.resize(reg + 1, invalidReg);
      mReg2Idx[reg] = idx;
    }
  }

public:  // mark function
  void markAsDiscarded(uint32_t reg) {
    assert(mReg2Idx[reg] != invalidReg);
    const auto mask = static_cast<int64_t>(1) << mReg2Idx[reg];
    assert((mFree & mask) == 0);
    mFree ^= mask;
  }
  void markAsUsed(uint32_t reg) {
    assert(mReg2Idx[reg] != invalidReg);
    const auto mask = static_cast<int64_t>(1) << mReg2Idx[reg];
    assert((mFree & mask) == mask);
    mFree ^= mask;
  }

public:  // check function
  bool isFree(uint32_t reg) const {
    if (mReg2Idx[reg] == invalidReg) return false;
    const auto mask = static_cast<int64_t>(1) << mReg2Idx[reg];
    return (mFree & mask) == mask;
  }

public:  // get function
  uint32_t getFreeRegister() const {
    if (mFree == 0) return invalidReg;
    return mIdx2Reg[static_cast<uint32_t>(
      __builtin_ctzll(static_cast<uint64_t>(mFree)))];
  }
};

/*
 * @brief: MultiClassRegisterSelector Class
 * @note:
 *      Multi-Class Register Selector (include General-Purpose Registers and
 * Floating-Point Registers)
 */
class MultiClassRegisterSelector final {
  class TargetRegisterInfo& mRegisterInfo;
  std::vector<std::unique_ptr<RegisterSelector>> mSelectors;

public:
  MultiClassRegisterSelector(class TargetRegisterInfo& info)
    : mRegisterInfo(info) {
    for (uint32_t idx = 0; idx < info.get_alloca_class_cnt(); idx++) {
      mSelectors.push_back(
        std::make_unique<RegisterSelector>(info.get_allocation_list(idx)));
    }
  }
  ~MultiClassRegisterSelector() = default;

public:  // mark function
  void markAsDiscarded(MIROperand reg);
  void markAsUsed(MIROperand reg);

public:  // check function
  bool isFree(MIROperand reg) const;

public:  // get function
  MIROperand getFreeRegister(OperandType type);
};

}  // namespace mir
