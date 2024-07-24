#include "mir/registerinfo.hpp"

namespace mir {

void MultiClassRegisterSelector::markAsDiscarded(MIROperand reg) {
  assert(isISAReg(reg.reg()));
  const auto classId = mRegisterInfo.getAllocationClass(reg.type());
  auto& selector = *mSelectors[classId];
  selector.markAsDiscarded(reg.reg());
}
void MultiClassRegisterSelector::markAsUsed(MIROperand reg) {
  assert(isISAReg(reg.reg()));
  const auto classId = mRegisterInfo.getAllocationClass(reg.type());
  auto& selector = *mSelectors[classId];
  selector.markAsUsed(reg.reg());
}
bool MultiClassRegisterSelector::isFree(MIROperand reg) const {
  assert(isISAReg(reg.reg()));
  const auto classId = mRegisterInfo.getAllocationClass(reg.type());
  auto& selector = *mSelectors[classId];
  return selector.isFree(reg.reg());
}
MIROperand MultiClassRegisterSelector::getFreeRegister(
  OperandType type) {
  const auto classId = mRegisterInfo.getAllocationClass(type);
  const auto& selector = *mSelectors[classId];
  const auto reg = selector.getFreeRegister();
  if (reg == invalidReg) return MIROperand{};
  return MIROperand::asISAReg(reg,
                              mRegisterInfo.getCanonicalizedRegisterType(type));
}

}  // namespace mir
