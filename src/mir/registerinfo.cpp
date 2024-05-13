#include "mir/registerinfo.hpp"

namespace mir {
RegisterSelector::RegisterSelector(const std::vector<uint32_t>& list) {
    assert(list.size() <= maxRegisterCount);
    _mFree = (static_cast<int64_t>(1) << list.size()) - 1;
    for (uint32_t idx = 0; idx < list.size(); idx++) {
        auto reg = list[idx];
        _mIdx2Reg[idx] = reg;
        if (reg >= _mReg2Idx.size())
            _mReg2Idx.resize(reg + 1, invalidReg);
        _mReg2Idx[reg] = idx;
    }
}

void RegisterSelector::markAsDiscarded(uint32_t reg) {
    assert(_mReg2Idx[reg] != invalidReg);
    auto mask = static_cast<int64_t>(1) << _mReg2Idx[reg];
    assert((mask & _mFree) == 0);
    _mFree ^= mask;
}

void RegisterSelector::markAsUsed(uint32_t reg) {
    assert(_mReg2Idx[reg] != invalidReg);
    auto mask = static_cast<int64_t>(1) << _mReg2Idx[reg];
    assert((mask & _mFree) == mask);
    _mFree ^= mask;
}

bool RegisterSelector::isFree(uint32_t reg) const {
    if (_mReg2Idx[reg] == invalidReg)
        return false;
    auto mask = static_cast<int64_t>(1) << _mReg2Idx[reg];
    return (_mFree & mask) == mask;
}

uint32_t RegisterSelector::getFreeRegister() const {
    if (_mFree == 0)
        return invalidReg;
    return _mIdx2Reg[static_cast<uint32_t>(
        __builtin_ctzll(static_cast<uint64_t>(_mFree)))];
}

void MultiClassRegisterSelector::markAsDiscarded(MIROperand reg) {
    assert(isISAReg(reg.reg()));
    auto classId = _mRegisterInfo.get_alloca_class(reg.type());
    auto& selector = *_mSelectors[classId];
    selector.markAsDiscarded(reg.reg());
}

void MultiClassRegisterSelector::markAsUsed(MIROperand reg) {
    assert(isISAReg(reg.reg()));
    auto classId = _mRegisterInfo.get_alloca_class(reg.type());
    auto& selector = *_mSelectors[classId];
    std::cerr << "Marking register " << reg.reg() << " as used in class "
              << classId << std::endl;
    selector.markAsUsed(reg.reg());
}

bool MultiClassRegisterSelector::isFree(MIROperand reg) const {
    assert(isISAReg(reg.reg()));
    auto classId = _mRegisterInfo.get_alloca_class(reg.type());
    auto& selector = *_mSelectors[classId];
    return selector.isFree(reg.reg());
}

MIROperand* MultiClassRegisterSelector::getFreeRegister(OperandType type) {
    auto classId = _mRegisterInfo.get_alloca_class(type);
    auto& selector = *_mSelectors[classId];
    auto reg = selector.getFreeRegister();
    if (reg == invalidReg) return new MIROperand;
    return MIROperand::as_isareg(reg, _mRegisterInfo.getCanonicalizedRegisterType(type));

}
}  // namespace mir
