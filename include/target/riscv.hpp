#pragma once
#include "mir/mir.hpp"
namespace riscv {
// clang-format off
enum RISCVRegister : uint32_t {
    GPRBegin,
    X0=GPRBegin, X1, X2, X3, X4, X5, X6, X7,
    X8, X9, X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X19, X20, X21, X22, X23,
    X24, X25, X26, X27, X28, X29, X30, X31,
    GPREnd,
    FPRBegin,
    F0=FPRBegin, F1, F2, F3, F4, F5, F6, F7,
    F8, F9, F10, F11, F12, F13, F14, F15,
    F16, F17, F18, F19, F20, F21, F22, F23,
    F24, F25, F26, F27, F28, F29, F30, F31,
    FPREnd,
};
// clang-format on


} // namespace riscv