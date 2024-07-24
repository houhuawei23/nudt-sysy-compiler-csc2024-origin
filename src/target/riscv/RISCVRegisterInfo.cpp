#include "mir/mir.hpp"
#include "mir/utils.hpp"
#include "target/riscv/RISCVTarget.hpp"
#include "autogen/riscv/InstInfoDecl.hpp"
#include "autogen/riscv/ISelInfoDecl.hpp"
#include "support/StaticReflection.hpp"

namespace mir {

std::vector<uint32_t>& RISCVRegisterInfo::get_allocation_list(
  uint32_t classId) {
  if (classId == 0) {  // General Purpose Registers
    static std::vector<uint32_t> list{
      // clang-format off
      // $a0-$a5
      RISCV::X10, RISCV::X11, RISCV::X12, RISCV::X13, RISCV::X14, RISCV::X15,
      // $a6-$a7
      RISCV::X16, RISCV::X17,
      // $t0-$t6
      // RISCV::X5, 
      RISCV::X6, RISCV::X7, RISCV::X28, RISCV::X29, RISCV::X30, RISCV::X31,
      // $s0-$s1
      RISCV::X8, RISCV::X9,
      // $s2-$s11
      RISCV::X18, RISCV::X19, RISCV::X20, RISCV::X21, RISCV::X22, RISCV::X23, RISCV::X24,
      RISCV::X25, RISCV::X26, RISCV::X27,
      // $gp
      RISCV::X3,
      // clang-format on
    };
    return list;
  } else if (classId == 1) {  // Floating Point Registers
    static std::vector<uint32_t> list{
      // clang-format off
      // $fa0-$fa5
      RISCV::F10, RISCV::F11, RISCV::F12, RISCV::F13, RISCV::F14, RISCV::F15,
      // $fa6-$fa7
      RISCV::F16, RISCV::F17,
      // $ft0-$ft11
      RISCV::F0, RISCV::F1, RISCV::F2, RISCV::F3, RISCV::F4, RISCV::F5, RISCV::F6, RISCV::F7,
      RISCV::F28, RISCV::F29, RISCV::F30, RISCV::F31,
      // $fs0-$fs1
      RISCV::F8, RISCV::F9,
      // $fs2-$fs11
      RISCV::F18, RISCV::F19, RISCV::F20, RISCV::F21, RISCV::F22, RISCV::F23, RISCV::F24,
      RISCV::F25, RISCV::F26, RISCV::F27,
      // clang-format on
    };
    return list;
  } else {
    assert(false && "invalid type regoster");
  }
}

}  // namespace mir