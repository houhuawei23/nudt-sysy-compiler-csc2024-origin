#include "ir/ir.hpp"

#include "mir/mir.hpp"
#include "mir/instinfo.hpp"

#include "target/riscv/InstInfoDecl.hpp"
#include "target/riscv/riscv.hpp"

#include <iostream>

namespace mir::RISCV {

static std::ostream& operator<<(std::ostream& out, mir::MIROperand& mop) {
    if (mop.is_reg()) {
        // if ()
    } else if (mop.is_imm()) {

    } else if (mop.is_prob()) {

    } else if (mop.is_reloc()) {

    } else {
        std::cerr << "unknown operand type" << std::endl;
    }


}

}  // namespace RISCV
