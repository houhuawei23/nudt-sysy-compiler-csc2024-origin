// #include "target/generic/"
#include "mir/mir.hpp"
#include "mir/instinfo.hpp"

#include "target/generic/generic.hpp"
#include "autogen/generic/InstInfoDecl.hpp"

//! InstInfoImpl.hpp can olny be included once
/* it is more than like a cpp file, may need to change*/

#include <iostream>
namespace mir::GENERIC {

// static std::ostream& operator<<(std::ostream& os, MIROperand* operand) {
//     if (operand->is_reg()) {
//         if (isVirtualReg(operand->reg())) {
//             dumpVirtualReg(os, operand);
//         }
//         os << "reg: " << operand->reg();
//     }
//     if (operand->is_imm()) {
//         os << "imm: " << operand->imm();
//     } else if (operand->is_prob()) {
//         // os << "prob: " << operand->prob();
//         os << "prob ";
//     } else if (operand->is_reloc()) {
//         // operand->reloc()-
//         os << "reloc ";
//     } else {
//         std::cerr << "unknown operand type" << std::endl;
//     }

//     return os;
// }

}  // namespace mir::GENERIC

#include "autogen/generic/InstInfoImpl.cpp"