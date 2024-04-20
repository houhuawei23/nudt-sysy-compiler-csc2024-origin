#include "mir/mir.hpp"
#include "mir/instinfo.hpp"

#include "target/generic/InstInfoDecl.hpp"

namespace mir {
uint32_t offset = GENERIC::GENERICInstBegin + 1;
InstInfo& TargetInstInfo::get_instinfo(uint32_t opcode) {
    return GENERIC::getGENERICInstInfo().get_instinfo(opcode + offset);
}
static std::string_view getType(OperandType type) {
    switch (type) {
        case OperandType::Bool:
            return "i1 ";
        case OperandType::Int8:
            return "i8 ";
        case OperandType::Int16:
            return "i16 ";
        case OperandType::Int32:
            return "i32 ";
        case OperandType::Int64:
            return "i64 ";
        case OperandType::Float32:
            return "f32 ";
        case OperandType::Special:
            return "special ";
        case OperandType::HighBits:
            return "hi ";
        case OperandType::LowBits:
            return "lo ";
        default:
            assert(false && "Invalid operand type");
            return "unknown ";
            // reportUnreachable(CMMC_LOCATION());
    }
};

// void dumpVirtualReg(std::ostream& out, const MIROperand& operand) {
//     out << '[' << getType(operand.type()) << 'v'
//         << (operand.reg() ^ virtualRegBegin)
//         << (operand.regFlag() & RegisterFlagDead ? " <dead>" : "") << ']';
// }
void dumpVirtualReg(std::ostream& os, MIROperand* operand) {
    assert(operand != nullptr);
    os << '[' << getType(operand->type()) << "vreg"
       << (operand->reg() ^ virtualRegBegin) << ']';
    //    << (operand->regFlag() & RegisterFlagDead ? " <dead>" : "") << ']';
}

}  // namespace mir

namespace mir::GENERIC {
// struct OperandDumper final {
//     MIROperand* operand;
// };

// static std::ostream& operator<<(std::ostream& os, OperandDumper& opdp) {
//     auto operand = opdp.operand;
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
