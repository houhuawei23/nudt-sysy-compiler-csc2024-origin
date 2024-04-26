#include "mir/mir.hpp"
#include "mir/instinfo.hpp"

#include "autogen/generic/InstInfoDecl.hpp"

namespace mir {

// TargetInstInfo
uint32_t offset = GENERIC::GENERICInstBegin + 1;
InstInfo& TargetInstInfo::get_instinfo(uint32_t opcode) { return GENERIC::getGENERICInstInfo().get_instinfo(opcode + offset); }

bool TargetInstInfo::matchBranch(MIRInst* inst, MIRBlock*& target, double& prob) {
    auto oldOpcode = inst->opcode();
    inst->set_opcode(oldOpcode + offset);
    bool res = GENERIC::getGENERICInstInfo().matchBranch(inst, target, prob);
    inst->set_opcode(oldOpcode);
    return false;
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
    }
};

void dumpVirtualReg(std::ostream& os, MIROperand* operand) {
    assert(operand != nullptr);
    os << '[' << getType(operand->type()) << "vreg";
    os << (operand->reg() ^ virtualRegBegin) << ']';
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
