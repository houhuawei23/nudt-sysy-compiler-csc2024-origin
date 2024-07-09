#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "autogen/generic/InstInfoDecl.hpp"
#include "support/StaticReflection.hpp"

namespace mir {
uint32_t offset = GENERIC::GENERICInstBegin + 1;
InstInfo& TargetInstInfo::get_instinfo(uint32_t opcode) {
  return GENERIC::getGENERICInstInfo().get_instinfo(opcode + offset);
}

bool TargetInstInfo::matchBranch(MIRInst* inst,
                                 MIRBlock*& target,
                                 double& prob) {
  auto oldOpcode = inst->opcode();
  inst->set_opcode(oldOpcode + offset);
  bool res = GENERIC::getGENERICInstInfo().matchBranch(inst, target, prob);
  inst->set_opcode(oldOpcode);
  return res;
}

static std::string_view getType(OperandType type) {
  switch (type) {
    case OperandType::Bool: return "i1 ";
    case OperandType::Int8: return "i8 ";
    case OperandType::Int16: return "i16 ";
    case OperandType::Int32: return "i32 ";
    case OperandType::Int64: return "i64 ";
    case OperandType::Float32: return "f32 ";
    case OperandType::Special: return "special ";
    case OperandType::HighBits: return "hi ";
    case OperandType::LowBits: return "lo ";
    case OperandType::Alignment: return "align ";
    default: assert(false && "Invalid operand type");
  }
};
void dumpVirtualReg(std::ostream& os, MIROperand* operand) {
  assert(operand != nullptr);
  os << getType(operand->type()) << "v";
  os << (operand->reg() ^ virtualRegBegin);
}
}  // namespace mir

namespace mir::GENERIC {
struct OperandDumper {
  MIROperand* operand;
};

static std::ostream& operator<<(std::ostream& os, OperandDumper opdp) {
  auto operand = opdp.operand;
  os << "[";
  if (operand->is_reg()) {
    if (isVirtualReg(operand->reg())) {
      dumpVirtualReg(os, operand);
    } else if (isStackObject(operand->reg())) {
      os << "so" << (operand->reg() ^ stackObjectBegin);
    } else {
      os << "isa " << operand->reg();
    }
  } else if (operand->is_imm()) {
    os << getType(operand->type()) << operand->imm();
    if (operand->type() == OperandType::Special) {
      os << " (" << utils::enumName(static_cast<CompareOp>(operand->imm()))
         << ")";
    }

  } else if (operand->is_prob()) {
    os << "prob " << operand->prob();
  } else if (operand->is_reloc()) {
    os << "reloc ";
    os << operand->reloc()->name();
  } else {
    os << "unknown";
  }
  os << "]";
  return os;
}

}  // namespace mir::GENERIC

#include "autogen/generic/InstInfoImpl.cpp"
