#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
#include "autogen/generic/InstInfoDecl.hpp"
#include "support/StaticReflection.hpp"

namespace mir {

static bool isOperandFReg(MIROperand operand) {
  return operand.isReg() and isFloatType(operand.type());
}

uint32_t offset = GENERIC::GENERICInstBegin + 1;
const InstInfo& TargetInstInfo::getInstInfo(uint32_t opcode) const {
  return GENERIC::getGENERICInstInfo().getInstInfo(opcode + offset);
}

bool TargetInstInfo::matchBranch(MIRInst* inst,
                                 MIRBlock*& target,
                                 double& prob) const {
  auto oldOpcode = inst->opcode();
  inst->set_opcode(oldOpcode + offset);
  bool res = GENERIC::getGENERICInstInfo().matchBranch(inst, target, prob);
  inst->set_opcode(oldOpcode);
  return res;
}

bool TargetInstInfo::matchCopy(MIRInst* inst,
                               MIROperand& dst,
                               MIROperand& src) const {
  //
  const auto& info = getInstInfo(inst);
  if (requireFlag(info.inst_flag(), InstFlagRegCopy)) {
    if (info.operand_num() != 2) {
      std::cerr << "Error: invalid operand number for copy instruction: \n";
      info.print(std::cerr, *inst, false);
      std::cerr << std::endl;
    }
    assert(info.operand_num() == 2);
    dst = inst->operand(0);
    src = inst->operand(1);
    return (isOperandIReg(dst) and isOperandIReg(src)) or
           (isOperandFReg(dst) and isOperandFReg(src));
  }
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
    case OperandType::Alignment:
      return "align ";
    default:
      assert(false && "Invalid operand type");
  }
};
void dumpVirtualReg(std::ostream& os, const MIROperand& operand) {
  // assert(operand != nullptr);
  assert(isVirtualReg(operand.reg()));
  os << getType(operand.type()) << "v";
  os << (operand.reg() ^ virtualRegBegin);
}
}  // namespace mir

namespace mir::GENERIC {
struct OperandDumper {
  MIROperand operand;
};

static std::ostream& operator<<(std::ostream& os, OperandDumper opdp) {
  auto operand = opdp.operand;
  os << "[";
  if (operand.isReg()) {
    if (isVirtualReg(operand.reg())) {
      dumpVirtualReg(os, operand);
    } else if (isStackObject(operand.reg())) {
      os << "so" << (operand.reg() ^ stackObjectBegin);
    } else {
      os << "isa " << operand.reg();
    }
  } else if (operand.isImm()) {
    os << getType(operand.type()) << operand.imm();
    if (operand.type() == OperandType::Special) {
      os << " (" << utils::enumName(static_cast<CompareOp>(operand.imm()))
         << ")";
    }

  } else if (operand.isProb()) {
    os << "prob " << operand.prob();
  } else if (operand.isReloc()) {
    os << "reloc ";
    os << operand.reloc()->name();
  } else {
    os << "unknown";
  }
  os << "]";
  return os;
}

}  // namespace mir::GENERIC

#include "autogen/generic/InstInfoImpl.hpp"
