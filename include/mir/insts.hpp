#pragma once
#include "mir/mir.hpp"

namespace mir {
class RtypeInst : public MIRInst {
    MIROperand* rd() { return _operands[0]; }
    MIROperand* rs1() { return _operands[1]; }
    MIROperand* rs2() { return _operands[2]; }
};

class ItypeInst : public MIRInst {
    auto rd() { return _operands[0]; }
    auto rs1() { return _operands[1]; }
    auto imm() { return _operands[2]; }
};


class StypeInst : public MIRInst {
    auto rd() { return _operands[0]; }
    auto rs1() { return _operands[1]; }
    auto imm() { return _operands[2]; }
};

class JtypeInst : public MIRInst {
    auto rd() {return _operands[0];}
    auto imm();
};
// class 
}  // namespace mir
