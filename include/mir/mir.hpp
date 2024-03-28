#pragma once
#include <array>
#include <list>
#include <variant>
#include <vector>
#include "ir/ir.hpp"
namespace mir {
class MIRModule;
class MIRFunction;
class MIRBlock;
class MIROperand;
class MIRInst;
class MIRRegister;
class MIRGlobalObject;
// class MIR

class MIRModule {
   private:
    std::vector<MIRFunction*> _functions;
    std::vector<MIRGlobalObject*> global_objs;
    ir::Module* _ir_module;

   public:
    MIRModule() = default;
    MIRModule(ir::Module* ir_module);

   public:
    void print();
};

class MIRFunction {
   private:
    MIRModule* _parent;
    std::vector<MIRBlock*> _blocks;
    ir::Function* _ir_func;

   public:
    MIRFunction() = default;
    MIRFunction(ir::Function* ir_func, MIRModule* parent);

   public:
    void print();
};

class MIRBlock {
   private:
    MIRFunction* _parent;
    std::list<MIRInst*> _insts;
    ir::BasicBlock* _ir_block;

   public:
    MIRBlock() = default;
    MIRBlock(ir::BasicBlock* ir_block, MIRFunction* parent);

   public:
    void print();
};

class MIRInst {
    static const int max_operand_num = 7;

   private:
    MIRBlock* _parent;
    std::array<MIROperand*, max_operand_num> _operands;

   public:
    MIRInst() = default;
    MIRInst(ir::Instruction* ir_inst, MIRBlock* parent);

   public:
    void print();
};

class MIROperand {
   private:
    // std::variant<MIRRegister /*others*/> _operand;
    int tmp;

   public:
    void print();
};

class MIRRegister {
   public:
    enum RegType {
        VIRTUAL,
        PHYSIC,
    };

   private:
    RegType _type;

   public:
    void print();
};

class MIRGlobalObject {};
}  // namespace mir
