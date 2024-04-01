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
    std::vector<MIRGlobalObject*> _global_objs;
    ir::Module* _ir_module;

    public:
    MIRModule() = default;
    MIRModule(ir::Module* ir_module);

    void print(std::ostream &os);

    public:
    void print();
};

class MIRFunction {
   private:
    MIRModule* _parent;
    std::vector<MIRBlock*> _blocks;
    ir::Function* _ir_func;

    std::string _name;

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

    void inst_sel(ir::BasicBlock* ir_bb);

   public:
    void print();
};
// addi	sp,sp,-16
// sd	s0,8(sp)
// addi	s0,sp,16
class MIRInst {
    // static const int max_operand_num = 7;

   protected:
    MIRBlock* _parent;
    // std::array<MIROperand*, max_operand_num> _operands;
    std::vector<MIROperand*> _operands;

   public:
    MIRInst() = default;
    MIRInst(ir::Instruction* ir_inst, MIRBlock* parent);

   public:
    void print();
};

class MIROperand {
   private:
    // std::variant<MIRRegister /*others*/> _operand;
    // int tmp;
    union _operand {
        MIRRegister* reg;
        // int i;
        // float f;
    };

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

/*
 * @brief Global Value
 */
class MIRGlobalObject {
    private:
    MIRModule* _parent;
    ir::Value* _ir_global;

    public:
    MIRGlobalObject() = default;
    MIRGlobalObject(ir::Value* ir_global, MIRModule* parent) : _parent(parent), _ir_global(ir_global) {
        std::string var_name = _ir_global->name();
        var_name = var_name.substr(1, var_name.length() - 1);
        _ir_global->set_name(var_name);
    }

    public:
    void print(std::ostream& os);
};
}  // namespace mir
