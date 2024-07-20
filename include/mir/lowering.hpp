#pragma once
#include "pass/analysisinfo.hpp"
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "mir/iselinfo.hpp"

namespace mir {

class FloatPointConstantPool {
  MIRDataStorage* mFloatDataStorage = nullptr;
  std::unordered_map<uint32_t, uint32_t> mFloatMap;

public:
  MIROperand* getFloatConstant(class LoweringContext& ctx, float val);
};

/* LoweringContext */
class LoweringContext {
public:
  Target& _target;
  MIRModule& _mir_module;
  MIRFunction* _mir_func = nullptr;
  MIRBlock* _mir_block = nullptr;

  // clang-format off
  std::unordered_map<ir::Value*, MIROperand*> _val_map;                /* local variable mappings */
  std::unordered_map<ir::Function*, MIRFunction*> func_map;            /* function */
  std::unordered_map<ir::GlobalVariable*, MIRGlobalObject*> gvar_map;  /* global */
  std::unordered_map<ir::BasicBlock*, MIRBlock*> _block_map;           /* local in function */
  // clang-format on

  /* Float Point Constant Pool */
  FloatPointConstantPool mFloatConstantPool;

  /* Pointer Type for Target Platform */
  OperandType _ptr_type = OperandType::Int64;

  /* Code Generate Context */
  CodeGenContext* _code_gen_ctx = nullptr;
  MIRFunction* memsetFunc;

public:
  LoweringContext(MIRModule& mir_module, Target& target)
    : _mir_module(mir_module), _target(target) {
    _mir_module.functions().push_back(
      std::make_unique<MIRFunction>("_memset", &mir_module));
    memsetFunc = _mir_module.functions().back().get();
  }

  // set function
  void set_code_gen_ctx(CodeGenContext* code_gen_ctx) {
    _code_gen_ctx = code_gen_ctx;
  }
  void set_mir_func(MIRFunction* mir_func) { _mir_func = mir_func; }
  void set_mir_block(MIRBlock* mir_block) { _mir_block = mir_block; }

  // get function
  MIRModule& get_mir_module() { return _mir_module; };
  MIRBlock* get_mir_block() const { return _mir_block; }
  OperandType get_ptr_type() { return _ptr_type; }

  // gen function
  MIROperand* new_vreg(ir::Type* type);
  MIROperand* new_vreg(OperandType type);

  // emit function
  void emit_inst(MIRInst* inst);
  void emit_copy(MIROperand* dst, MIROperand* src);

  // ir_val -> mir_operand
  void add_valmap(ir::Value* ir_val, MIROperand* mir_operand);
  MIROperand* map2operand(ir::Value* ir_val);
  MIRBlock* map2block(ir::BasicBlock* ir_block) {
    return _block_map.at(ir_block);
  }
};

std::unique_ptr<MIRModule> create_mir_module(
  ir::Module& ir_module,
  Target& target,
  pass::topAnalysisInfoManager* tAIM);
}  // namespace mir