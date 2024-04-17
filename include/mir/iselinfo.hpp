#pragma once
#include <unordered_set>
#include "mir/mir.hpp"
// #include "mir/target.hpp"

namespace mir {
class CodeGenContext;
class ISelContext {
    CodeGenContext& _codegen_ctx;
    std::unordered_map<MIROperand*, MIRInst*> _inst_map, _constant_map;
    MIRBlock* _curr_block;
    std::list<MIRInst*>::iterator _insert_point;
    
    // mReplaceList
    std::unordered_map<MIROperand*, MIROperand*> _replace_map;

    std::unordered_set<MIRInst*> _remove_work_list, _replace_block_list;

    std::unordered_map<MIROperand*, uint32_t> _use_cnt;

   public:
    ISelContext(CodeGenContext& codegen_ctx) : _codegen_ctx(codegen_ctx) {}
    void run_isel(MIRFunction* func);
    bool has_one_use(MIROperand* op);
    MIRInst* lookup_def(MIROperand* op);
};

class TargetISelInfo {
   public:
    virtual ~TargetISelInfo() = default;
    virtual bool is_legal_geninst(uint32_t opcode) const = 0;
    virtual bool match_select(MIRInst* inst, ISelContext* ctx) const = 0;
};

//! helper function to create a new MIRInstq

uint32_t select_copy_opcode(MIROperand* dst, MIROperand* src);

}  // namespace mir
