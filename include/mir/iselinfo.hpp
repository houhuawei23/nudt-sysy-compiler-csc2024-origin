#pragma once
#include <unordered_set>
#include "mir/mir.hpp"
#include "mir/instinfo.hpp"
// #include "mir/target.hpp"
#include <optional>

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

    void remove_inst(MIRInst* inst);
    void replace_operand(MIROperand* src, MIROperand* dst);

    MIROperand* get_inst_def(MIRInst* inst);

    void insert_inst(MIRInst* inst) {
        assert(inst != nullptr);
        _curr_block->insts().emplace(_insert_point, inst);
    }
    MIRInst* new_inst(uint32_t opcode) {
        MIRInst* inst = new MIRInst(opcode);
        insert_inst(inst);
        return inst;
    }
    CodeGenContext& codegen_ctx() { return _codegen_ctx; }
    MIRBlock* curr_block() { return _curr_block; }
};

struct InstLegalizeContext final {
    MIRInst*& inst;
    MIRInstList& instructions;
    MIRInstList::iterator iter;
    CodeGenContext& codeGenCtx;
    std::optional<std::list<std::unique_ptr<MIRBlock>>::iterator> blockIter;
    MIRFunction& func;
};

class TargetISelInfo {
   public:
    virtual ~TargetISelInfo() = default;
    virtual bool is_legal_geninst(uint32_t opcode) const = 0;

    virtual bool match_select(MIRInst* inst, ISelContext& ctx) const = 0;

    /* */
    virtual void legalizeInstWithStackOperand(InstLegalizeContext& ctx,
                                              MIROperand* op,
                                              StackObject& obj) const = 0;

    virtual void postLegalizeInst(const InstLegalizeContext& ctx) const = 0;
};

enum CompareOp : uint32_t {
    ICmpEqual,
    ICmpNotEqual,
    ICmpSignedLessThan,
    ICmpSignedLessEqual,
    ICmpSignedGreaterThan,
    ICmpSignedGreaterEqual,
    ICmpUnsignedLessThan,
    ICmpUnsignedLessEqual,
    ICmpUnsignedGreaterThan,
    ICmpUnsignedGreaterEqual,

    FCmpOrderedEqual,
    FCmpOrderedNotEqual,
    FCmpOrderedLessThan,
    FCmpOrderedLessEqual,
    FCmpOrderedGreaterThan,
    FCmpOrderedGreaterEqual,
    FCmpUnorderedEqual,
    FCmpUnorderedNotEqual,
    FCmpUnorderedLessThan,
    FCmpUnorderedLessEqual,
    FCmpUnorderedGreaterThan,
    FCmpUnorderedGreaterEqual
};
static bool isCompareOp(MIROperand* operand, CompareOp cmpOp) {
    auto op = static_cast<uint32_t>(operand->imm());
    return op == static_cast<uint32_t>(cmpOp);
}

//! helper function to create a new MIRInstq

uint32_t select_copy_opcode(MIROperand* dst, MIROperand* src);

inline MIROperand* getHighBits(MIROperand* operand) {
    assert(isOperandReloc(operand));
    return new MIROperand{operand->storage(), OperandType::HighBits};
}
inline MIROperand* getLowBits(MIROperand* operand) {
    assert(isOperandReloc(operand));
    return new MIROperand{operand->storage(), OperandType::LowBits};
}

}  // namespace mir
