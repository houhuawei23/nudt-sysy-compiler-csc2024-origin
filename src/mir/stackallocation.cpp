#include "mir/utils.hpp"
#include "target/riscv/riscv.hpp"
namespace mir {
struct StackObjectInterval final {
    uint32_t begin, end;
};
using Intervals =
    std::unordered_map<MIROperand*, StackObjectInterval*, MIROperandHasher>;

struct Slot final {
    uint32_t color;
    uint32_t end;

    bool operator<(const Slot& rhs) const noexcept { return end > rhs.end; }
};

static void remove_unused_spill_stack_objects(MIRFunction* mfunc) {
    std::unordered_set<MIROperand*> spill_stack;
    /* find the unused spill_stack */
    {
        for (auto& [op, stack] : mfunc->stack_objs()) {
            assert(isStackObject(op->reg()));
            if (stack.usage == StackObjectUsage::RegSpill)
                spill_stack.emplace(op);
        }
        for (auto& block : mfunc->blocks()) {
            for (auto inst : block->insts()) {
                if (inst->opcode() == InstLoadRegFromStack) {
                    spill_stack.erase(inst->operand(1));
                }
            }
        }
    }

    /* remove dead store */
    {
        for (auto& block : mfunc->blocks()) {
            block->insts().remove_if([&](auto inst) {
                return inst->opcode() == InstStoreRegToStack &&
                       spill_stack.count(inst->operand(0));
            });
        }
        for (auto op : spill_stack) {
            mfunc->stack_objs().erase(op);
        }
    }
}

/*
 * Stack Layout
 * ------------------------ <----- Last sp
 * Locals & Spills
 * ------------------------
 * Return Address
 * ------------------------
 * Callee Arguments
 * ------------------------ <----- Current sp
 */
void allocateStackObjects(MIRFunction* func, CodeGenContext& ctx) {
    bool debugSA = false;
    auto dumpOperand = [&](MIROperand* op) {
        std::cerr << mir::RISCV::OperandDumper{op} << std::endl;
    };
    // func is like a callee
    /* 1. callee saved: 被调用者需要保存的寄存器 */
    std::unordered_set<MIROperand*> callee_saved;
    for (auto& block : func->blocks()) {
        forEachDefOperand(*block, ctx, [&](MIROperand* op) {
            if (op->is_unused())
                return;
            if (op->is_reg() && isISAReg(op->reg()) &&
                ctx.frameInfo.is_callee_saved(*op)) {
                if (debugSA)
                    dumpOperand(op);

                callee_saved.insert(MIROperand::as_isareg(
                    op->reg(),  //
                    ctx.registerInfo->getCanonicalizedRegisterType(
                        op->type())));
            }
        });
    }
    const auto preAllocatedBase = ctx.frameInfo.insert_prologue_epilogue(
        func, callee_saved, ctx,
        ctx.registerInfo->get_return_address_register());

    remove_unused_spill_stack_objects(func);  // remove dead code

    int32_t allocationBase = 0;
    auto sp_alignment =
        static_cast<int32_t>(ctx.frameInfo.get_stackpointer_alignment());
    auto align_to = [&](int32_t alignment) {
        assert(alignment <= sp_alignment);
        allocationBase =
            (allocationBase + alignment - 1) / alignment * alignment;
    };

    /* 2. callee arguments */
    for (auto& [ref, stack] : func->stack_objs()) {
        assert(isStackObject(ref->reg()));
        if (stack.usage == StackObjectUsage::CalleeArgument) {
            allocationBase =
                std::max(allocationBase,
                         stack.offset + static_cast<int32_t>(stack.size));
        }
    }
    align_to(sp_alignment);

    /* 3. local variables */
    std::vector<MIROperand*> local_callee_saved;
    for (auto& [op, stack] : func->stack_objs()) {
        if (stack.usage == StackObjectUsage::CalleeSaved) {
            local_callee_saved.push_back(op);
        }
    }
    std::sort(local_callee_saved.begin(), local_callee_saved.end(),
              [&](const MIROperand* lhs, const MIROperand* rhs) {
                  return lhs->reg() > rhs->reg();
              });

    auto allocate_for = [&](StackObject& stack) {
        align_to(static_cast<int32_t>(stack.alignment));
        stack.offset = allocationBase;
        allocationBase += static_cast<int32_t>(stack.size);
    };
    for (auto op : local_callee_saved) {
        allocate_for(func->stack_objs().at(op));
    }
    for (auto& [op, stack] : func->stack_objs()) {
        if (stack.usage == StackObjectUsage::Local ||
            stack.usage == StackObjectUsage::RegSpill) {
            allocate_for(stack);
        }
    }

    align_to(sp_alignment);
    auto gap =
        (preAllocatedBase + sp_alignment - 1) / sp_alignment * sp_alignment -
        preAllocatedBase;
    auto stack_size = allocationBase + gap;
    assert((stack_size + preAllocatedBase) % sp_alignment == 0);
    for (auto& [op, stack] : func->stack_objs()) {
        assert(isStackObject(op->reg()));
        if (stack.usage == StackObjectUsage::Argument) {
            stack.offset += stack_size + preAllocatedBase;
        }
    }

    /* 4. emit prologue and epilogue */
    ctx.frameInfo.emit_postsa_prologue(func->blocks().front().get(),
                                       stack_size);

    for (auto& block : func->blocks()) {
        auto terminator = block->insts().back();
        auto& instInfo = ctx.instInfo.get_instinfo(terminator);
        if (requireFlag(instInfo.inst_flag(), InstFlagReturn)) {
            ctx.frameInfo.emit_postsa_epilogue(block.get(),
                                               stack_size);
        }
    }
}

}  // namespace mir
