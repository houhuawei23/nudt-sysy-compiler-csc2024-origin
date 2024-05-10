#include "mir/utils.hpp"

#include <string_view>

namespace mir {
enum class DataSection {
    Data,
    RoData,
    Bss,
};
static const std::unordered_map<DataSection, std::string_view>
    data_section_names = {
        {DataSection::Data, ".data"},
        {DataSection::RoData, ".rodata"},
        {DataSection::Bss, ".bss"},
};

inline int32_t ilog2(size_t x) {
    return __builtin_ctzll(x);
}

void dump_assembly(std::ostream& os, MIRModule& module, CodeGenContext& ctx) {
    /* 1: data section */
    os << ".data\n";

    auto select_data_section = [](MIRRelocable* reloc) {
        if (auto data = dynamic_cast<MIRDataStorage*>(reloc)) {
            return data->is_readonly() ? DataSection::RoData
                                       : DataSection::Data;
        }
        if (auto zero = dynamic_cast<MIRZeroStorage*>(reloc)) {
            return DataSection::Bss;
        }
        assert(false && "Unsupported data section");
    };

    std::unordered_map<DataSection, std::vector<MIRGlobalObject*>>
        data_sections;

    for (auto& gobj : module.global_objs()) {
        data_sections[select_data_section(gobj->reloc.get())].emplace_back(
            gobj.get());
    }

    for (auto ds : {DataSection::Data, DataSection::RoData, DataSection::Bss}) {
        if (data_sections[ds].empty()) {
            continue;
        }
        os << data_section_names.at(ds) << "\n";
        for (auto gobj : data_sections[ds]) {
            os << ".globl " << gobj->reloc->name() << "\n";
            os << ".p2align " << ilog2(gobj->align) << std::endl;
            os << gobj->reloc->name() << ":\n";
            gobj->reloc->print(os, ctx);
        }
    }

    /* 2: text section */
    os << ".text\n";

    for (auto& func : module.functions()) {
        if (func->blocks().empty())
            continue;
        os << ".globl " << func->name() << "\n";

        for (auto& bb : func->blocks()) {
            if (bb == func->blocks().front()) {
                os << func->name() << ":\n";
                /* dump stack usage comment */
                uint32_t calleeArgument = 0, loacl = 0, reSpill = 0,
                         calleeSaved = 0;
                for (auto& [operand, stackobj] : func->stack_objs()) {
                    switch (stackobj.usage) {
                        case StackObjectUsage::CalleeArgument:
                            calleeArgument += stackobj.size;
                            break;
                        case StackObjectUsage::Local:
                            loacl += stackobj.size;
                            break;
                        case StackObjectUsage::RegSpill:
                            reSpill += stackobj.size;
                            break;
                        case StackObjectUsage::CalleeSaved:
                            calleeSaved += stackobj.size;
                            break;
                    }
                }
                os << "\t" << "# stack usage: \n";
                os << "\t# CalleeArgument=" << calleeArgument << ", ";
                os << "Local=" << loacl << ", \n";
                os << "\t# RegSpill=" << reSpill << ", ";
                os << "CalleeSaved=" << calleeSaved << "\n";
            } else
                os << bb->name() << ":\n";
            bb->print(os, ctx);
        }
    }
}

void forEachDefOperand(MIRBlock& block,
                       CodeGenContext& ctx,
                       const std::function<void(MIROperand* op)>& functor) {
    for (auto& inst : block.insts()) {
        auto& inst_info = ctx.instInfo.get_instinfo(inst);
        for (uint32_t idx = 0; idx < inst_info.operand_num(); idx++) {
            if (inst_info.operand_flag(idx) & OperandFlagDef) {
                functor(inst->operand(idx));
            }
        }
    }
}

void forEachDefOperand(MIRFunction& func,
                       CodeGenContext& ctx,
                       const std::function<void(MIROperand* op)>& functor) {
    for (auto& block : func.blocks()) {
        forEachDefOperand(*block, ctx, functor);
    }
}
}  // namespace mir
