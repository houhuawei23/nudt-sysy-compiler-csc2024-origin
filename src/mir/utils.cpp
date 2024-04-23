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

void dump_assembly(std::ostream& os, MIRModule& module, CodeGenContext& ctx) {
    /* data section */
    os << ".data\n";

    auto select_data_section = [](MIRRelocable* reloc) {
        if (auto data = dynamic_cast<MIRDataStorage*>(reloc)) {
            return data->is_readonly() ? DataSection::RoData : DataSection::Data;
        }
        if (auto zero = dynamic_cast<MIRZeroStorage*>(reloc)) {
            return DataSection::Bss;
        }
        // else jumptable
        assert(false && "Unsupported data section");
    };

    std::unordered_map<DataSection, std::vector<MIRGlobalObject*>> data_sections;

    for (auto& gobj : module.global_objs()) {
        data_sections[select_data_section(gobj->_reloc.get())].emplace_back(gobj.get());
    }

    for (auto ds : {DataSection::Data, DataSection::RoData, DataSection::Bss}) {
        if (data_sections[ds].empty()) {
            continue;
        }
        os << data_section_names.at(ds) << "\n";
        for (auto gobj : data_sections[ds]) {
            os << ".globl" << gobj->_reloc->name() << ":\n";
            // gobj->reloc->print
        }
    }

    /* text section */
    os << ".text\n";

    for (auto& func : module.functions()) {
        if (func->blocks().empty()) {
            continue;
        }
        os << ".globl " << func->name() << "\n";

        for (auto& bb : func->blocks()) {
            // os << bb->name() << ":\n";
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
            } else {
                os << bb->name() << ":\n";
            }
            bb->print(os, ctx);
        }
    }
}
}  // namespace mir
