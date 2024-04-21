#include "ir/ir.hpp"

#include "mir/mir.hpp"
#include "mir/target.hpp"

namespace mir {
void MIRBlock::print(std::ostream& os, CodeGenContext& ctx) {
    os << "  ";
    for (auto& inst : _insts) {
        os << "\t";
        auto& info = ctx.instInfo.get_instinfo(inst);
        // os << '[' << info.name() << ']';
        info.print(os, *inst, false);
        os << std::endl;
    }
}

void MIRFunction::print(std::ostream& os, CodeGenContext& ctx) {}

void MIRZeroStorage::print(std::ostream& os, CodeGenContext& ctx) {}

void MIRDataStorage::print(std::ostream& os, CodeGenContext& ctx) {}

}  // namespace mir
