#include "ir/ir.hpp"

#include "mir/mir.hpp"
#include "mir/target.hpp"

namespace mir {
void MIRBlock::print(std::ostream& os, CodeGenContext& ctx) {
    os << "  ";
    for (auto& inst : _insts) {
        os << "\t";
        auto& info = ctx.instInfo.get_instinfo(inst);
        os << "[" << info.name() << "] ";
        info.print(os, *inst, false);
        os << std::endl;
    }
}

void MIRFunction::print(std::ostream& os, CodeGenContext& ctx) {
    for (auto &[ref, obj] : _stack_objs) {
        os << " so" << (ref->reg() ^ stackObjectBegin)
           << " size = " << obj.size << " align = " << obj.alignment
           << " offset = " << obj.offset << std::endl;
    }
    for (auto& block : _blocks) {
        os << block->name() << ":" << std::endl;
        block->print(os, ctx);
    }
}

void MIRZeroStorage::print(std::ostream& os, CodeGenContext& ctx) {

}

void MIRDataStorage::print(std::ostream& os, CodeGenContext& ctx) {
    for(auto& val : _data) {
        // if (std::holds_alternative<<)
        os << "\t.4byte\t" << val << std::endl;
    }
}

}  // namespace mir
