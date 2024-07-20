#include "ir/ir.hpp"
#include "mir/mir.hpp"
#include "mir/target.hpp"

namespace mir {
void MIRBlock::print(std::ostream& os, CodeGenContext& ctx) {
    os << " ";
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

/* Information of MIRRelocable */
void MIRZeroStorage::print(std::ostream& os, CodeGenContext& ctx) {}
void MIRDataStorage::print(std::ostream& os, CodeGenContext& ctx) {
    for (auto& val : _data) {
        os << "\t.4byte\t";
        if (is_float()) os << val << std::endl;
        else os << val << std::endl;
    }
}

bool MIRInst::verify(std::ostream& os, CodeGenContext& ctx) const {
    // TODO: implement verification
    return true;
}

bool MIRBlock::verify(std::ostream& os, CodeGenContext& ctx) const {
    if(_insts.empty()) return false;

    for(auto& inst : _insts) {
        if(not inst->verify(os, ctx)) {
            return false;
        }
    }
    const auto lastInst = _insts.back();
    const auto& lastInstInfo = ctx.instInfo.get_instinfo(lastInst);
    if((lastInstInfo.inst_flag() & InstFlagTerminator) == 0) {
        os << "Error: block " << name() << " does not end with a terminator" << std::endl;
        return false;
    }
    for(auto& inst : _insts) {
        const auto& info = ctx.instInfo.get_instinfo(inst);
        if((info.inst_flag() & InstFlagTerminator) and inst != lastInst) {
            os << "Error: block " << name() << " has multiple terminators" << std::endl;
            return false;
        } 
    }
    return true;
}

bool MIRFunction::verify(std::ostream& os, CodeGenContext& ctx) const {
    for(auto& block : _blocks) {
        if(not block->verify(os, ctx)) {
            return false;
        }
    }
    return true;
}


// bool MIRModule::verify() const {
//     for (auto& func : _functions) {
//         if(not func->verify()) {
//             return false;
//         }
//     }
//     return true;
// }


}  // namespace mir