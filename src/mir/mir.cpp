#include "ir/ir.hpp"
#include "mir/mir.hpp"
#include "mir/target.hpp"
#include "support/StaticReflection.hpp"
namespace mir {
/* utils function */
void find_consecutive_zeros(std::vector<uint32_t> data) {
    int start = -1;  // 初始时未找到连续0的起始位置

    for (int i = 0; i < data.size(); i++) {
        if (data[i] == 0) {
            if (start == -1) {
                start = i;  // 记录连续0的起始位置
            }
        } else {
            if (start != -1) {
                int end = i - 1; // 记录连续0的结束位置
                start = -1; // 重置起始位置
            }
        }
    }

    // 检查最后一个元素是否是0，如果是，需要记录最后一个序列
    if (start != -1) {
        int end = data.size() - 1;
    }
}

/* Information of MIRBlock */
void MIRBlock::print(std::ostream& os, CodeGenContext& ctx) {
    os << " ";
    for (auto& inst : mInsts) {
        os << "\t";
        auto& info = ctx.instInfo.getInstInfo(inst);
        os << "[" << info.name() << "] ";
        info.print(os, *inst, false);
        os << std::endl;
    }
}
/* Information of MIRFunction */
void MIRFunction::print(std::ostream& os, CodeGenContext& ctx) {
    for (auto &[ref, obj] : mStackObjects) {
        os << " so" << (ref.reg() ^ stackObjectBegin)
           << " size = " << obj.size << " align = " << obj.alignment
           << " offset = " << obj.offset 
           << " usage = " << utils::enumName(obj.usage)
           << std::endl;
    }
    for (auto& block : mBlocks) {
        os << block->name() << ":" << std::endl;
        block->print(os, ctx);
    }
}
/* Information of MIRRelocable */
void MIRZeroStorage::print(std::ostream& os, CodeGenContext& ctx) {}
void MIRDataStorage::print(std::ostream& os, CodeGenContext& ctx) {
    // if (mData.size() > 100) {
    //     for (auto& val : mData) {
    //         os << "\t.4byte\t";
    //         if (is_float()) os << val << std::endl;
    //         else os << val << std::endl;
    //     }
    // } else {
    //     find_consecutive_zeros(mData);
    // }
    for (auto& val : mData) {
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
    if(mInsts.empty()) return false;

    for(auto& inst : mInsts) {
        if(not inst->verify(os, ctx)) {
            return false;
        }
    }
    const auto lastInst = mInsts.back();
    const auto& lastInstInfo = ctx.instInfo.getInstInfo(lastInst);
    if((lastInstInfo.inst_flag() & InstFlagTerminator) == 0) {
        os << "Error: block " << name() << " does not end with a terminator" << std::endl;
        return false;
    }
    for(auto& inst : mInsts) {
        const auto& info = ctx.instInfo.getInstInfo(inst);
        if((info.inst_flag() & InstFlagTerminator) and inst != lastInst) {
            os << "Error: block " << name() << " has multiple terminators" << std::endl;
            return false;
        } 
    }
    return true;
}
bool MIRFunction::verify(std::ostream& os, CodeGenContext& ctx) const {
    for(auto& block : mBlocks) {
        if(not block->verify(os, ctx)) {
            return false;
        }
    }
    return true;
}
}