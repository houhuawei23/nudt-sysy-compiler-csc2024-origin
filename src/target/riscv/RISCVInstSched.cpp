#include "ir/ir.hpp"

#include "mir/mir.hpp"
#include "mir/instinfo.hpp"

#include "target/riscv/RISCV.hpp"
#include "autogen/riscv/ScheduleModelDecl.hpp"

#include "autogen/riscv/ScheduleModelImpl.hpp"

namespace mir::RISCV {

MicroArchInfo& RISCVScheduleModel_sifive_u74::getMicroArchInfo() {
    static MicroArchInfo info{
        .enablePostRAScheduling = true,
        .hasRegRenaming = false,
        .hasMacroFusion = false,
        .issueWidth = 2,
        .outOfOrder = false,
        .hardwarePrefetch = true,
        .maxDataStreams = 8,
        .maxStrideByBytes = 256,
    };
    return info;
}
bool RISCVScheduleModel_sifive_u74::peepholeOpt(MIRFunction& func,
                                                CodeGenContext& context) {
    return false;
}
bool RISCVScheduleModel_sifive_u74::isExpensiveInst(MIRInst* inst,
                                                    CodeGenContext& context) {
    return false;
}

}  // namespace mir::RISCV

//! Dont Change This Line!"