#include "mir/utils.hpp"

#include "target/riscv/riscvtarget.hpp"
#include "autogen/riscv/InstInfoDecl.hpp"
#include "autogen/riscv/ISelInfoDecl.hpp"

namespace mir {

void RISCVTarget::emit_assembly(std::ostream& out, MIRModule& module) {
    auto& target = *this;
    CodeGenContext codegen_ctx{
        target, target.get_datalayout(), target.get_target_inst_info(),
        target.get_target_frame_info(), MIRFlags{false, false}};
    dump_assembly(out, module, codegen_ctx);
}


}  // namespace mir