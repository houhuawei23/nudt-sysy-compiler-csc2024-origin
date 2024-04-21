#include "mir/mir.hpp"
#include "mir/utils.hpp"

#include "target/generic/generic.hpp"
#include "target/generic/generictarget.hpp"
#include "target/generic/InstInfoDecl.hpp"

#include <iostream>

namespace mir {
void GENERICTarget::emit_assembly(std::ostream& out, MIRModule& module) {

    auto& target = *this;
    CodeGenContext codegen_ctx {
        target,
        target.get_datalayout(),
        target.get_target_inst_info(),
        target.get_target_frame_info(),
        MIRFlags{false, false}
    };
    dump_assembly(out, module, codegen_ctx);

}
}  // namespace mir
