#include "mir/mir.hpp"

#include "target/generic/generic.hpp"
#include "target/generic/generictarget.hpp"
#include "target/generic/InstInfoDecl.hpp"

#include <iostream>

namespace mir {
void GENERICTarget::emit_assembly(std::ostream& out, MIRModule& module) {
    std::cerr << "Not Impl emit_assembly" << std::endl;
    for (auto& fun : module.functions()) {
        std::cout << "function: " << std::endl;
        for (auto& bb : fun->blocks()) {
            // std::cout << "hello" << std::endl;
            std::cout << "mblock: " << std::endl;
            for (auto& inst : bb->insts()) {
                auto& instinfo = get_target_inst_info().get_instinfo(inst);
                instinfo.print(std::cout, *inst, false);
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}
}  // namespace mir
