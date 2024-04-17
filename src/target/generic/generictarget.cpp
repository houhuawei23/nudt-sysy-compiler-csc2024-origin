#include "mir/mir.hpp"

#include "target/generic/generic.hpp"
#include "target/generic/generictarget.hpp"
#include "target/generic/InstInfoDecl.hpp"

#include <iostream>

namespace mir {
void GENERICTarget::emit_assembly(std::ostream& out, MIRModule& module) {
    std::cerr << "Not Impl emit_assembly" << std::endl;
}
}  // namespace mir
