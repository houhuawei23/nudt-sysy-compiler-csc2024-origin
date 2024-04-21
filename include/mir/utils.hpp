#pragma once
#include "mir/mir.hpp"
#include "mir/target.hpp"

#include <iostream>

namespace mir {
void dump_assembly(std::ostream& os, MIRModule& module, CodeGenContext& ctx);
}  // namespace mir
