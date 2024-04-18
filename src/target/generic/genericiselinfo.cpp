#include "mir/mir.hpp"
#include "mir/iselinfo.hpp"

#include "target/generic/InstInfoDecl.hpp"
#include "target/generic/ISelInfoImpl.hpp"

namespace mir::GENERIC {
bool GENERICISelInfo::is_legal_geninst(uint32_t opcode) const {
    return true;
}
bool GENERICISelInfo::match_select(MIRInst* inst, ISelContext& ctx) const {
    return matchAndSelectImpl(inst, ctx);
}
}  // namespace mir::GENERIC
