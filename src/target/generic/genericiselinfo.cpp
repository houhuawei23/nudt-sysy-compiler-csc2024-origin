#include "mir/mir.hpp"
#include "mir/iselinfo.hpp"

#include "autogen/generic/InstInfoDecl.hpp"

namespace mir::GENERIC {


}  // namespace mir::GENERIC

#include "autogen/generic/ISelInfoImpl.hpp"

namespace mir::GENERIC {
bool GENERICISelInfo::is_legal_geninst(uint32_t opcode) const {
    return true;
}
bool GENERICISelInfo::match_select(MIRInst* inst, ISelContext& ctx) const {
    return matchAndSelectImpl(inst, ctx);
}
}