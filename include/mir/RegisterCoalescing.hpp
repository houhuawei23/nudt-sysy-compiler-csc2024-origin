#include "mir/mir.hpp"
#include "mir/instinfo.hpp"

namespace mir {
void RegisterCoalescing(MIRFunction& mfunc, CodeGenContext& ctx) {
    while (genericPeepholeOpt(mfunc, ctx)) ;
}
}