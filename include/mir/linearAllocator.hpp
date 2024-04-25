#pragma once
#include "mir/LiveInterval.hpp"

namespace mir {
static void linearAllocator(MIRFunction& mfunc, CodeGenContext& ctx) {
    calcLiveIntervals(mfunc, ctx);
}

}