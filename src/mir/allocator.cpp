#include "mir/allocator.hpp"

namespace mir {
RegAllocator& RegAllocator::get() {
    static RegAllocator instance;
    return instance;
}

static void fastAllocator(MIRFunction& mfunc, CodeGenContext& ctx) {
    
}
}