#pragma once
#include "ir/ir.hpp"

namespace mir {
enum Endian { Big, Little };

// virtual class, define the api
class DataLayout {
   public:
    virtual ~DataLayout() = default;
    virtual Endian edian() = 0;
    // getBuiltinAlignment
    virtual size_t type_align(ir::Type* type) = 0;
    virtual size_t ptr_size() = 0;
    virtual size_t code_align() = 0;
    virtual size_t mem_align() { return 8; }
};

}  // namespace mir
