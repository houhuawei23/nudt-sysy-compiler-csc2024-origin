#pragma once
#include "ir/ir.hpp"

namespace mir {
enum Endian { Big, Little };

// virtual class, define the api
class DataLayout {
   public:
    virtual ~DataLayout() = default;
    virtual Endian edian() const = 0;
    // getBuiltinAlignment
    virtual size_t type_align(const ir::Type* type) const = 0;
    virtual size_t ptr_size() const = 0;
    virtual size_t code_align() const = 0;
    virtual size_t mem_align() const { return 8; }
};


}  // namespace mir
