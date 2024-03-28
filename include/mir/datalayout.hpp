#pragma once
#include "ir/ir.hpp"

namespace mir {
enum Edian { Big, Little };

// virtual class, define the api
class DataLayout {
   public:
    virtual ~DataLayout() = default;
    virtual Edian edian() const = 0;
    virtual int type_align(const ir::Type* type) const = 0;
    vitrual int ptr_size() const = 0;
    virtual int code_align() const = 0;
    virtual int mem_align() const { return 8; }
};

class RISCVDataLayout : public DataLayout {
    
}
}  // namespace mir
