#pragma once
#include "ir/ir.hpp"

namespace mir {

// big endian OR little endian
enum Endian { Big, Little };

/*
 * @brief: DataLayout Class (抽象基类)
 * @note:
 *      Data Layout (virtual class, define the api)
 */
class DataLayout {
public:
  virtual ~DataLayout() = default;

public:  // get function
  virtual Endian edian() = 0;
  virtual size_t pointerSize() = 0;

  virtual size_t typeAlign(ir::Type* type) = 0;
  virtual size_t codeAlign() = 0;
  virtual size_t memAlign() { return 8; }
};
}  // namespace mir
