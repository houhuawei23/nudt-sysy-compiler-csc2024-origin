#pragma once

#include <cstdint>

namespace utils {
inline bool isPowerOf2(size_t x) {
  return __builtin_popcountll(x) == 1;
}
inline size_t log2(size_t x) {
  return __builtin_ctzll(x);
}
}  // namespace utils