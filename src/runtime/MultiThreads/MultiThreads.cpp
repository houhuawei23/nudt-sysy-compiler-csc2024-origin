#include <unistd.h>
#include <cstdint>

using LoopFuncHeader = void (*)(int32_t beg, int32_t end);

namespace {}

extern "C" {

void parallelFor(int32_t beg, int32_t end, LoopFuncHeader func) {
  const auto size = static_cast<uint32_t>(end - beg);
  constexpr uint32_t smallTaskSize = 32;
  if (size <= smallTaskSize) {
    func(beg, end);
    return;
  }
  
}
}
