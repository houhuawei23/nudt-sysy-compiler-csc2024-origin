#pragma once

#include <chrono>

namespace utils{

using Clock = std::chrono::high_resolution_clock;
using Duration = Clock::duration;
using TimePoint = Clock::time_point;


}  // namespace utils
