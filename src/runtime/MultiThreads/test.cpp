#include <iostream>
#include "MultiThreads.hpp"

using namespace std;

int globalSum;

void exampleFunc(int32_t beg, int32_t end) {
  int localSum = 0;
  for (int32_t i = beg; i < end; ++i) {
    globalSum += 1;
  }
  // globalSum += localSum;
}

int main() {
  int32_t beg = 0;
  int32_t end = 1e5;
  parallelFor(beg, end, exampleFunc);
  std::cout << "Global sum: " << globalSum << std::endl;
  return 0;
}