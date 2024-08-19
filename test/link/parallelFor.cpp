#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

using LoopFuncHeader = void (*)(int32_t beg, int32_t end);

extern "C" {
void parallelFor(int32_t beg, int32_t end, LoopFuncHeader parallelBody) {
  // Determine the number of threads you want to use
  const int32_t numThreads = 4;

  // Determine if we're going forward or backward
  bool isForward = beg < end;

  // Calculate the total number of iterations
  int32_t totalIterations = isForward ? end - beg : beg - end;
  int32_t iterationsPerThread = totalIterations / numThreads;
  int32_t remainderIterations = totalIterations % numThreads;

  // Create a vector to store the threads
  std::vector<std::thread> threads;

  int32_t currentBeg = beg;

  for (int32_t i = 0; i < numThreads; ++i) {
    int32_t currentEnd;

    // Calculate the end of the current thread's range
    if (isForward) {
      currentEnd = currentBeg + iterationsPerThread;
      if (i < remainderIterations) {
        currentEnd++;
      }
    } else {
      currentEnd = currentBeg - iterationsPerThread;
      if (i < remainderIterations) {
        currentEnd--;
      }
    }

    // Create and launch the thread
    std::cerr << "Launching thread " << i << " with range ["
              << currentBeg << ", " << currentEnd << ")\n";
    threads.emplace_back(parallelBody, currentBeg, currentEnd);

    // Move to the next range of iterations
    currentBeg = currentEnd;
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }
}
}
