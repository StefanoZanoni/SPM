#ifndef SPM_TIMER_H
#define SPM_TIMER_H

#include <chrono>

/**
 * Measure the execution time of a function.
 * @tparam Func The type of the function to measure.
 * @param func The function to measure.
 * @return The execution time of the function in seconds.
 */
template<typename Func>
double measureExecutionTime(Func func) {
    const auto start = std::chrono::high_resolution_clock::now();
    func();
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - start;
    return duration.count();
}

#endif //SPM_TIMER_H