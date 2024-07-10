#ifndef SPM_TIMER_H
#define SPM_TIMER_H

#include <chrono>

template<typename Func>
double measureExecutionTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

#endif //SPM_TIMER_H
