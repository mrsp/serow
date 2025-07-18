#pragma once
#include <atomic>
#include <chrono>
#include <limits>
#include <string>

// High-performance timer that calculates stats on-the-fly
class Timer {
private:
    std::atomic<uint64_t> count{0};
    std::atomic<double> sum{0.0};
    std::atomic<double> min_val{std::numeric_limits<double>::max()};
    std::atomic<double> max_val{std::numeric_limits<double>::lowest()};
    std::chrono::high_resolution_clock::time_point start_time;
    std::atomic<bool> is_running{false};

    void updateStats(double duration);

public:
    Timer() = default;

    void reset();

    void start();

    void stop();

    void logStats(const std::string& label = "") const;

    // Getters
    uint64_t getCount() const;

    double getMean() const;

    double getMin() const;

    double getMax() const;
};
