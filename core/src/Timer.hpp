/**
 * Copyright (C) Stylianos Piperakis, Ownage Dynamics L.P.
 * Serow is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, version 3.
 *
 * Serow is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Serow. If not,
 * see <https://www.gnu.org/licenses/>.
 **/
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
