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
#include "Timer.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

void Timer::updateStats(double duration) {
    count++;

    // Update sum
    double current_sum = sum.load();
    while (!sum.compare_exchange_weak(current_sum, current_sum + duration))
        ;

    // Update min
    double current_min = min_val.load();
    while (duration < current_min && !min_val.compare_exchange_weak(current_min, duration))
        ;

    // Update max
    double current_max = max_val.load();
    while (duration > current_max && !max_val.compare_exchange_weak(current_max, duration))
        ;
}

void Timer::reset() {
    count = 0;
    sum = 0.0;
    min_val = std::numeric_limits<double>::max();
    max_val = std::numeric_limits<double>::lowest();
    is_running = false;  // Ensure timer is not considered running after reset
}

void Timer::start() {
    if (is_running.exchange(true)) {
        throw std::logic_error(
            "Timer::start() called while timer is already running. Call stop() first.");
    }
    start_time = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
    if (!is_running.exchange(false)) {
        throw std::logic_error(
            "Timer::stop() called while timer is not running. Call start() first.");
    }
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / 1e6;
    updateStats(duration);
}

void Timer::logStats(const std::string& label) const {
    uint64_t n = count.load();
    if (n == 0) {
        std::cout << "No timing data available\n";
        return;
    }

    double mean = sum.load() / n;
    double min_time = min_val.load();
    double max_time = max_val.load();

    std::cout << std::fixed << std::setprecision(3);
    if (!label.empty())
        std::cout << label << ": ";
    std::cout << "Count=" << n << ", Mean=" << mean << "ms"
              << ", Min=" << min_time << "ms"
              << ", Max=" << max_time << "ms\n";
}

// Getters
uint64_t Timer::getCount() const {
    return count.load();
}

double Timer::getMean() const {
    uint64_t n = count.load();
    return n > 0 ? sum.load() / n : 0.0;
}

double Timer::getMin() const {
    return count.load() > 0 ? min_val.load() : 0.0;
}

double Timer::getMax() const {
    return count.load() > 0 ? max_val.load() : 0.0;
}
