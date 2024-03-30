/**
 * Copyright (C) 2024 Stylianos Piperakis, Ownage Dynamics L.P.
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

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif

#include <deque>
#include <iostream>

namespace serow {

class MovingMedianFilter {
   private:
    size_t window_size_{};
    std::deque<double> window_buffer_;

   public:
    MovingMedianFilter(size_t window_size) { window_size_ = window_size; }

    double filter(double x) {
        if (window_buffer_.size() == window_size_) {
            window_buffer_.pop_front();
        }
        window_buffer_.push_back(x);

        // Trivial case
        if (window_buffer_.size() == 1) {
            return x;
        }

        // sort the buffer
        auto window_buffer = window_buffer_;
        std::sort(window_buffer.begin(), window_buffer.end());
        if (window_buffer.size() % 2 == 0) {
            // Mean case
            return (window_buffer[window_buffer.size() / 2 - 1] +
                    window_buffer[window_buffer.size() / 2]) /
                   2;
        } else {
            // Median case
            return window_buffer[window_buffer.size() / 2];
        }
    }
};

}  // namespace serow
