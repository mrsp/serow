/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
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
    int window_size_{};
    std::deque<double> window_buffer_;

   public:
    MovingMedianFilter(int window_size) { window_size_ = window_size; }

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
                    window_buffer[window_buffer.size() / 2]) / 2;
        } else {
            // Median case
            return window_buffer[window_buffer.size() / 2];
        }
    }
};

}  // namespace serow
