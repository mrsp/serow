/**
 * @file MovingMedianFilter.hpp
 * @brief Implements a moving median filter for smoothing data streams
 * @author Stylianos Piperakis
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

/**
 * @class MovingMedianFilter
 * @brief Implements a moving median filter for smoothing data streams
 */
class MovingMedianFilter {
   private:
    size_t window_size_;            ///< Size of the sliding window for median calculation
    std::deque<double> window_buffer_;  ///< Buffer to store the current window of values

   public:
    /**
     * @brief Constructs a MovingMedianFilter object with a specified window size
     * @param window_size Size of the sliding window for median calculation
     */
    explicit MovingMedianFilter(size_t window_size) : window_size_(window_size) {}

    /**
     * @brief Applies the moving median filter to a new data point
     * @param x New data point to filter
     * @return Filtered value (median of the current window)
     */
    double filter(double x) {
        // Maintain the window size by removing the oldest element if necessary
        if (window_buffer_.size() == window_size_) {
            window_buffer_.pop_front();
        }

        // Add the new data point to the end of the window buffer
        window_buffer_.push_back(x);

        // Trivial case: If only one element in the buffer, return it as-is
        if (window_buffer_.size() == 1) {
            return x;
        }

        // Copy and sort the window buffer
        auto sorted_window = window_buffer_;
        std::sort(sorted_window.begin(), sorted_window.end());

        // Calculate median based on the size of the window buffer
        if (sorted_window.size() % 2 == 0) {
            // Even number of elements: return the mean of the two middle elements
            return (sorted_window[sorted_window.size() / 2 - 1] +
                    sorted_window[sorted_window.size() / 2]) / 2;
        } else {
            // Odd number of elements: return the middle element
            return sorted_window[sorted_window.size() / 2];
        }
    }
};

}  // namespace serow
