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
/**
 * @file MovingMedianFilter.hpp
 * @brief Implements a moving median filter for smoothing data streams
 * @author Stylianos Piperakis
 */

#pragma once

#include <deque>
#include <iostream>
#include <set>

namespace serow {

/**
 * @class MovingMedianFilter
 * @brief Implements a moving median filter for smoothing data streams using efficient two-heap
 * approach
 */
class MovingMedianFilter {
private:
    size_t window_size_;                ///< Size of the sliding window for median calculation
    std::deque<double> window_buffer_;  ///< Buffer to store the current window of values
    std::multiset<double> left_heap_;   ///< Max heap (largest elements) - stores smaller half
    std::multiset<double> right_heap_;  ///< Min heap (smallest elements) - stores larger half

public:
    /// @brief Default constructor
    MovingMedianFilter() = default;

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
        // Add new value to window buffer
        window_buffer_.push_back(x);

        // Add to appropriate heap
        if (left_heap_.empty() || x <= *left_heap_.rbegin()) {
            left_heap_.insert(x);
        } else {
            right_heap_.insert(x);
        }

        // Balance heaps
        if (left_heap_.size() > right_heap_.size() + 1) {
            right_heap_.insert(*left_heap_.rbegin());
            left_heap_.erase(std::prev(left_heap_.end()));
        } else if (right_heap_.size() > left_heap_.size()) {
            left_heap_.insert(*right_heap_.begin());
            right_heap_.erase(right_heap_.begin());
        }

        // Remove oldest element if window is full
        if (window_buffer_.size() > window_size_) {
            double old_value = window_buffer_.front();
            window_buffer_.pop_front();

            // Remove from appropriate heap
            auto left_it = left_heap_.find(old_value);
            if (left_it != left_heap_.end()) {
                left_heap_.erase(left_it);
            } else {
                auto right_it = right_heap_.find(old_value);
                if (right_it != right_heap_.end()) {
                    right_heap_.erase(right_it);
                }
            }

            // Rebalance after removal
            if (left_heap_.size() > right_heap_.size() + 1) {
                right_heap_.insert(*left_heap_.rbegin());
                left_heap_.erase(std::prev(left_heap_.end()));
            } else if (right_heap_.size() > left_heap_.size()) {
                left_heap_.insert(*right_heap_.begin());
                right_heap_.erase(right_heap_.begin());
            }
        }

        // Return median
        if (left_heap_.empty()) {
            return 0.0;
        }
        return *left_heap_.rbegin();
    }

    /**
     * @brief Gets the current median without adding a new value
     * @return Current median value
     */
    double getMedian() const {
        if (left_heap_.empty()) {
            return 0.0;
        }
        return *left_heap_.rbegin();
    }

    /**
     * @brief Gets the current window size
     * @return Number of elements in the current window
     */
    size_t size() const {
        return window_buffer_.size();
    }

    /**
     * @brief Gets the maximum window size
     * @return Maximum window size
     */
    size_t maxSize() const {
        return window_size_;
    }

    /**
     * @brief Gets the current window buffer for external calculations (e.g., MAD)
     * @return Reference to the current window buffer
     */
    const std::deque<double>& getWindow() const {
        return window_buffer_;
    }

    /**
     * @brief Clears all data from the filter
     */
    void clear() {
        window_buffer_.clear();
        left_heap_.clear();
        right_heap_.clear();
    }
};

}  // namespace serow
