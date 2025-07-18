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

/**
 * @file ContactDetector.hpp
 * @brief Header file for the ContactDetector class.
 * @details Provides a mechanism to estimate contact status using a Schmitt-Trigger (ST) detector
 * with optional median filtering. The contact status is determined based on vertical ground
 * reaction force thresholds.
 */

#pragma once

#include <memory>
#include <string>

#include "MovingMedianFilter.hpp"

namespace serow {

/**
 * @class ContactDetector
 * @brief Class for estimating contact status using a Schmitt-Trigger detector.
 */
class ContactDetector {
public:
    /**
     * @brief Default constructor.
     */
    ContactDetector() = default;

    /**
     * @brief Initializes the contact estimation with a Schmitt-Trigger (ST) detector.
     * @param contact_frame Contact frame name e.g., "l_foot_frame".
     * @param high_threshold Vertical ground reaction high force threshold of the ST detector in
     * Newtons (N).
     * @param low_threshold Vertical ground reaction low force threshold of the ST detector in
     * Newtons (N).
     * @param mass Mass of the robot in kilograms (kg).
     * @param g Gravity constant in meters per second squared (m/s^2).
     * @param median_window Rolling median filter buffer size, used for outlier detection.
     */
    ContactDetector(std::string contact_frame, double high_threshold, double low_threshold,
                    double mass, double g, int median_window = 11) {
        contact_status_ = 0;
        contact_force_ = 0.0;
        contact_frame_ = contact_frame;
        high_threshold_ = high_threshold;
        low_threshold_ = low_threshold;
        mass_ = mass;
        g_ = g;
        mdf_ = std::make_unique<MovingMedianFilter>(median_window);
    }

    /**
     * @brief Applies a digital Schmitt-Trigger detector for binary contact status estimation e.g.,
     * contact or no contact.
     * @param contact_force Vertical ground reaction force at the contact_frame in world
     * coordinates.
     */
    void SchmittTrigger(double contact_force) {
        contact_force_ = mdf_->filter(std::clamp(contact_force, 0.0, mass_ * g_));
        if (contact_status_ == 0) {
            if (contact_force_ > high_threshold_) {
                contact_status_ = 1;
            }
        } else {
            if (contact_force_ < low_threshold_) {
                contact_status_ = 0;
            }
        }
    }

    /**
     * @brief Returns the estimated contact status.
     * @return The estimated contact status (0 or 1).
     */
    int getContactStatus() {
        return contact_status_;
    }

    /**
     * @brief Returns the filtered vertical ground reaction force in world coordinates.
     * @return Filtered vertical ground reaction force in Newtons (N).
     */
    double getContactForce() {
        return contact_force_;
    }

    /**
     * @brief Returns the contact frame name where detection is done.
     * @return The name of the contact frame e.g., "l_foot_frame".
     */
    std::string getContactFrame() {
        return contact_frame_;
    }

    /**
     * @brief Sets the state of the contact detector.
     * @param contact_status The estimated contact status (0 or 1).
     * @param contact_force The filtered vertical ground reaction force in world coordinates (N).
     */
    void setState(int contact_status, double contact_force) {
        contact_status_ = contact_status;
        contact_force_ = contact_force;
    }

private:
    std::unique_ptr<MovingMedianFilter> mdf_; /**< Rolling median filter. */
    int contact_status_{};                    /**< Estimated contact status (0 or 1). */
    double
        contact_force_{}; /**< Filtered vertical ground reaction force in world coordinates (N). */
    std::string
        contact_frame_; /**< Contact frame name where detection is done e.g., "l_foot_frame". */
    double high_threshold_{}; /**< Vertical ground reaction high force threshold of the ST detector
                                 in Newtons (N). */
    double low_threshold_{}; /**< Vertical ground reaction low force threshold of the ST detector in
                                Newtons (N). */
    double mass_{};          /**< Mass of the robot in kilograms (kg). */
    double g_{};             /**< Gravity constant in meters per second squared (m/s^2). */
};

}  // namespace serow
