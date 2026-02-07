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
 * @file ContactDetector.hpp
 * @brief Header file for the ContactDetector class.
 * @details Provides a mechanism to estimate the contact probability of a leg end-effector downstreams based on a filtered vertical ground reaction force.
 */

#pragma once

#include <memory>
#include <string>

#include "MovingMedianFilter.hpp"

namespace serow {

/**
 * @class ContactDetector
 * @brief Class for estimating a filtered vertical ground reaction force.
 */
class ContactDetector {
public:
    /**
     * @brief Default constructor.
     */
    ContactDetector() = default;

    /**
     * @brief Initializes the contact estimator.
     * @param contact_frame Contact frame name e.g., "l_foot_frame".
     * @param mass Mass of the robot in kilograms (kg).
     * @param g Gravity constant in meters per second squared (m/s^2).
     * @param median_window Rolling median filter buffer size, used for outlier detection.
     */
    ContactDetector(std::string contact_frame, double mass, double g, int median_window = 11) {
        contact_force_ = 0.0;
        contact_frame_ = contact_frame;
        mass_ = mass;
        g_ = g;
        mdf_ = std::make_unique<MovingMedianFilter>(median_window);
    }

    /**
     * @brief Applies a rolling median filter for outlier detection in contact force measurement.
     * @param contact_force Vertical ground reaction force at the contact_frame (N).
     */
    void run(const double contact_force) {
        contact_force_ = mdf_->filter(std::clamp(contact_force, 0.0, mass_ * g_));
    }

    /**
     * @brief Returns the filtered vertical ground reaction force.
     * @return Filtered vertical ground reaction force in Newtons (N).
     */
    double getContactForce() {
        return contact_force_;
    }

    /**
     * @brief Returns the contact frame name where filtering is done.
     * @return The name of the contact frame e.g., "l_foot_frame".
     */
    std::string getContactFrame() {
        return contact_frame_;
    }

    /**
     * @brief Sets the state of the contact detector.
     * @param contact_force The vertical ground reaction force at the contact_frame (N).
     */
    void setState(double contact_force) {
        contact_force_ = contact_force;
    }

private:
    std::unique_ptr<MovingMedianFilter> mdf_; /**< Rolling median filter. */
    double contact_force_{};                  /**< Filtered vertical ground reaction force in Newtons (N). */
    std::string contact_frame_;               /**< Contact frame name where detection is done e.g., "l_foot_frame". */
    double mass_{};                           /**< Mass of the robot in kilograms (kg). */
    double g_{};                              /**< Gravity constant in meters per second squared (m/s^2). */
};

}  // namespace serow
