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

#include <memory>
#include <string>

#include "MovingMedianFilter.hpp"

namespace serow {

class ContactDetector {
   public:
    /// @brief default constructor
    ContactDetector() = default;

    /// @brief Initializes the contact estimation with a Schmitt-Trigger (ST) detector
    /// @param contact_frame contact frame name e.g. "l_foot_frame"
    /// @param high_threshold vertical ground reaction high force threshold of the ST detector in (N)
    /// @param low_threshold vertical ground reaction low force threshold of the ST detector in (N) 
    /// @param mass mass of the robot (kg)
    /// @param g gravity constant (m/s^2)
    /// @param median_window rolling median filter buffer size, used for outlier detection
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

    /// @brief Applies a digital Schmitt-Trigger detector for binary contact status estimation e.g. contact or no contact
    /// @param contact_force vertical ground reaction force at the contact_frame in world coordinates
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

    /// @brief returns the estimated contact status
    /// @return The estimated contact status (0 or 1)
    int getContactStatus() { return contact_status_; };

    /// @brief returns the filtered with a rolling median filter vertical ground reaction force in world coordinates
    /// @return filtered vertical ground reaction force (N)
    double getContactForce() { return contact_force_; };

    /// @brief returns the contact frame name where detection is done
    /// @return the name of the contact frame e.g. "l_foot_frame"
    std::string getContactFrame() { return contact_frame_; };

   private:
    /// rolling median filter
    std::unique_ptr<MovingMedianFilter> mdf_;
    /// estimated contact status (0 or 1)
    int contact_status_{};
    /// filtered with a rolling median filter vertical ground reaction force in world coordinates (N)
    double contact_force_{};
    /// contact frame name where detection is done e.g. "l_foot_frame"
    std::string contact_frame_;
    /// vertical ground reaction high force threshold of the ST detector in (N)
    double high_threshold_{};
    /// vertical ground reaction low force threshold of the ST detector in (N)
    double low_threshold_{};
    /// mass of the robot (kg)
    double mass_{};
    /// gravity constant (m/s^2)
    double g_{};
};

}  // namespace serow
