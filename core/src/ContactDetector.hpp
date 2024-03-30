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
    ContactDetector() = default;
    /** @fn ContactDetector(std::string contact_frame, double high_threshold, double low_threshold,
     * double mass, double g, int median_window = 10)
     *   @brief initializes the contact estimation with a Schmitt-Trigger
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

    /** @fn  SchmittTrigger(double contact_force)
     *  @brief applies a digital Schmitt-Trigger for contact detection
     *  @param force normal ground reaction force
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

    int getContactStatus() { return contact_status_; };

    double getContactForce() { return contact_force_; };

    std::string getContactFrame() { return contact_frame_; };

   private:
    std::unique_ptr<MovingMedianFilter> mdf_;
    int contact_status_{};
    double contact_force_{};
    std::string contact_frame_;
    double high_threshold_{};
    double low_threshold_{};
    double mass_{};
    double g_{};
};

}  // namespace serow
