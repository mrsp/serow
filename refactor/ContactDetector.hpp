/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#pragma once
#include <string>

#include "Mediator.hpp"

namespace serow {

class ContactDetector {
   public:
    ContactDetector() = default;
    /** @fn ContactDetector(std::string contact_frame, double high_threshold, double low_threshold,
     * double mass, double g, int median_window = 10)
     *   @brief initializes the contact estimation with a Schmitt-Trigger
     */
    ContactDetector(std::string contact_frame, double high_threshold, double low_threshold,
                    double mass, double g, int median_window = 10) {
        contact_status_ = 0;
        contact_force_ = 0.0;
        contact_frame_ = contact_frame;
        high_threshold_ = high_threshold;
        low_threshold_ = low_threshold;
        mass_ = mass;
        g_ = g;
        mdf_ = MediatorNew(median_window);
    }

    ~ContactDetector() { delete mdf_; }
    
    /** @fn  SchmittTrigger(double contact_force)
     *  @brief applies a digital Schmitt-Trigger for contact detection
     *  @param force normal ground reaction force
     */
    void SchmittTrigger(double contact_force) {
        // MediatorInsert(mdf_, cropContactForce(contact_force));
        // contact_force_ = MediatorMedian(mdf_);
        contact_force_ = cropContactForce(contact_force);

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

    /** @fn     double cropContactForce(double force)
     *  @brief  Crops the vertical ground reaction force (GRF) in the margins [0, mass * g]
     *  @param  force GRF
     *  @return  The cropped GRF
     */
    double cropContactForce(double force) const {
        return std::max(0.0, std::min(force, mass_ * g_));
    }

   private:
    Mediator* mdf_;
    int contact_status_{};
    double contact_force_{};
    std::string contact_frame_;
    double high_threshold_{};
    double low_threshold_{};
    double mass_{};
    double g_{};
};

}  // namespace serow
