
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
 * @file ContactWrenchEstimator.hpp
 * @brief Estimates contact wrenches from joint efforts using the Generalized Momentum Observer
 * @author Stylianos Piperakis
 */
#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "ButterworthLPF.hpp"
#include "Measurement.hpp"
#include "RobotKinematics.hpp"

namespace serow {

/**
 * @class ContactWrenchEstimator
 * @brief Estimates contact wrenches at specified frames using the Generalized Momentum Observer.
 */
class ContactWrenchEstimator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Constructs the estimator from a kinematic estimator
     * @param kinematic_estimator The kinematic estimator
     * @param contact_frames The contact frames to estimate the wrenches for
     * @param gain    Observer gain K_I (typical range 10–100)
     * @param rate    Rate of the joint data (e.g. 100 Hz)
     * @param cutoff_frequency Cutoff frequency of the residual LPF (e.g. 10 Hz)
     * @param lambda Regularization parameter
     * @param point_feet Whether the feet are point contacts or not
     */
    ContactWrenchEstimator(std::shared_ptr<RobotKinematics> kinematic_estimator,
                           const std::set<std::string>& contact_frames, const double gain,
                           const double rate, const double cutoff_frequency, const double lambda,
                           const bool point_feet = true)
        : kinematic_estimator_(kinematic_estimator),
          contact_frames_(contact_frames),
          point_feet_(point_feet),
          gain_(gain),
          lambda_(lambda) {
        nv_ = kinematic_estimator->ndofActuated();
        residual_.setZero(nv_);
        integral_.setZero(nv_);
        cols_per_contact_ = point_feet_ ? 3 : 6;
        p_prev_.setZero(nv_);
        lpf_.resize(nv_);
        for (int i = 0; i < nv_; ++i) {
            lpf_[i] = std::make_unique<ButterworthLPF>(
                std::string("Residual LPF ") + std::to_string(i), rate, cutoff_frequency, false);
        }

        // Construct all possible contact cases
        std::vector<std::string> frames(contact_frames_.begin(), contact_frames_.end());
        const int n = static_cast<int>(frames.size());
        for (int mask = 1; mask < (1 << n); ++mask) {
            std::set<std::string> active;
            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {
                    active.insert(frames[i]);
                }
            }
            contact_cases_[mask] = std::move(active);
        }

        // Allocate all possible A matrices for the contact cases
        for (const auto& [mask, frames] : contact_cases_) {
            const int n = static_cast<int>(frames.size());
            A_[mask] = Eigen::MatrixXd::Zero(nv_, cols_per_contact_ * n);
        }
    }

    ~ContactWrenchEstimator() = default;

    /**
     * @brief Integrates the observer one step forward
     * @param dt Elapsed time since last call in seconds
     */
    void update(const double timestamp) {
        const Eigen::VectorXd qdot = kinematic_estimator_->getJointVelocities();
        kinematic_estimator_->computeDynamicTerms();
        const Eigen::MatrixXd M = kinematic_estimator_->getMassMatrix();
        const Eigen::VectorXd p = M * qdot;
        if (!last_timestamp_.has_value()) {
            last_timestamp_ = timestamp;
            p_prev_ = p;
            return;
        }
        const double dt = timestamp - last_timestamp_.value();
        last_timestamp_ = timestamp;
        if (dt <= 0.0) {
            p_prev_ = p;
            return;
        }

        const Eigen::VectorXd effort = kinematic_estimator_->getJointEfforts();
        const Eigen::VectorXd nle = kinematic_estimator_->getNonlinearEffects();
        integral_ += (effort - nle - residual_) * dt;
        residual_ = gain_ * (integral_ - (p - p_prev_));
        for (int i = 0; i < nv_; ++i) {
            residual_[i] = lpf_[i]->filter(residual_[i]);
        }
        p_prev_ = p;
    }

    std::map<std::string, ForceTorqueMeasurement> contactWrenches() {
        std::map<std::string, ForceTorqueMeasurement> ft;

        // Build stacked Jacobian for each contact case
        for (const auto& [mask, frames] : contact_cases_) {
            const int n = static_cast<int>(frames.size());
            A_[mask].setZero();
            for (int i = 0; i < n; ++i) {
                const std::string& frame = *std::next(frames.begin(), i);
                const Eigen::MatrixXd J = kinematic_estimator_->geometricJacobian(frame, false);
                A_[mask].middleCols(cols_per_contact_ * i, cols_per_contact_) = J.transpose();
            }
        }

        // Solve for contact wrenches and find case with minimum reconstruction error
        int optimal_mask = -1;
        double min_cost = std::numeric_limits<double>::max();
        std::map<int, Eigen::VectorXd> wrenches;

        for (const auto& [mask, frames] : contact_cases_) {
            const int num_contacts = static_cast<int>(frames.size());
            Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(A_[mask]);
            wrenches[mask] = cod.solve(residual_);

            // Compute costs
            const double residual_norm = residual_.squaredNorm();
            const double cost =
                (A_[mask] * wrenches[mask] - residual_).squaredNorm() / (residual_norm + 1e-9) +
                lambda_ * num_contacts * cols_per_contact_;
            if (cost < min_cost) {
                min_cost = cost;
                optimal_mask = mask;
            }
        }

        if (optimal_mask < 0)
            return ft;

        // Extract only from the optimal case
        const auto& optimal_frames = contact_cases_[optimal_mask];
        Eigen::VectorXd& wrench = wrenches[optimal_mask];

        // Active feet — extract from solved wrench
        int index = 0;
        for (const std::string& frame : optimal_frames) {
            ft[frame].force = wrench.segment<3>(cols_per_contact_ * index);
            if (!point_feet_) {
                ft[frame].torque = wrench.segment<3>(cols_per_contact_ * index + 3);
            }
            ++index;
        }

        // Inactive feet — zero out explicitly for the caller
        for (const std::string& frame : contact_frames_) {
            if (optimal_frames.count(frame) == 0) {
                ft[frame].force = Eigen::Vector3d::Zero();
                if (!point_feet_) {
                    ft[frame].torque = Eigen::Vector3d::Zero();
                }
            }
        }

        return ft;
    }

    /**
     * @brief Resets the observer state
     */
    void reset() {
        last_timestamp_.reset();
        residual_.setZero(nv_);
        integral_.setZero(nv_);
        for (auto& [mask, A] : A_) {
            A.setZero();
        }
        p_prev_.setZero(nv_);
        for (int i = 0; i < nv_; ++i) {
            lpf_[i]->reset();
        }
    }

    /**
     * @brief Sets the observer gain at runtime
     * @param gain  New K_I value
     */
    void setGain(const double gain) {
        gain_ = gain;
    }

private:
    std::shared_ptr<RobotKinematics> kinematic_estimator_;
    std::set<std::string> contact_frames_;
    bool point_feet_{true};
    int cols_per_contact_{3};
    double gain_{0.0};
    std::optional<double> last_timestamp_;
    Eigen::VectorXd residual_;
    Eigen::VectorXd integral_;
    /// Stacked Jacobian matrix for the contact frames. Avoids reallocation of memory.
    std::map<int, Eigen::MatrixXd> A_;
    std::map<int, std::set<std::string>> contact_cases_;
    Eigen::VectorXd p_prev_;
    int nv_;
    std::vector<std::unique_ptr<ButterworthLPF>> lpf_;
    double lambda_{5e-3};
};

}  // namespace serow
