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
#include <unordered_map>
#include <unordered_set>
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
     * @param lambda Regularization parameter
     * @param point_feet Whether the feet are point contacts or not
     * @param type The type of pseudo-inverse to use ("llt" or "cod")
     * @param mu The Tikhonov regularization parameter only applies to the "llt" type
     */
    ContactWrenchEstimator(std::shared_ptr<RobotKinematics> kinematic_estimator,
                           const std::set<std::string>& contact_frames, const double gain,
                           const double lambda, const bool point_feet = true,
                           const std::string& type = "llt", const double mu = 1e-6)
        : kinematic_estimator_(kinematic_estimator),
          contact_frames_(contact_frames),
          gain_(gain),
          lambda_(lambda),
          point_feet_(point_feet),
          type_(type),
          mu_(mu) {
        // GMO runs on actuated DoF only; floating-base rows have zero actuator torque.
        n_actuated_ = kinematic_estimator->ndofActuated();
        residual_.setZero(n_actuated_);
        integral_.setZero(n_actuated_);
        cols_per_contact_ = point_feet_ ? 3 : 6;

        // Construct all possible contact cases.
        // NOTE: mask ranges densely over [1, 2^n - 1], so every downstream container keyed by
        // mask is a plain vector indexed by mask.
        std::vector<std::string> frames(contact_frames_.begin(), contact_frames_.end());
        const int n = static_cast<int>(frames.size());
        const int num_masks = (1 << n);

        contact_cases_.resize(num_masks);  // index 0 unused (empty mask never queried)
        A_.resize(num_masks);
        cost_offset_.resize(num_masks, 0.0);

        for (int mask = 1; mask < num_masks; ++mask) {
            std::vector<std::string> active;
            active.reserve(n);
            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {
                    active.push_back(frames[i]);
                }
            }
            const int num_contacts = static_cast<int>(active.size());
            contact_cases_[mask] = std::move(active);

            // Allocate the A matrix for this case once, up front.
            A_[mask] = Eigen::MatrixXd::Zero(n_actuated_, cols_per_contact_ * num_contacts);

            // Precompute the regularization term. It depends only on the static structure of
            // contact_cases_ (mask -> num_contacts).
            cost_offset_[mask] = lambda_ * static_cast<double>(num_contacts) * cols_per_contact_;
        }

        // Reusable scratch storage, sized once so contactWrenches() never reallocates.
        wrenches_.resize(num_masks);
        jacobian_cache_.reserve(contact_frames_.size());
        active_set_scratch_.reserve(frames.size());
    }

    ~ContactWrenchEstimator() = default;

    /**
     * @brief Integrates the observer one step forward
     * @param dt Elapsed time since last call in seconds
     */
    void update(const double timestamp) {
        // 1. Fetch full-space configurations (nv elements)
        const Eigen::VectorXd qdot = kinematic_estimator_->getJointVelocities();
        kinematic_estimator_->computeDynamicTerms();

        const Eigen::MatrixXd M = kinematic_estimator_->getMassMatrix();
        // Actuated momentum mapping: row-slice matching actuated joint space
        const Eigen::VectorXd p_actuated = M.bottomRows(n_actuated_) * qdot;

        if (!last_timestamp_.has_value()) {
            last_timestamp_ = timestamp;
            integral_ = p_actuated;
            return;
        }
        const double dt = timestamp - last_timestamp_.value();
        last_timestamp_ = timestamp;
        if (dt <= 0.0) {
            return;
        }

        // 2. Extract joint-space effort and gravity vectors (n_actuated elements)
        const Eigen::VectorXd effort = kinematic_estimator_->getJointEfforts().tail(n_actuated_);
        const Eigen::VectorXd gravity = kinematic_estimator_->getGravityEffects().tail(n_actuated_);

        // 3. Extract the coriolis effects (n_actuated elements)
        const Eigen::MatrixXd C = kinematic_estimator_->getCoriolisMatrix();
        const Eigen::VectorXd C_transpose_qdot = (C.transpose() * qdot).tail(n_actuated_);

        // 4. GMO Integration step
        const Eigen::VectorXd integral_dot = effort + C_transpose_qdot - gravity + residual_;
        integral_ += integral_dot * dt;

        // 5. Compute residual and update observer
        residual_ = gain_ * (p_actuated - integral_);
    }

    std::map<std::string, ForceTorqueMeasurement> contactWrenches() {
        std::map<std::string, ForceTorqueMeasurement> ft;

        // Jacobian cache: compute each unique frame's Jacobian exactly once.
        jacobian_cache_.clear();
        for (const auto& frame : contact_frames_) {
            jacobian_cache_.emplace(
                frame,
                kinematic_estimator_->geometricJacobian(frame, false).topRows(cols_per_contact_));
        }

        // Build stacked Jacobian for each contact case.
        for (int mask = 1; mask < static_cast<int>(contact_cases_.size()); ++mask) {
            const auto& frames = contact_cases_[mask];
            Eigen::MatrixXd& A = A_[mask];
            A.setZero();
            int i = 0;
            for (const std::string& frame : frames) {
                A.middleCols(cols_per_contact_ * i, cols_per_contact_) =
                    jacobian_cache_.at(frame).transpose();
                ++i;
            }
        }

        // Solve for contact wrenches and find case with minimum reconstruction error.
        int optimal_mask = -1;
        double min_cost = std::numeric_limits<double>::max();
        const double residual_norm = residual_.squaredNorm();
        const double inv_residual_norm = 1.0 / (residual_norm + 1e-9);
        for (int mask = 1; mask < static_cast<int>(contact_cases_.size()); ++mask) {
            const Eigen::MatrixXd& A = A_[mask];
            Eigen::VectorXd& wrench = wrenches_[mask];
            if (type_ == "llt") {
                // Right pseudo-inverse via normal equations on A*A^T (size n_actuated_ x
                // n_actuated_, independent of contact count) instead of QR-based COD on the full
                // wide A.
                Eigen::MatrixXd AAT = A * A.transpose();
                AAT.diagonal().array() +=
                    mu_;  // Tikhonov term for numerical safety near rank deficiency
                Eigen::LLT<Eigen::MatrixXd> llt(AAT);
                wrench = A.transpose() * llt.solve(residual_);
            } else {
                // Use Complete Orthogonal Decomposition to solve the system of equations.
                // More expensive than LLT but more numerically stable.
                Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(A);
                wrench = cod.solve(residual_);
            }
            const double cost =
                (A * wrench - residual_).squaredNorm() * inv_residual_norm + cost_offset_[mask];
            if (cost < min_cost) {
                min_cost = cost;
                optimal_mask = mask;
            }
        }
        if (optimal_mask < 0)
            return ft;

        // Extract only from the optimal case.
        const auto& optimal_frames = contact_cases_[optimal_mask];
        const Eigen::VectorXd& wrench = wrenches_[optimal_mask];

        // Active feet — extract from solved wrench.
        active_set_scratch_.clear();
        int index = 0;
        for (const std::string& frame : optimal_frames) {
            ft[frame].force = wrench.segment<3>(cols_per_contact_ * index);
            if (!point_feet_) {
                ft[frame].torque = wrench.segment<3>(cols_per_contact_ * index + 3);
            }
            active_set_scratch_.insert(frame);  // O(1) membership check for the loop below
            ++index;
        }

        // Inactive feet — zero out explicitly for the caller.
        for (const std::string& frame : contact_frames_) {
            if (active_set_scratch_.find(frame) == active_set_scratch_.end()) {
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
        residual_.setZero(n_actuated_);
        integral_.setZero(n_actuated_);
        for (Eigen::MatrixXd& A : A_) {
            A.setZero();
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
    int cols_per_contact_{3};
    std::optional<double> last_timestamp_;
    Eigen::VectorXd residual_;
    Eigen::VectorXd integral_;

    /// Stacked Jacobian matrix per contact case, indexed directly by mask (dense range
    /// [1, 2^n - 1]). Avoids reallocation of memory.
    std::vector<Eigen::MatrixXd> A_;
    /// Frames active in each contact case, indexed by mask.
    std::vector<std::vector<std::string>> contact_cases_;
    /// Precomputed lambda_ * num_contacts * cols_per_contact_ per mask (static, call-invariant).
    std::vector<double> cost_offset_;
    /// Solved wrench per mask, reused across calls.
    std::vector<Eigen::VectorXd> wrenches_;
    /// One geometricJacobian() call per unique frame per contactWrenches() call.
    std::unordered_map<std::string, Eigen::MatrixXd> jacobian_cache_;
    /// O(1) active-frame membership test when zeroing out inactive feet.
    std::unordered_set<std::string> active_set_scratch_;

    int n_actuated_;
    double gain_{100.0};
    double lambda_{1e-2};
    bool point_feet_{true};
    std::string type_{"llt"};
    double mu_{1e-6};
};

}  // namespace serow
