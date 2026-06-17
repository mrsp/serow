
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

#include "DerivativeEstimator.hpp"
#include "Measurement.hpp"
#include "RobotKinematics.hpp"

namespace serow {

/**
 * @class ContactWrenchEstimator
 * @brief Estimates contact wrenches at specified frames using the Generalized Momentum Observer.
 *
 * Implements De Luca & Mattone (2003). The observer integrates a residual signal:
 *
 *   r(t) = K_I * integral{ tau - beta(q, qdot) - r } dt
 *
 * where beta = C(q, qdot) * qdot + g(q). The residual satisfies r ≈ J^T * F_ext
 * asymptotically. Individual contact wrenches are recovered via F = (J^T)^+ * r.
 */
class ContactWrenchEstimator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Constructs the estimator from a kinematic estimator
     * @param kinematic_estimator The kinematic estimator
     * @param contact_frames The contact frames to estimate the wrenches for
     * @param gain    Observer gain K_I (typical range 10–100)
     */
    ContactWrenchEstimator(std::shared_ptr<RobotKinematics> kinematic_estimator,
                           const std::set<std::string>& contact_frames, const double gain,
                           const std::vector<double>& coeffs_joint, const double joint_rate,
                           const bool point_feet = true)
        : kinematic_estimator_(kinematic_estimator),
          contact_frames_(contact_frames),
          point_feet_(point_feet),
          gain_(gain) {
        const int nv = kinematic_estimator->ndofActuated();
        residual_.setZero(nv);
        integral_.setZero(nv);
        cols_per_contact_ = point_feet_ ? 3 : 6;
        A_.setZero(nv, cols_per_contact_ * static_cast<int>(contact_frames.size()));
        p_derivative_estimator_ =
            std::make_unique<DerivativeEstimator>("p Derivative", coeffs_joint, joint_rate, nv);
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
            p_derivative_estimator_->filter(p, Eigen::VectorXd::Ones(qdot.size()), timestamp);
            return;
        }
        const double dt = timestamp - last_timestamp_.value();
        last_timestamp_ = timestamp;
        if (dt <= 0.0) {
            return;
        }

        const Eigen::VectorXd effort = kinematic_estimator_->getJointEfforts();
        const Eigen::VectorXd nle = kinematic_estimator_->getNonlinearEffects();
        const Eigen::VectorXd dp_dt =
            p_derivative_estimator_->filter(p, Eigen::VectorXd::Ones(qdot.size()), timestamp);
        integral_ += (effort - nle - dp_dt - residual_) * dt;
        residual_ = gain_ * integral_;
    }

    std::map<std::string, ForceTorqueMeasurement> contactWrenches() {
        std::map<std::string, ForceTorqueMeasurement> ft;

        A_.setZero();
        for (int i = 0; i < static_cast<int>(contact_frames_.size()); ++i) {
            const std::string& frame = *std::next(contact_frames_.begin(), i);
            const Eigen::MatrixXd J = kinematic_estimator_->geometricJacobian(frame, false);
            if (point_feet_) {
                // Point contact: residual ~= Jv(q)^T * f
                A_.middleCols(cols_per_contact_ * i, cols_per_contact_) = J.topRows(3).transpose();
            } else {
                // Full wrench: residual ~= J(q)^T * wrench
                A_.middleCols(cols_per_contact_ * i, cols_per_contact_) = J.transpose();
            }
        }

        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(A_);
        const Eigen::VectorXd wrench_stack = cod.solve(residual_);

        int offset = 0;
        for (const std::string& frame : contact_frames_) {
            ft[frame].force = -wrench_stack.segment<3>(offset);
            ft[frame].force.z() = std::max(0.0, ft[frame].force.z());
            if (!point_feet_) {
                ft[frame].torque = -wrench_stack.segment<3>(offset + 3);
            }
            offset += cols_per_contact_;
            ft[frame].timestamp = last_timestamp_.value();
        }

        return ft;
    }

    /**
     * @brief Resets the observer state
     */
    void reset() {
        last_timestamp_.reset();
        residual_.setZero();
        integral_.setZero();
        A_.setZero();
        p_derivative_estimator_->reset();
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
    Eigen::MatrixXd A_;
    std::unique_ptr<DerivativeEstimator> p_derivative_estimator_;  // derivative of the momentum
};

}  // namespace serow
