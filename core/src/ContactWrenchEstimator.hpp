
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

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <string>
#include <vector>

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
     * @brief Constructs the estimator from a Pinocchio model
     * @param model   Pinocchio model shared with RobotKinematics
     * @param gain    Observer gain K_I (typical range 10–100)
     */
    ContactWrenchEstimator(const pinocchio::Model& model, double gain)
        : model_(model), data_(model), gain_(gain) {
        const int nv = model_.nv;
        residual_.setZero(nv);
        integral_.setZero(nv);
        p_prev_.setZero(nv);
    }

    ~ContactWrenchEstimator() = default;

    /**
     * @brief Integrates the observer one step forward
     * @param q       Joint positions (size nq)
     * @param qdot    Joint velocities (size nv)
     * @param effort  Joint efforts (size nv)
     * @param dt      Elapsed time since last call in seconds
     */
    void update(const Eigen::VectorXd& q, const Eigen::VectorXd& qdot,
                const Eigen::VectorXd& effort, double dt) {
        pinocchio::crba(model_, data_, q);
        data_.M.triangularView<Eigen::StrictlyLower>() =
            data_.M.transpose().triangularView<Eigen::StrictlyLower>();

        const Eigen::VectorXd beta =
            pinocchio::rnea(model_, data_, q, qdot, Eigen::VectorXd::Zero(model_.nv));

        const Eigen::VectorXd p = data_.M * qdot;

        if (!initialized_) {
            p_prev_      = p;
            initialized_ = true;
            return;
        }

        const Eigen::VectorXd dp_dt = (p - p_prev_) / dt;
        integral_ += (effort - beta - dp_dt - residual_) * dt;
        residual_  = gain_ * integral_;
        p_prev_    = p;
    }

    /**
     * @brief Computes the contact wrench at a frame from the current observer residual
     * @param q             Joint positions (size nq) — required to compute the Jacobian
     * @param frame_id      Pinocchio frame index of the contact frame
     * @param in_body_frame Express the wrench in the body frame when true, world frame when false
     * @return [fx, fy, fz, tx, ty, tz] contact wrench
     */
    Eigen::Matrix<double, 6, 1> wrench(const Eigen::VectorXd& q, 
                                       pinocchio::FrameIndex frame_id,
                                       bool in_body_frame = true) const {
        if (!initialized_) {
            return Eigen::Matrix<double, 6, 1>::Zero();
        }

        // 1. Compute Jacobians and Frame Placements based on current 'q'
        pinocchio::computeJointJacobians(model_, data_, q);
        pinocchio::updateFramePlacements(model_, data_);

        // 2. Now it is safe to extract the Jacobian
        pinocchio::Data::Matrix6x J = pinocchio::Data::Matrix6x::Zero(6, model_.nv);
        pinocchio::getFrameJacobian(model_, data_, frame_id, pinocchio::LOCAL, J);

        // 3. Rotate to body frame if requested (data_.oMf is now populated)
        if (in_body_frame) {
            const Eigen::Matrix3d& R = data_.oMf[frame_id].rotation();
            J.topRows(3)    = R * J.topRows(3);
            J.bottomRows(3) = R * J.bottomRows(3);
        }

        // 4. Solve for the wrench
        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(J.transpose());
        return cod.solve(residual_);
    }
    


    /**
     * @brief Computes point-contact forces for several contact frames simultaneously.
     *
     * The generalized momentum residual contains the combined contribution of all
     * external contacts:
     *
     *   residual ~= sum_i J_i(q)^T f_i
     *
     * Solving one foot at a time makes every foot try to explain the full residual.
     * For point feet, build the stacked system
     *
     *   [J_1^T J_2^T ... J_N^T] [f_1 ... f_N]^T = residual
     *
     * and solve it once. Forces are returned in each contact frame's LOCAL frame,
     * which matches the convention expected by Serow::runContactEstimator before
     * it applies R_foot_to_force and the foot/base transforms.
     */
    std::vector<Eigen::Vector3d> contactForces(
        const Eigen::VectorXd& q,
        const std::vector<pinocchio::FrameIndex>& frame_ids) const {
        std::vector<Eigen::Vector3d> output(frame_ids.size(), Eigen::Vector3d::Zero());

        if (!initialized_ || frame_ids.empty()) {
            return output;
        }

        pinocchio::computeJointJacobians(model_, data_, q);
        pinocchio::updateFramePlacements(model_, data_);

        Eigen::MatrixXd A(model_.nv, 3 * static_cast<int>(frame_ids.size()));
        A.setZero();

        for (size_t i = 0; i < frame_ids.size(); ++i) {
            pinocchio::Data::Matrix6x J = pinocchio::Data::Matrix6x::Zero(6, model_.nv);
            pinocchio::getFrameJacobian(model_, data_, frame_ids[i], pinocchio::LOCAL, J);

            // Point contact: generalized external torque contribution is Jv^T * f.
            A.middleCols(3 * static_cast<int>(i), 3) = J.topRows(3).transpose();
        }

        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(A);
        const Eigen::VectorXd f_stack = cod.solve(residual_);

        for (size_t i = 0; i < frame_ids.size(); ++i) {
            output[i] = f_stack.segment<3>(3 * static_cast<int>(i));
        }

        return output;
    }

    /**
     * @brief Resets the observer state
     *
     * Call after liftoff, re-initialization, or prolonged swing to prevent integral drift.
     */
    void reset() {
        residual_.setZero();
        integral_.setZero();
        p_prev_.setZero();
        initialized_ = false;
    }

    /**
     * @brief Sets the observer gain at runtime
     * @param gain  New K_I value
     */
    void setGain(double gain) {
        gain_ = gain;
    }

    /**
     * @brief Returns the raw observer residual r ≈ J^T * F_ext (size nv)
     */
    const Eigen::VectorXd& residual() const {
        return residual_;
    }

private:
    pinocchio::Model model_;
    mutable pinocchio::Data data_;

    double          gain_;
    Eigen::VectorXd residual_;
    Eigen::VectorXd integral_;
    Eigen::VectorXd p_prev_;
    bool            initialized_{false};
};

}  // namespace serow
