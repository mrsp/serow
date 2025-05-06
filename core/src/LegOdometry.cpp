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
#include "LegOdometry.hpp"

#include <iostream>

#include "lie.hpp"

namespace serow {

LegOdometry::LegOdometry(
    const Eigen::Vector3d& base_position,
    std::map<std::string, Eigen::Vector3d> feet_position,
    std::map<std::string, Eigen::Quaterniond> feet_orientation, double mass, double alpha1,
    double alpha3, double freq, double g, double eps,
    std::optional<std::map<std::string, Eigen::Vector3d>> force_torque_offset) {
    params_.freq = freq;
    params_.mass = mass;
    params_.alpha1 = alpha1;
    params_.alpha3 = alpha3;
    params_.Tm = 1.0 / freq;
    params_.Tm2 = params_.Tm * params_.Tm;
    params_.Tm3 = (mass * mass * g * g) * params_.Tm2;
    params_.g = g;
    params_.eps = eps;
    params_.num_leg_ee = feet_position.size();
    for (const auto& [key, value] : feet_position) {
        pivots_[key] = Eigen::Vector3d::Zero();
    }
    feet_position_prev_ = std::move(feet_position);
    feet_orientation_prev_ = std::move(feet_orientation);
    if (force_torque_offset.has_value()) {
        force_torque_offset_ = std::move(force_torque_offset);
    }
    base_position_ = base_position;
    base_position_prev_ = base_position;
}

const Eigen::Vector3d& LegOdometry::getBasePosition() const {
    return base_position_;
}

const Eigen::Vector3d& LegOdometry::getBaseLinearVelocity() const {
    return base_linear_velocity_;
}

const std::map<std::string, Eigen::Vector3d> LegOdometry::getContactPositions() const {
    return contact_positions_;
}

const std::map<std::string, Eigen::Quaterniond> LegOdometry::getContactOrientations() const {
    return contact_orientations_;
}

void LegOdometry::computeIMP(const std::string& frame, const Eigen::Matrix3d& R,
                             const Eigen::Vector3d& angular_velocity,
                             const Eigen::Vector3d& linear_velocity, const Eigen::Vector3d& force,
                             std::optional<Eigen::Vector3d> torque) {
    const Eigen::Matrix3d force_skew = lie::so3::wedge(force);
    const Eigen::Matrix3d omega_skew = lie::so3::wedge(R.transpose() * angular_velocity);

    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    A.noalias() = 1.0 / params_.Tm2 * Eigen::Matrix3d::Identity();
    A.noalias() -= params_.alpha1 * omega_skew * omega_skew;
    b.noalias() = 1.0 / params_.Tm2 * pivots_.at(frame);
    b.noalias() += params_.alpha1 * omega_skew * R.transpose() * linear_velocity;

    if (params_.alpha3 > 0 && torque.has_value() && force_torque_offset_.has_value()) {
        A.noalias() -= params_.alpha3 / params_.Tm3 * force_skew * force_skew;
        b.noalias() += params_.alpha3 / params_.Tm3 *
            (force_skew * torque.value() -
             force_skew * force_skew * force_torque_offset_->at(frame));
    }
    pivots_.at(frame).noalias() = A.inverse() * b;
}

void LegOdometry::estimate(
    const Eigen::Quaterniond& base_orientation, const Eigen::Vector3d& base_angular_velocity,
    const std::map<std::string, Eigen::Quaterniond>& base_to_foot_orientations,
    const std::map<std::string, Eigen::Vector3d>& base_to_foot_positions,
    const std::map<std::string, Eigen::Vector3d>& base_to_foot_linear_velocities,
    const std::map<std::string, Eigen::Vector3d>& base_to_foot_angular_velocities,
    const std::map<std::string, Eigen::Vector3d>& contact_forces,
    const std::map<std::string, double>& contact_probabilities,
    std::optional<std::map<std::string, Eigen::Vector3d>> contact_torques) {
    const Eigen::Matrix3d Rwb = base_orientation.toRotationMatrix();

    double den = params_.eps * params_.num_leg_ee;
    std::map<std::string, double> force_weights;
    for (const auto& [key, value] : contact_probabilities) {
        den += value;
    }

    for (const auto& [key, value] : contact_probabilities) {
        force_weights[key] = std::clamp((value + params_.eps) / den, 0.0, 1.0);
    }

    if (!is_initialized) {
        base_linear_velocity_ = Eigen::Vector3d::Zero();
        for (const auto& [key, value] : base_to_foot_positions) {
            base_linear_velocity_ += force_weights.at(key) *
                (-lie::so3::wedge(base_angular_velocity) * Rwb * base_to_foot_positions.at(key) -
                 Rwb * base_to_foot_linear_velocities.at(key));
        }
    } else {
        base_linear_velocity_ = (base_position_ - base_position_prev_) * params_.freq;
    }

    for (const auto& [key, value] : base_to_foot_orientations) {
        const Eigen::Vector3d foot_angular_velocity =
            base_angular_velocity + Rwb * base_to_foot_angular_velocities.at(key);

        const Eigen::Vector3d foot_linear_velocity = base_linear_velocity_ +
            lie::so3::wedge(base_angular_velocity) * Rwb * base_to_foot_positions.at(key) +
            Rwb * base_to_foot_linear_velocities.at(key);

        if (contact_torques.has_value()) {
            computeIMP(key, Rwb * value.toRotationMatrix(), foot_angular_velocity,
                       foot_linear_velocity, contact_forces.at(key),
                       contact_torques.value().at(key));
        } else {
            computeIMP(key, Rwb * value.toRotationMatrix(), foot_angular_velocity,
                       foot_linear_velocity, contact_forces.at(key));
        }
    }

    for (const auto& [key, value] : pivots_) {
        feet_position_prev_.at(key) +=
            -Rwb * base_to_foot_orientations.at(key).toRotationMatrix() * value +
            feet_orientation_prev_.at(key).toRotationMatrix() * value;
    }

    std::map<std::string, Eigen::Vector3d> contact_contributions;
    for (const auto& [key, value] : base_to_foot_positions) {
        contact_contributions[key] = feet_position_prev_.at(key) - Rwb * value;
    }

    base_position_prev_ = base_position_;
    base_position_ = Eigen::Vector3d::Zero();
    for (const auto& [key, value] : force_weights) {
        base_position_ += value * contact_contributions.at(key);
    }

    for (const auto& [key, value] : contact_contributions) {
        feet_position_prev_.at(key) += base_position_ - value;
        feet_orientation_prev_.at(key) =
            Eigen::Quaterniond(Rwb * base_to_foot_orientations.at(key).toRotationMatrix());
    }

    for (const auto& [key, value] : pivots_) {
        contact_positions_[key] = base_to_foot_positions.at(key) +
            base_to_foot_orientations.at(key).toRotationMatrix() * value;
        contact_orientations_[key] = base_to_foot_orientations.at(key);
    }

    if (!is_initialized) {
        is_initialized = true;
    }
}

}  // namespace serow
