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
 * @brief leg odometry for legged robots based on force/torque or pressure, and encoder measurement
 * @author Stylianos Piperakis
 * @details Estimates the 3D leg odometry of the base and the corresponding relative leg
 * measurements
 */
#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif
#include <optional>
#include <string>
#include <unordered_map>

namespace serow {

class LegOdometry {
    struct Params {
        double Tm{};
        double Tm2{};
        double Tm3{};
        double mass{};
        double g{};
        double num_leg_ee{};
        double freq{};
        double alpha1{};
        double alpha3{};
        double eps{};
    };

   private:
    void computeIMP(const std::string& frame, const Eigen::Matrix3d& R,
                    const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& linear_velocity,
                    Eigen::Vector3d force, std::optional<Eigen::Vector3d> torque = std::nullopt);

    bool is_initialized{};
    Eigen::Vector3d base_position_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d base_position_prev_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d base_linear_velocity_ = Eigen::Vector3d::Zero();
    Params params_;

    std::unordered_map<std::string, Eigen::Vector3d> pivots_;
    std::unordered_map<std::string, Eigen::Vector3d> contact_positions_;
    std::unordered_map<std::string, Eigen::Quaterniond> contact_orientations_;

    std::unordered_map<std::string, Eigen::Vector3d> feet_position_prev_;
    std::unordered_map<std::string, Eigen::Quaterniond> feet_orientation_prev_;

    std::optional<std::unordered_map<std::string, Eigen::Vector3d>> force_torque_offset_;

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LegOdometry(std::unordered_map<std::string, Eigen::Vector3d> feet_position,
                std::unordered_map<std::string, Eigen::Quaterniond> feet_orientation,
                double mass = 5.14, double alpha1 = 1.0, double alpha3 = 0.01, double freq = 100.0,
                double g = 9.81, double eps = 0.1,
                std::optional<std::unordered_map<std::string, Eigen::Vector3d>>
                    force_torque_offset = std::nullopt);

    const Eigen::Vector3d& getBasePosition() const;

    const Eigen::Vector3d& getBaseLinearVelocity() const;

    const std::unordered_map<std::string, Eigen::Vector3d> getContactPositions() const;

    const std::unordered_map<std::string, Eigen::Quaterniond> getContactOrientations() const;

    void estimate(
        const Eigen::Quaterniond& base_orientation, const Eigen::Vector3d& base_angular_velocity,
        const std::unordered_map<std::string, Eigen::Quaterniond>& base_to_foot_orientations,
        const std::unordered_map<std::string, Eigen::Vector3d>& base_to_foot_positions,
        const std::unordered_map<std::string, Eigen::Vector3d>& base_to_foot_linear_velocities,
        const std::unordered_map<std::string, Eigen::Vector3d>& base_to_foot_angular_velocities,
        const std::unordered_map<std::string, Eigen::Vector3d>& contact_forces,
        std::optional<std::unordered_map<std::string, Eigen::Vector3d>> contact_torques =
            std::nullopt);
};

}  // namespace serow
