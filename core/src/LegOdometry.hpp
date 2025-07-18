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
 * @file LegOdometry.hpp
 * @brief Header file for the LegOdometry class.
 * @details Estimates the 3D leg odometry of the base and the corresponding relative leg
 * measurements based on force/torque or pressure, and encoder measurement.
 */

#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif
#include <map>
#include <optional>
#include <string>

namespace serow {

/**
 * @class LegOdometry
 * @brief Class for estimating the leg odometry of legged robots.
 */
class LegOdometry {
    /**
     * @struct Params
     * @brief Struct to hold various parameters for the LegOdometry class.
     */
    struct Params {
        double Tm{};          ///< Tm parameter
        double Tm2{};         ///< Tm2 parameter
        double Tm3{};         ///< Tm3 parameter
        double mass{};        ///< Mass of the robot
        double g{};           ///< Gravity constant
        double num_leg_ee{};  ///< Number of leg end-effectors
        double freq{};        ///< Joint state nominal frequency
        double alpha1{};      ///< Alpha1 parameter
        double alpha3{};      ///< Alpha3 parameter
        double eps{};         ///< Epsilon parameter
    };

private:
    /**
     * @brief Computes the Instanteneous Moment Pivot (IMP) for a given contact frame.
     * @param frame Contact frame name
     * @param R Base orientation in world coordinates
     * @param angular_velocity Angular velocity of the base in world coordinates
     * @param linear_velocity Linear velocity of the base in world coordinates
     * @param force Ground reaction force at the contact frame in world coordinates
     * @param torque Ground reaction torque at the contact frame in world coordinates (optional)
     */
    void computeIMP(const std::string& frame, const Eigen::Matrix3d& R,
                    const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& linear_velocity,
                    const Eigen::Vector3d& force,
                    std::optional<Eigen::Vector3d> torque = std::nullopt);

    bool is_initialized{};  ///< Flag indicating if the odometry has been initialized
    Eigen::Vector3d base_position_ =
        Eigen::Vector3d::Zero();  ///< Estimated base position in world coordinates
    Eigen::Vector3d base_position_prev_ =
        Eigen::Vector3d::Zero();  ///< Previous base position in world coordinates
    Eigen::Vector3d base_linear_velocity_ =
        Eigen::Vector3d::Zero();  ///< Estimated base linear velocity in world coordinates
    Params params_;               ///< Optimization parameters

    std::map<std::string, Eigen::Vector3d>
        pivots_;  ///< Pivot points for the feet in relative foot coordinates
    std::map<std::string, Eigen::Vector3d>
        contact_positions_;  ///< Contact positions of the feet relative to the base frame
    std::map<std::string, Eigen::Quaterniond>
        contact_orientations_;  ///< Contact orientations of the feet relative to the base frame

    std::map<std::string, Eigen::Vector3d>
        feet_position_prev_;  ///< Previous positions of the feet in world coordinates
    std::map<std::string, Eigen::Quaterniond>
        feet_orientation_prev_;  ///< Previous orientations of the feet in world coordinates

    std::optional<std::map<std::string, Eigen::Vector3d>>
        force_torque_offset_;  ///< Force/Torque sensor offsets from the contact frames

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Constructor for the LegOdometry class.
     * @param base_position Initial position of the base in world coordinates
     * @param feet_position Initial positions of the feet in world coordinates
     * @param feet_orientation Initial orientations of the feet in world coordinates
     * @param mass Mass of the robot
     * @param alpha1 Kinematic-term optimization weight parameter
     * @param alpha3 F/T-term optimization weight parameter
     * @param freq Joint state nominal frequency
     * @param g Gravity constant
     * @param eps Epsilon parameter
     * @param force_torque_offset Force/Torque sensor offsets from the contact frames (optional)
     */
    LegOdometry(
        const Eigen::Vector3d& base_position, std::map<std::string, Eigen::Vector3d> feet_position,
        std::map<std::string, Eigen::Quaterniond> feet_orientation, double mass, double alpha1,
        double alpha3, double freq, double g, double eps,
        std::optional<std::map<std::string, Eigen::Vector3d>> force_torque_offset = std::nullopt);

    /**
     * @brief Gets the estimated base position in world coordinates.
     * @return The estimated base position in world coordinates.
     */
    const Eigen::Vector3d& getBasePosition() const;

    /**
     * @brief Gets the estimated base linear velocity in world coordinates.
     * @return The estimated base linear velocity in world coordinates.
     */
    const Eigen::Vector3d& getBaseLinearVelocity() const;

    /**
     * @brief Gets the feet contact positions relative to the base frame.
     * @return The feet contact positions relative to the base frame.
     */
    const std::map<std::string, Eigen::Vector3d> getContactPositions() const;

    /**
     * @brief Gets the feet contact orientations relative to the base frame.
     * @return The feet contact orientations relative to the base frame.
     */
    const std::map<std::string, Eigen::Quaterniond> getContactOrientations() const;

    /**
     * @brief Estimates the odometry based on the provided measurements.
     * @param base_orientation Orientation of the base in world coordinates
     * @param base_angular_velocity Angular velocity of the base in world coordinates
     * @param base_to_foot_orientations Relative foot orientations from the base to the feet frame
     * @param base_to_foot_positions Relative foot positions from the base to the feet frame
     * @param base_to_foot_linear_velocities Relative foot linear velocities from the base to the
     * feet frame
     * @param base_to_foot_angular_velocities Relative foot angular velocities from the base to the
     * feet frame
     * @param contact_forces Contact forces at the feet frame
     * @param contact_torques Contact torques at the feet frame (optional)
     */
    void estimate(
        const Eigen::Quaterniond& base_orientation, const Eigen::Vector3d& base_angular_velocity,
        const std::map<std::string, Eigen::Quaterniond>& base_to_foot_orientations,
        const std::map<std::string, Eigen::Vector3d>& base_to_foot_positions,
        const std::map<std::string, Eigen::Vector3d>& base_to_foot_linear_velocities,
        const std::map<std::string, Eigen::Vector3d>& base_to_foot_angular_velocities,
        const std::map<std::string, Eigen::Vector3d>& contact_forces,
        const std::map<std::string, double>& contact_probabilities,
        std::optional<std::map<std::string, Eigen::Vector3d>> contact_torques = std::nullopt);
};

}  // namespace serow
