/*
 * Copyright 2017-2021 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH)
 *		 nor the names of its contributors may be used to endorse or promote products
 *derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @brief leg odometry for Bipeds based on force/torque or pressure, and encoder measurement
 * @author Stylianos Piperakis
 * @details Estimates the 3D leg odometry of the base and the corresponding relative leg
 * measurements
 */
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#ifdef __linux__
    #include <eigen3/Eigen/Dense>
#else
    #include <Eigen/Dense>
#endif

namespace serow {

class LegOdometry {
    struct Params {
        double Tm{};
        double Tm2{};
        double Tm3{};
        double eps{};
        double mass{};
        double g{};
        double freq{};
        double alpha1{};
        double alpha3{};
    };

   private:
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

    LegOdometry(Eigen::Vector3d base_position,
                std::unordered_map<std::string, Eigen::Vector3d> feet_position,
                std::unordered_map<std::string, Eigen::Quaterniond> feet_orientation,
                double mass = 5.14, double alpha1 = 1.0, double alpha3 = 0.01, double freq = 100.0,
                double g = 9.81,
                std::optional<std::unordered_map<std::string, Eigen::Vector3d>>
                    force_torque_offset = std::nullopt);

    const Eigen::Vector3d& getBasePosition() const;

    const Eigen::Vector3d& getBaseLinearVelocity() const;

    void computeIMP(const std::string& frame, const Eigen::Matrix3d& R,
                    const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& linear_velocity,
                    Eigen::Vector3d force, Eigen::Vector3d torque);

    void estimate(
        const Eigen::Quaterniond& base_orientation, const Eigen::Vector3d& base_angular_velocity,
        const std::unordered_map<std::string, Eigen::Quaterniond>& base_to_foot_orientations,
        const std::unordered_map<std::string, Eigen::Vector3d>& base_to_foot_positions,
        const std::unordered_map<std::string, Eigen::Vector3d>& base_to_foot_linear_velocities,
        const std::unordered_map<std::string, Eigen::Vector3d>& base_to_foot_angular_velocities,
        std::unordered_map<std::string, double> contact_probabilities,
        const std::unordered_map<std::string, Eigen::Vector3d>& contact_forces,
        std::optional<std::unordered_map<std::string, Eigen::Vector3d>> contact_torques =
            std::nullopt);

    /** @fn     double cropGRF(double force)
     *  @brief  Crops the measured vertical ground reaction force (GRF) in the margins [0, mass * g]
     *  @param  force Measured GRF
     *  @return  The cropped GRF
     */
    double cropGRF(double force) const;
};

}  // namespace serow
