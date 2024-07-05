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
 * @file lie.hpp
 * @brief Header file for SO(3) Lie group and so(3) Lie algebra operations.
 * @details A thin wrapper of the essential SO(3)/so(3) operations e.g. expMap, logMap, plus, minus.
 * This header provides various utility functions to work with rotations and their algebra.
 */

#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif

namespace lie {

namespace so3 {
/// @brief Computes the skew symmetric matrix of a 3D vector
/// @param v 3D twist vector
/// @return 3x3 skew symmetric representation
inline Eigen::Matrix3d wedge(const Eigen::Vector3d& v) {
    Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
    skew(0, 1) = -v(2);
    skew(0, 2) = v(1);
    skew(1, 2) = -v(0);
    skew(1, 0) = v(2);
    skew(2, 0) = -v(1);
    skew(2, 1) = v(0);
    return skew;
}

/// @brief Computes the vector representation of a skew symmetric matrix
/// @param M 3x3 skew symmetric matrix
/// @return 3D twist vector
inline Eigen::Vector3d vec(const Eigen::Matrix3d& M) {
    return Eigen::Vector3d(M(2, 1), M(0, 2), M(1, 0));
}

/// @brief Computes the exponential map according to the Rodriquez Formula for component in SO(3)
/// @param omega 3D twist in so(3) algebra
/// @return 3x3 Rotation in SO(3) group
inline Eigen::Matrix3d expMap(const Eigen::Vector3d& omega) {
    Eigen::Matrix3d res = Eigen::Matrix3d::Identity();
    const double omeganorm = omega.norm();

    if (omeganorm > std::numeric_limits<double>::epsilon()) {
        Eigen::Matrix3d omega_skew = Eigen::Matrix3d::Zero();
        omega_skew = wedge(omega);
        res += omega_skew * (sin(omeganorm) / omeganorm);
        res += (omega_skew * omega_skew) * ((1.0 - cos(omeganorm)) / (omeganorm * omeganorm));
    }
    return res;
}

/// @brief Computes the logarithmic map for a component in SO(3) group
/// @param Rt 3x3 Rotation in SO(3) group
/// @return 3D twist in so(3) algebra
inline Eigen::Vector3d logMap(const Eigen::Matrix3d& Rt) {
    Eigen::Vector3d res = Eigen::Vector3d::Zero();
    const double costheta = (Rt.trace() - 1.0) / 2.0;
    const double theta = acos(costheta);

    if (std::fabs(theta) > std::numeric_limits<double>::epsilon()) {
        Eigen::Matrix3d lnR = Rt - Rt.transpose();
        lnR *= theta / (2.0 * sin(theta));
        res = vec(lnR);
    }

    return res;
}

/// @brief Computes the logarithmic map for a component in SO(3) group
/// @param q quaternion in SO(3) group
/// @return 3D twist in so(3) algebra
inline Eigen::Vector3d logMap(const Eigen::Quaterniond& q) {
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();
    // Get the vector part
    Eigen::Vector3d qv = Eigen::Vector3d(q.x(), q.y(), q.z());
    qv *= (1.0 / qv.norm());
    omega = qv * (2.0 * std::acos(q.w() / q.norm()));
    if (std::isnan(omega(0) + omega(1) + omega(2))) {
        omega = Eigen::Vector3d::Zero();
    }
    return omega;
}

/// @brief Performs the SO(3) group plus operation
/// @param R 3x3 Rotation in SO(3) group
/// @param tau 3D twist in so(3) algebra
/// @return A 3x3 Rotation in SO(3) group
inline Eigen::Matrix3d plus(const Eigen::Matrix3d& R, const Eigen::Vector3d& tau) {
    return R * expMap(tau);
}

/// @brief Performs the SO(3) group minus operation
/// @param R0 3x3 Rotation in SO(3) group
/// @param R1 3x3 Rotation in SO(3) group
/// @return A 3D twist in so(3) algebra
inline Eigen::Vector3d minus(const Eigen::Matrix3d& R0, const Eigen::Matrix3d& R1) {
    return logMap(R0.transpose() * R1);
}

/// @brief Performs the SO(3) group minus operation
/// @param q0 quaternion in SO(3) group
/// @param q1 quaternion in SO(3) group
/// @return A 3D twist in so(3) algebra
inline Eigen::Vector3d minus(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1) {
    return logMap(q0.inverse() * q1);
}

}  // namespace so3

}  // namespace lie
