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

    const double theta2 = omega.dot(omega);
    const double theta = std::sqrt(theta2);
    const Eigen::Matrix3d omega_skew = wedge(omega);
    const Eigen::Matrix3d omega_skew2 = omega_skew * omega_skew;
    double alpha;
    double beta;

    if (theta2 > std::numeric_limits<double>::epsilon()) {
        const double sin_theta = std::sin(theta);
        alpha = sin_theta / theta;
        const double s2 = std::sin(theta / 2.0);
        const double one_minus_cos = 2.0 * s2 * s2;  // numerically better than [1 - cos(theta)]
        beta = one_minus_cos / theta2;
    } else {
        // Taylor expansion at 0
        alpha = 1.0 - theta2 * 1.0 / 6.0;
        beta = 0.5 - theta2 * 1.0 / 24.0;
    }

    res += alpha * omega_skew;
    res += beta * omega_skew2;
    return res;
}

/// @brief Computes the logarithmic map for a component in SO(3) group
/// @param Rt 3x3 Rotation in SO(3) group
/// @return 3D twist in so(3) algebra
/// @note Transferred from GTSAM
inline Eigen::Vector3d logMap(const Eigen::Matrix3d& Rt) {
    // note switch to base 1
    const Eigen::Matrix3d& R = Rt.matrix();
    const double& R11 = R(0, 0);
    const double& R12 = R(0, 1);
    const double& R13 = R(0, 2);
    const double& R21 = R(1, 0);
    const double& R22 = R(1, 1);
    const double& R23 = R(1, 2);
    const double& R31 = R(2, 0);
    const double& R32 = R(2, 1);
    const double& R33 = R(2, 2);

    // Get the trace of R
    const double tr = R.trace();

    Eigen::Vector3d omega = Eigen::Vector3d::Zero();

    // Special case when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
    if (tr + 1.0 < 1e-3) {
        if (R33 > R22 && R33 > R11) {
            // R33 is the largest diagonal, a=3, b=1, c=2
            const double W = R21 - R12;
            const double Q1 = 2.0 + 2.0 * R33;
            const double Q2 = R31 + R13;
            const double Q3 = R23 + R32;
            const double r = std::sqrt(Q1);
            const double one_over_r = 1 / r;
            const double norm = std::sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
            const double sgn_w = W < 0 ? -1.0 : 1.0;
            const double mag = M_PI - (2 * sgn_w * W) / norm;
            const double scale = 0.5 * one_over_r * mag;
            omega = sgn_w * scale * Eigen::Vector3d(Q2, Q3, Q1);
        } else if (R22 > R11) {
            // R22 is the largest diagonal, a=2, b=3, c=1
            const double W = R13 - R31;
            const double Q1 = 2.0 + 2.0 * R22;
            const double Q2 = R23 + R32;
            const double Q3 = R12 + R21;
            const double r = std::sqrt(Q1);
            const double one_over_r = 1 / r;
            const double norm = std::sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
            const double sgn_w = W < 0 ? -1.0 : 1.0;
            const double mag = M_PI - (2 * sgn_w * W) / norm;
            const double scale = 0.5 * one_over_r * mag;
            omega = sgn_w * scale * Eigen::Vector3d(Q3, Q1, Q2);
        } else {
            // R11 is the largest diagonal, a=1, b=2, c=3
            const double W = R32 - R23;
            const double Q1 = 2.0 + 2.0 * R11;
            const double Q2 = R12 + R21;
            const double Q3 = R31 + R13;
            const double r = std::sqrt(Q1);
            const double one_over_r = 1 / r;
            const double norm = std::sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
            const double sgn_w = W < 0 ? -1.0 : 1.0;
            const double mag = M_PI - (2 * sgn_w * W) / norm;
            const double scale = 0.5 * one_over_r * mag;
            omega = sgn_w * scale * Eigen::Vector3d(Q1, Q2, Q3);
        }
    } else {
        double magnitude = 0.0;
        const double tr_3 = tr - 3.0;  // could be non-negative if the matrix is off orthogonal
        if (tr_3 < -1e-6) {
            // this is the normal case -1 < trace < 3
            const double theta = std::acos((tr - 1.0) / 2.0);
            magnitude = theta / (2.0 * std::sin(theta));
        } else {
            // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            // use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            // see https://github.com/borglab/gtsam/issues/746 for details
            magnitude = 0.5 - tr_3 / 12.0 + tr_3 * tr_3 / 60.0;
        }
        omega = magnitude * Eigen::Vector3d(R32 - R23, R13 - R31, R21 - R12);
    }

    return omega;
}

/// @brief Computes the logarithmic map for a component in SO(3) group
/// @param q quaternion in SO(3) group
/// @return 3D twist in so(3) algebra
inline Eigen::Vector3d logMap(const Eigen::Quaterniond& q) { return logMap(q.toRotationMatrix()); }

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
    return logMap(R1.transpose() * R0);
}

/// @brief Performs the SO(3) group minus operation
/// @param q0 quaternion in SO(3) group
/// @param q1 quaternion in SO(3) group
/// @return A 3D twist in so(3) algebra
inline Eigen::Vector3d minus(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1) {
    return logMap(q1.inverse() * q0);
}

}  // namespace so3

}  // namespace lie
