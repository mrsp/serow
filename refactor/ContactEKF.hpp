#pragma once
/**
 * @brief Base Estimator combining Inertial Measurement Unit (IMU) and Odometry Measuruements either
 * from leg odometry or external odometry e.g Visual Odometry (VO) or Lidar Odometry (LO)
 * @author Stylianos Piperakis
 * @details State is  position in World frame
 * velocity in  Base frame
 * orientation of Body frame wrt the World frame
 * accelerometer bias in Base frame
 * gyro bias in Base frame
 * Measurements are: Base Position/Orinetation in World frame by Leg Odometry or Visual Odometry
 * (VO) or Lidar Odometry (LO), when VO/LO is considered the kinematically computed base velocity
 * (Twist) is also employed for update.
 */

#include <iostream>

#include "State.hpp"

// State is pos - vel - rot - accel - gyro bias - 15 + 6 x N contact pos - contact or
using namespace Eigen;

struct ImuMeasurement;
struct KinematicMeasurement;

class ContactEKF {
   public:
    ContactEKF();
    void init(State state);
    void predict(State state, ImuMeasurement imu, KinematicMeasurement kin);

   private:
    int num_states_{};
    int num_inputs_{};
    int contact_dim_{};
    int num_leg_end_effectors_{};
    double nominal_dt_{};

    Eigen::Vector3d g_;
    Eigen::Array3i v_idx_;
    Eigen::Array3i r_idx_;
    Eigen::Array3i p_idx_;
    Eigen::Array3i bg_idx_;
    Eigen::Array3i ba_idx_;
    std::unordered_map<std::string, Eigen::ArrayXi> pl_idx_;

    Eigen::Array3i ng_idx_;
    Eigen::Array3i na_idx_;
    Eigen::Array3i nbg_idx_;
    Eigen::Array3i nba_idx_;
    std::unordered_map<std::string, Eigen::ArrayXi> npl_idx_;

    /// Error Covariance, Linearized state transition model, Identity matrix, state uncertainty
    /// matrix 15 + 6N x 15 + 6N
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> I_, P_;
    /// Linearized state-input model 15 + 6N x 12 + 6N
    Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Lc_;

    State computeDiscreteDynamics(
        State state, double dt, Eigen::Vector3d angular_velocity,
        Eigen::Vector3d linear_acceleration,
        std::optional<std::unordered_map<std::string, bool>> contacts_status,
        std::optional<std::unordered_map<std::string, Eigen::Vector3d>> contacts_position,
        std::optional<std::unordered_map<std::string, Eigen::Quaterniond>> contacts_orientations =
            std::nullopt);

    MatrixXd computeDiscreteTransitionMatrix(State state, Vector3d angular_velocity,
                                             Vector3d linear_acceleration, double dt);

    std::optional<double> last_timestamp_;
};
