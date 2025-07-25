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
#pragma once

#include <map>
#include <string>

#include "BaseEKF.hpp"
#include "CoMEKF.hpp"
#include "ContactDetector.hpp"
#include "ContactEKF.hpp"
#include "DerivativeEstimator.hpp"
#include "ExteroceptionLogger.hpp"
#include "LegOdometry.hpp"
#include "LocalTerrainMapper.hpp"
#include "Mahony.hpp"
#include "Measurement.hpp"
#include "MeasurementLogger.hpp"
#include "NaiveLocalTerrainMapper.hpp"
#include "ProprioceptionLogger.hpp"
#include "RobotKinematics.hpp"
#include "State.hpp"
#include "ThreadPool.hpp"
#include "Timer.hpp"
#include "common.hpp"

namespace serow {

/**
 * @class Serow
 * @brief Implements the SEROW legged robot state estimator
 */
class Serow {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// @brief Constructor
    Serow();

    /// @brief Destructor
    ~Serow();

    /// @brief initializes SEROW's configuration
    /// @param config configuration to initialize SEROW with
    /// @return true if SEROW was initialized successfully
    bool initialize(const std::string& config);

    /// @brief runs SEROW's estimator and updates the internal state
    /// @param imu base imu measurement includes linear acceleration and angular velocity
    /// @param joints joint encoder measurements includes position and optionally velocity
    /// @param ft optional leg end-effector Force/Torque measurements includes ground reaction force
    /// and optionally ground reaction torque
    /// @param odom an optional exteroceptive base odometry measurement
    /// @param contact_probabilities optional leg end-effector contact probabilities if not provided
    /// they will be estimated from the corresponding F/T measurement
    /// @param base_pose_ground_truth optional ground truth base pose measurement for logging
    /// @return true if the filter was successful
    bool filter(ImuMeasurement imu, std::map<std::string, JointMeasurement> joints,
                std::optional<std::map<std::string, ForceTorqueMeasurement>> ft = std::nullopt,
                std::optional<OdometryMeasurement> odom = std::nullopt,
                std::optional<std::map<std::string, ContactMeasurement>> contact_probabilities =
                    std::nullopt,
                std::optional<BasePoseGroundTruth> base_pose_ground_truth = std::nullopt);

    /// @brief fetches SEROW's internal state
    /// @param allow_invalid whether to return the state even if SEROW hasn't yet converged to a
    /// valid estimate
    /// @return SEROW's internal state if available
    std::optional<State> getState(bool allow_invalid = false);

    /// @brief fetches SEROW's contact state
    /// @param allow_invalid whether to return the state even if SEROW hasn't yet converged to a
    /// @return SEROW's contact state if available
    std::optional<ContactState> getContactState(bool allow_invalid = false);

    /// @brief fetches SEROW's base state
    /// @param allow_invalid whether to return the state even if SEROW hasn't yet converged to a
    /// valid estimate
    /// @return SEROW's base state if available
    std::optional<BaseState> getBaseState(bool allow_invalid = false);

    /// @brief Returns the terrain_estimator_ object
    const std::shared_ptr<TerrainElevation>& getTerrainEstimator() const;

    /// @brief Returns true if SEROW is initialized
    /// @return true if SEROW is initialized
    bool isInitialized() const;

    /// @brief Resets SEROW's state and estimators
    void reset();

    /// @brief Sets the state of SEROW's internal state also resets the estimators
    /// @param state the state to set
    void setState(const State& state);

private:
    struct Params {
        /// @brief name of the robot
        std::string robot_name{};
        /// @brief base frame name
        std::string base_frame{};
        /// @brief gravity constant (m/s^2)
        double g{};
        /// @brief whether or not to estimate initial values for the IMU gyro/accelerometer biases
        bool calibrate_initial_imu_bias{};
        /// @brief number of IMU measurements to use for estimating the IMU gyro/accelerometer
        /// biases
        size_t max_imu_calibration_cycles{};
        /// @brief rate at which IMU measurements are available (Hz)
        double imu_rate{};
        /// @brief low-pass filter cutoff frequency (Hz), used to filter the IMU gyro measurement.
        /// Cannot be greater than imu_rate / 2.0
        double gyro_cutoff_frequency{};
        /// @brief if available - gyro bias (rad/s). Only applies if calibrate_imu = false
        Eigen::Vector3d bias_gyro{Eigen::Vector3d::Zero()};
        /// @brief if available - accelerometer bias (m/s^2). Only applies if calibrate_imu = false
        Eigen::Vector3d bias_acc{Eigen::Vector3d::Zero()};
        /// @brief rotation matrix from base frame to IMU gyro frame
        Eigen::Matrix3d R_base_to_gyro{Eigen::Matrix3d::Identity()};
        /// @brief rotation matrix from base frame to IMU accelerometer frame
        Eigen::Matrix3d R_base_to_acc{Eigen::Matrix3d::Identity()};
        /// @brief rate at which joint encoders produce measurements (Hz)
        double joint_rate{};
        /// @brief whether or not to estimate the joint velocities, if set to false then the user
        /// must provide them
        bool estimate_joint_velocity{};
        /// @brief low-pass filter cutoff frequency (Hz), used to filter the joint encoder
        /// measurements. Cannot be greater than joint_rate / 2.0. Only applies if
        /// estimate_joint_velocity = true
        double joint_cutoff_frequency{};
        /// @brief measurement noise of joint encoder (rad^2)
        double joint_position_variance{};
        /// @brief low-pass filter cutoff frequency (Hz), used to filter the angular momentum
        /// around the CoM. Cannot be greater than joint_rate / 2.0
        double angular_momentum_cutoff_frequency{};
        /// @brief cost weight when computing the instantaneous moment pivot with kinematics. Only
        /// applies for flat feet
        double tau_0{};
        /// @brief cost weight when computing the instantaneous moment pivot with F/T. Only applies
        /// for flat feet
        double tau_1{};
        /// @brief rate at which leg end-effector force (and optionally torque) measurements are
        /// available
        double force_torque_rate{};
        /// @brief rotation matrix from each foot frame to force sensor frame
        std::map<std::string, Eigen::Matrix3d> R_foot_to_force;
        /// @brief rotation matrix from each foot frame to torque sensor frame. Can be empty if the
        /// robot only has FSRs instead of F/T
        std::map<std::string, Eigen::Matrix3d> R_foot_to_torque;
        /// @brief whether or not to estimate the leg end-effector contact status. If set to false,
        /// the user should provide the end-effector contact probabilities
        bool estimate_contact_status{};
        /// @brief Schmidt-Trigger high threshold (N) i.e. mass * g / 3.0. Note, must be very
        /// carefully tuned since base estimator is directly affected by the robot's contact status.
        /// Only applies if estimate_contact_status = true
        double high_threshold{};
        /// @brief Schmidt-Trigger high threshold (N) i.e. mass * g / 6.0. Note, must be very
        /// carefully tuned since base estimator is directly affected by the robot's contact status.
        /// Only applies if estimate_contact_status = true
        double low_threshold{};
        /// @brief moving median filter batch used to remove leg end-effector vertical force
        /// outliers. Only applies if estimate_contact_status = true
        size_t median_window{};
        /// @brief whether or not to remove outlier leg end-effector kinematic measurements in base
        /// estimation
        bool outlier_detection{};
        /// @brief after how many loop cycles SEROW's estimate is considered valid e.g. all the
        /// filters have converged and estimate should become available for feedback
        size_t convergence_cycles{};
        /// @brief gyro random walk. Can be found in the IMU data sheet or computed with the Alan
        /// variance method
        Eigen::Vector3d angular_velocity_cov{Eigen::Vector3d::Zero()};
        /// @brief gyro bias instability (or stability). Can be found in the IMU data sheet or
        /// computed with the Alan variance method
        Eigen::Vector3d angular_velocity_bias_cov{Eigen::Vector3d::Zero()};
        /// @brief accelerometer random walk. Can be found in the IMU data sheet or computed with
        /// the Alan variance method
        Eigen::Vector3d linear_acceleration_cov{Eigen::Vector3d::Zero()};
        /// @brief accelerometer bias instanbility (or stability). Can be found in the IMU data
        /// sheet or computed with the Alan variance method
        Eigen::Vector3d linear_acceleration_bias_cov{Eigen::Vector3d::Zero()};
        /// @brief minimum leg end-effector contact relative to base frame position uncertainty
        /// (m^2)
        Eigen::Vector3d contact_position_cov{Eigen::Vector3d::Zero()};
        /// @brief minimum leg end-effector contact orientation relative to base frame uncertainty
        /// (rad^2)
        Eigen::Vector3d contact_orientation_cov{Eigen::Vector3d::Zero()};
        /// @brief leg end-effector contact position slip uncertainty (m^2)
        Eigen::Vector3d contact_position_slip_cov{Eigen::Vector3d::Zero()};
        /// @brief leg end-effector contact orientation slip uncertainty (rad^2)
        Eigen::Vector3d contact_orientation_slip_cov{Eigen::Vector3d::Zero()};
        /// @brief CoM position process noise (m^2)
        Eigen::Vector3d com_position_process_cov{Eigen::Vector3d::Zero()};
        /// @brief CoM linear velocity process noise (m^2/s^2)
        Eigen::Vector3d com_linear_velocity_process_cov{Eigen::Vector3d::Zero()};
        /// @brief External forces acting on CoM process noise (N^2)
        Eigen::Vector3d external_forces_process_cov{Eigen::Vector3d::Zero()};
        /// @brief CoM position uncertainty (m^2) as computed with robot kinematics
        Eigen::Vector3d com_position_cov{Eigen::Vector3d::Zero()};
        /// @brief CoM linear acceleration (m^2/s^4) as approximated with the IMU acceleration and
        /// robot kinematics
        Eigen::Vector3d com_linear_acceleration_cov{Eigen::Vector3d::Zero()};
        /// @brief initial uncertainty for the base position (m^2)
        Eigen::Vector3d initial_base_position_cov{Eigen::Vector3d::Zero()};
        /// @brief initial uncertainty for the base orientation (rad^2)
        Eigen::Vector3d initial_base_orientation_cov{Eigen::Vector3d::Zero()};
        /// @brief initial uncertainty for the base linear velocity (m^2/s^2)
        Eigen::Vector3d initial_base_linear_velocity_cov{Eigen::Vector3d::Zero()};
        /// @brief initial uncertainty for the leg end-effector contact position (m^2)
        Eigen::Vector3d initial_contact_position_cov{Eigen::Vector3d::Zero()};
        /// @brief initial uncertainty for the leg end-effector contact orientation (rad^2), only
        /// applies for flat feet
        Eigen::Vector3d initial_contact_orientation_cov{Eigen::Vector3d::Zero()};
        /// @brief initial uncertainty for IMU linear acceleration bias (m^2/s^4)
        Eigen::Vector3d initial_imu_linear_acceleration_bias_cov{Eigen::Vector3d::Zero()};
        /// @brief initial uncertainty for IMU gyro bias (rad^2/s^2)
        Eigen::Vector3d initial_imu_angular_velocity_bias_cov{Eigen::Vector3d::Zero()};
        /// @brief initial uncertainty for the CoM position (m^2)
        Eigen::Vector3d initial_com_position_cov{Eigen::Vector3d::Zero()};
        /// @brief initial uncertainty for the CoM linear velocity (m^2/s^2)
        Eigen::Vector3d initial_com_linear_velocity_cov{Eigen::Vector3d::Zero()};
        /// @brief initial uncertainty for the external forces acting on the CoM (N)
        Eigen::Vector3d initial_external_forces_cov{Eigen::Vector3d::Zero()};
        double eps{0.1};
        /// @brief rigid body transformation from optional exteroceptive (visual/lidar odometry) to
        /// base frame. Is not specified if no exteroceptive odometry is provided
        Eigen::Isometry3d T_base_to_odom{Eigen::Isometry3d::Identity()};
        /// @brief whether or not to enable the terrain elevation estimation
        bool enable_terrain_estimation{};
        /// @brief minimum terrain measurement uncertainty (m^2), only applies if
        /// enable_terrain_estimation = true
        double minimum_terrain_height_variance{};
        bool is_contact_ekf{};
        Eigen::Vector3d base_linear_velocity_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d base_orientation_cov{Eigen::Vector3d::Zero()};
        std::string terrain_estimator_type{};
        bool log_data{true};
        bool log_measurements{false};
        /// @brief directory where log files will be stored
        std::string log_dir{"/tmp"};
        /// @brief offset between the base frame and the ground truth base frame
        Eigen::Isometry3d T_base_to_ground_truth{Eigen::Isometry3d::Identity()};
        /// @brief proportional gain for the base attitude estimator
        double Kp{0.0};
        /// @brief integral gain for the base attitude estimator
        double Ki{0.0};
        /// @brief set of contact frames
        std::set<std::string> contacts_frame{};
        /// @brief whether or not the robot has point feet
        bool point_feet{false};
    };

    /// @brief SEROW's configuration
    Params params_;
    /// @brief holds the joint name to joint state estimator
    std::map<std::string, DerivativeEstimator> joint_estimators_;
    /// @brief angular momentum rate around the CoM estimator
    std::unique_ptr<DerivativeEstimator> angular_momentum_derivative_estimator;
    /// @brief base angular acceleration estimator
    std::unique_ptr<DerivativeEstimator> gyro_derivative_estimator;
    /// @brief holds the contact frame name to leg end-effector contact estimator
    std::map<std::string, ContactDetector> contact_estimators_;
    /// @brief SEROW's internal state
    State state_;
    /// @brief base estimator that fuses base IMU, leg end-effector contact and relative to the base
    /// leg kinematic measurements
    ContactEKF base_estimator_con_;
    BaseEKF base_estimator_;
    /// @brief coM estimator that fuses ground reaction force, base IMU, and CoM kinematic
    /// measurements
    CoMEKF com_estimator_;
    /// @brief base attitude estimator that utilizes base IMU measurements
    std::unique_ptr<Mahony> attitude_estimator_;
    /// @brief end-effector kinematic estimator that employs base attitude and joint measurements
    std::unique_ptr<RobotKinematics> kinematic_estimator_;
    /// @brief leg odometry estimator that employs end-effector kinematics to estimate the
    /// instantaneous moment pivot points
    std::unique_ptr<LegOdometry> leg_odometry_;
    /// @brief indicates whether SEROW is initialized or not
    bool is_initialized_{};
    /// @brief IMU bias estimation cycles
    size_t cycle_{};
    size_t imu_calibration_cycles_{};
    /// @brief Terrain elevation mapper
    std::shared_ptr<TerrainElevation> terrain_estimator_;
    /// @brief Data loggers
    std::unique_ptr<ProprioceptionLogger> proprioception_logger_;
    std::unique_ptr<ExteroceptionLogger> exteroception_logger_;
    std::unique_ptr<MeasurementLogger> measurement_logger_;
    /// @brief Threadpool job for proprioceptive logging
    std::unique_ptr<ThreadPool> proprioception_logger_job_;
    /// @brief Threadpool job for exteroceptive logging
    std::unique_ptr<ThreadPool> exteroception_logger_job_;
    /// @brief Threadpool job for measurement logging
    std::unique_ptr<ThreadPool> measurement_logger_job_;
    /// @brief Threadpool job for logging the filter timings
    std::unique_ptr<ThreadPool> timings_logger_job_;
    /// @brief Frame transformations
    std::map<std::string, Eigen::Isometry3d> frame_tfs_;
    /// @brief IMU outlier detection storage
    std::vector<MovingMedianFilter> imu_outlier_detector_;

    /// @brief Timers for the filter functions
    std::unordered_map<std::string, Timer> timers_;
    /// @brief Last time the timings were logged
    std::optional<std::chrono::high_resolution_clock::time_point> last_log_time_;
    /// @brief Logs the measurements
    /// @param imu IMU measurement
    /// @param joints joint measurements
    /// @param ft force/torque measurements
    /// @param base_pose_ground_truth ground truth base pose
    void logMeasurements(ImuMeasurement imu, const std::map<std::string, JointMeasurement>& joints,
                         std::map<std::string, ForceTorqueMeasurement> ft,
                         std::optional<BasePoseGroundTruth> base_pose_ground_truth = std::nullopt);

    /// @brief Runs all the joint estimators to estimate the joint positions and velocities
    /// @param state the state of the robot
    /// @param joints joint measurements
    void runJointsEstimator(State& state, const std::map<std::string, JointMeasurement>& joints);

    /// @brief Runs the IMU estimator
    /// @param state the state of the robot
    /// @param imu IMU measurement
    /// @return true if the IMU estimation is calibrated and initialized
    bool runImuEstimator(State& state, ImuMeasurement& imu);

    /// @brief Runs the forward kinematics
    /// @param state the state of the robot
    /// @return kinematic measurements
    KinematicMeasurement runForwardKinematics(State& state);

    /// @brief Computes the leg odometry and updates the kinematic measurements accordingly
    /// @param state the state of the robot
    /// @param imu IMU measurement
    /// @param kin kinematic measurements
    void computeLegOdometry(const State& state, const ImuMeasurement& imu,
                            KinematicMeasurement& kin);

    /// @brief Runs the angular momentum estimator
    /// @param state the state of the robot
    void runAngularMomentumEstimator(State& state);

    /// @brief Runs the contact estimator to estimate the leg end-effector contact state
    /// @param state the state of the robot
    /// @param ft force/torque measurements
    /// @param kin kinematic measurements
    /// @param contacts_probability contact probabilities
    void runContactEstimator(
        State& state, std::map<std::string, ForceTorqueMeasurement>& ft, KinematicMeasurement& kin,
        std::optional<std::map<std::string, ContactMeasurement>> contacts_probability);

    /// @brief Runs the base estimator
    /// @param state the state of the robot
    /// @param imu IMU measurement
    /// @param kin kinematic measurements
    /// @param odom exteroceptive odometry measurement
    void runBaseEstimator(State& state, const ImuMeasurement& imu, const KinematicMeasurement& kin,
                          std::optional<OdometryMeasurement> odom);

    /// @brief Runs the CoM estimator
    /// @param state the state of the robot
    /// @param kin kinematic measurements
    /// @param ft force/torque measurements
    void runCoMEstimator(State& state, KinematicMeasurement& kin,
                         std::map<std::string, ForceTorqueMeasurement> ft);

    /// @brief Computes the frame transformations for all frames in the robot model
    /// @param state the state of the robot
    void updateFrameTree(const State& state);

    /// @brief Logs the proprioceptive measurements
    /// @param state the state of the robot
    /// @param imu IMU measurement
    void logProprioception(const State& state, const ImuMeasurement& imu);

    /// @brief Logs the exteroceptive measurements
    /// @param state the state of the robot
    void logExteroception(const State& state);

    /// @brief Initializes all the data loggers and the corresponding thread pool jobs
    void initializeLogging();

    /// @brief Stops SEROW's logging threads
    void stopLogging();

    /// @brief Checks if IMU measurement is an outlier using Median Absolute Deviation (MAD)
    /// @param imu IMU measurement to check
    /// @return true if the measurement is an outlier
    bool isImuMeasurementOutlier(const ImuMeasurement& imu);

    /// @brief Logs the filter timings
    void logTimings();
};

}  // namespace serow
