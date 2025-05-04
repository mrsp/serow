#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <tuple>

#include "ContactEKF.hpp"
#include "Measurement.hpp"
#include "State.hpp"
#include "LocalTerrainMapper.hpp"
#include "common.hpp"
#include "Serow.hpp"

namespace py = pybind11;

// Helper function to convert numpy array to Eigen quaternion
Eigen::Quaterniond numpy_to_quaternion(const py::array_t<double>& arr) {
    auto buf = arr.request();
    if (buf.ndim != 1 || buf.shape[0] != 4) {
        throw std::runtime_error("Quaternion must be a 4-element array [w, x, y, z]");
    }
    double* ptr = static_cast<double*>(buf.ptr);
    return Eigen::Quaterniond(ptr[0], ptr[1], ptr[2], ptr[3]);
}

// Helper function to convert Eigen quaternion to numpy array
py::array_t<double> quaternion_to_numpy(const Eigen::Quaterniond& q) {
    auto result = py::array_t<double>(4);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    ptr[0] = q.w();
    ptr[1] = q.x();
    ptr[2] = q.y();
    ptr[3] = q.z();
    return result;
}

PYBIND11_MODULE(serow, m) {
    m.doc() = "Python bindings for Serow library classes (ContactEKF, State, and Measurements)";

    // Binding for BaseState
    py::class_<serow::BaseState>(m, "BaseState", "Represents the state of a humanoid robot")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::BaseState::timestamp, "Timestamp of the state")
        .def_readwrite("base_position", &serow::BaseState::base_position, "Base position (3D vector)")
        .def_property(
            "base_orientation",
            [](const serow::BaseState& self) { return quaternion_to_numpy(self.base_orientation); },
            [](serow::BaseState& self, const py::array_t<double>& arr) {
                self.base_orientation = numpy_to_quaternion(arr);
            },
            "Base orientation as a quaternion (w, x, y, z)")
        .def_readwrite("base_linear_velocity", &serow::BaseState::base_linear_velocity, "Base linear velocity (3D vector)")
        .def_readwrite("base_angular_velocity", &serow::BaseState::base_angular_velocity, "Base angular velocity (3D vector)")
        .def_readwrite("base_linear_acceleration", &serow::BaseState::base_linear_acceleration, "Base linear acceleration (3D vector)")
        .def_readwrite("base_angular_acceleration", &serow::BaseState::base_angular_acceleration, "Base angular acceleration (3D vector)")
        .def_readwrite("imu_angular_velocity_bias", &serow::BaseState::imu_angular_velocity_bias, "IMU angular velocity bias (3D vector)")
        .def_readwrite("imu_linear_acceleration_bias", &serow::BaseState::imu_linear_acceleration_bias, "IMU linear acceleration bias (3D vector)")
        .def_readwrite("contacts_position", &serow::BaseState::contacts_position, "Map of contact positions (string to 3D vector)")
        .def_property(
            "contacts_orientation",
            [](const serow::BaseState& self) {
                std::map<std::string, py::array_t<double>> result;
                if (self.contacts_orientation) {
                    for (const auto& [name, quat] : *self.contacts_orientation) {
                        result[name] = quaternion_to_numpy(quat);
                    }
                }
                return result;
            },
            [](serow::BaseState& self, const std::map<std::string, py::array_t<double>>& orientations) {
                std::map<std::string, Eigen::Quaterniond> eigen_orientations;
                for (const auto& [name, arr] : orientations) {
                    eigen_orientations[name] = numpy_to_quaternion(arr);
                }
                self.contacts_orientation = eigen_orientations;
            },
            "Map of contact orientations (string to quaternion)")
        .def_readwrite("base_position_cov", &serow::BaseState::base_position_cov, "Base position covariance (3x3 matrix)")
        .def_readwrite("base_orientation_cov", &serow::BaseState::base_orientation_cov, "Base orientation covariance (3x3 matrix)")
        .def_readwrite("base_linear_velocity_cov", &serow::BaseState::base_linear_velocity_cov, "Base linear velocity covariance (3x3 matrix)")
        .def_readwrite("base_angular_velocity_cov", &serow::BaseState::base_angular_velocity_cov, "Base angular velocity covariance (3x3 matrix)")
        .def_readwrite("imu_angular_velocity_bias_cov", &serow::BaseState::imu_angular_velocity_bias_cov, "IMU angular velocity bias covariance (3x3 matrix)")
        .def_readwrite("imu_linear_acceleration_bias_cov", &serow::BaseState::imu_linear_acceleration_bias_cov, "IMU linear acceleration bias covariance (3x3 matrix)")
        .def_readwrite("contacts_position_cov", &serow::BaseState::contacts_position_cov, "Map of contact position covariances (string to 3x3 matrix)")
        .def_readwrite("contacts_orientation_cov", &serow::BaseState::contacts_orientation_cov, "Map of contact orientation covariances (string to 3x3 matrix)");

    // Binding for ContactState
    py::class_<serow::ContactState>(m, "ContactState", "Represents the contact state of a humanoid robot")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::ContactState::timestamp, "Timestamp of the contact state")
        .def_readwrite("contacts_status", &serow::ContactState::contacts_status, "Map of contact statuses (string to bool)")
        .def_readwrite("contacts_probability", &serow::ContactState::contacts_probability, "Map of contact probabilities (string to double)")
        .def_readwrite("contacts_force", &serow::ContactState::contacts_force, "Map of contact forces (string to 3D vector)")
        .def_readwrite("contacts_torque", &serow::ContactState::contacts_torque, "Map of contact torques (string to 3D vector)");

    // Binding for JointMeasurement
    py::class_<serow::JointMeasurement>(m, "JointMeasurement", "Represents a joint measurement")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::JointMeasurement::timestamp, "Timestamp of the measurement (s)")
        .def_readwrite("position", &serow::JointMeasurement::position, "Joint position measurement (rad)")
        .def_readwrite("velocity", &serow::JointMeasurement::velocity, "Optional joint velocity measurement (rad/s)");

    // Binding for ImuMeasurement
    py::class_<serow::ImuMeasurement>(m, "ImuMeasurement", "Represents IMU measurements")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::ImuMeasurement::timestamp, "Timestamp of the measurement (s)")
        .def_readwrite("linear_acceleration", &serow::ImuMeasurement::linear_acceleration, "Linear acceleration measured by IMU (m/s^2)")
        .def_readwrite("angular_velocity", &serow::ImuMeasurement::angular_velocity, "Angular velocity measured by IMU (rad/s)")
        .def_property(
            "orientation",
            [](const serow::ImuMeasurement& self) { return quaternion_to_numpy(self.orientation); },
            [](serow::ImuMeasurement& self, const py::array_t<double>& arr) {
                self.orientation = numpy_to_quaternion(arr);
            },
            "Orientation measured by IMU (quaternion)")
        .def_readwrite("angular_acceleration", &serow::ImuMeasurement::angular_acceleration, "Angular acceleration measured by IMU (rad/s^2)")
        .def_readwrite("angular_velocity_cov", &serow::ImuMeasurement::angular_velocity_cov, "Covariance matrix of angular velocity (rad^2/s^2)")
        .def_readwrite("linear_acceleration_cov", &serow::ImuMeasurement::linear_acceleration_cov, "Covariance matrix of linear acceleration (m^2/s^4)")
        .def_readwrite("angular_velocity_bias_cov", &serow::ImuMeasurement::angular_velocity_bias_cov, "Covariance matrix of angular velocity bias (rad^2/s^2)")
        .def_readwrite("linear_acceleration_bias_cov", &serow::ImuMeasurement::linear_acceleration_bias_cov, "Covariance matrix of linear acceleration bias (m^2/s^4)");

    // Binding for ForceTorqueMeasurement
    py::class_<serow::ForceTorqueMeasurement>(m, "ForceTorqueMeasurement", "Represents force-torque sensor measurements")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::ForceTorqueMeasurement::timestamp, "Timestamp of the measurement (s)")
        .def_readwrite("force", &serow::ForceTorqueMeasurement::force, "Force measured by force-torque sensor (N)")
        .def_readwrite("cop", &serow::ForceTorqueMeasurement::cop, "Center of pressure (COP) measured by force-torque sensor (m)")
        .def_readwrite("torque", &serow::ForceTorqueMeasurement::torque, "Optional torque measured by force-torque sensor (Nm)");

    // Binding for KinematicMeasurement
    py::class_<serow::KinematicMeasurement>(m, "KinematicMeasurement", "Represents kinematic measurements")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::KinematicMeasurement::timestamp, "Timestamp of the measurement")
        .def_readwrite("base_linear_velocity", &serow::KinematicMeasurement::base_linear_velocity, "Base linear velocity (3D vector)")
        .def_property(
            "base_orientation",
            [](const serow::KinematicMeasurement& self) { return quaternion_to_numpy(self.base_orientation); },
            [](serow::KinematicMeasurement& self, const py::array_t<double>& arr) {
                self.base_orientation = numpy_to_quaternion(arr);
            },
            "Base orientation as a quaternion (w, x, y, z)")
        .def_readwrite("contacts_status", &serow::KinematicMeasurement::contacts_status, "Map of contact statuses (string to bool)")
        .def_readwrite("contacts_probability", &serow::KinematicMeasurement::contacts_probability, "Map of contact probabilities (string to double)")
        .def_readwrite("contacts_position", &serow::KinematicMeasurement::contacts_position, "Map of contact positions (string to 3D vector)")
        .def_readwrite("base_to_foot_positions", &serow::KinematicMeasurement::base_to_foot_positions, "Map of base-to-foot positions (string to 3D vector)")
        .def_readwrite("contacts_position_noise", &serow::KinematicMeasurement::contacts_position_noise, "Map of contact position noise (string to 3x3 matrix)")
        .def_property(
            "contacts_orientation",
            [](const serow::KinematicMeasurement& self) {
                std::map<std::string, py::array_t<double>> result;
                if (self.contacts_orientation) {
                    for (const auto& [name, quat] : *self.contacts_orientation) {
                        result[name] = quaternion_to_numpy(quat);
                    }
                }
                return result;
            },
            [](serow::KinematicMeasurement& self, const std::map<std::string, py::array_t<double>>& orientations) {
                std::map<std::string, Eigen::Quaterniond> eigen_orientations;
                for (const auto& [name, arr] : orientations) {
                    eigen_orientations[name] = numpy_to_quaternion(arr);
                }
                self.contacts_orientation = eigen_orientations;
            },
            "Map of contact orientations (string to quaternion)")
        .def_readwrite("contacts_orientation_noise", &serow::KinematicMeasurement::contacts_orientation_noise, "Map of contact orientation noise (string to 3x3 matrix)")
        .def_readwrite("com_angular_momentum_derivative", &serow::KinematicMeasurement::com_angular_momentum_derivative, "Center of mass angular momentum derivative (3D vector)")
        .def_readwrite("com_position", &serow::KinematicMeasurement::com_position, "Center of mass position (3D vector)")
        .def_readwrite("com_linear_acceleration", &serow::KinematicMeasurement::com_linear_acceleration, "Center of mass linear acceleration (3D vector)")
        .def_readwrite("base_linear_velocity_cov", &serow::KinematicMeasurement::base_linear_velocity_cov, "Base linear velocity covariance (3x3 matrix)")
        .def_readwrite("base_orientation_cov", &serow::KinematicMeasurement::base_orientation_cov, "Base orientation covariance (3x3 matrix)")
        .def_readwrite("position_slip_cov", &serow::KinematicMeasurement::position_slip_cov, "Position slip covariance (3x3 matrix)")
        .def_readwrite("orientation_slip_cov", &serow::KinematicMeasurement::orientation_slip_cov, "Orientation slip covariance (3x3 matrix)")
        .def_readwrite("position_cov", &serow::KinematicMeasurement::position_cov, "Position covariance (3x3 matrix)")
        .def_readwrite("orientation_cov", &serow::KinematicMeasurement::orientation_cov, "Orientation covariance (3x3 matrix)")
        .def_readwrite("com_position_process_cov", &serow::KinematicMeasurement::com_position_process_cov, "Center of mass position process covariance (3x3 matrix)")
        .def_readwrite("com_linear_velocity_process_cov", &serow::KinematicMeasurement::com_linear_velocity_process_cov, "Center of mass linear velocity process covariance (3x3 matrix)")
        .def_readwrite("external_forces_process_cov", &serow::KinematicMeasurement::external_forces_process_cov, "External forces process covariance (3x3 matrix)")
        .def_readwrite("com_position_cov", &serow::KinematicMeasurement::com_position_cov, "Center of mass position covariance (3x3 matrix)")
        .def_readwrite("com_linear_acceleration_cov", &serow::KinematicMeasurement::com_linear_acceleration_cov, "Center of mass linear acceleration covariance (3x3 matrix)");

    // Binding for OdometryMeasurement
    py::class_<serow::OdometryMeasurement>(m, "OdometryMeasurement", "Represents odometry measurements")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::OdometryMeasurement::timestamp, "Timestamp of the measurement")
        .def_readwrite("base_position", &serow::OdometryMeasurement::base_position, "Base position (3D vector)")
        .def_property(
            "base_orientation",
            [](const serow::OdometryMeasurement& self) { return quaternion_to_numpy(self.base_orientation); },
            [](serow::OdometryMeasurement& self, const py::array_t<double>& arr) {
                self.base_orientation = numpy_to_quaternion(arr);
            },
            "Base orientation as a quaternion (w, x, y, z)")
        .def_readwrite("base_position_cov", &serow::OdometryMeasurement::base_position_cov, "Base position covariance (3x3 matrix)")
        .def_readwrite("base_orientation_cov", &serow::OdometryMeasurement::base_orientation_cov, "Base orientation covariance (3x3 matrix)");

    // Binding for BasePoseGroundTruth
    py::class_<serow::BasePoseGroundTruth>(m, "BasePoseGroundTruth", "Represents ground truth base pose")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::BasePoseGroundTruth::timestamp, "Timestamp of the measurement")
        .def_readwrite("position", &serow::BasePoseGroundTruth::position, "Position (3D vector)")
        .def_property(
            "orientation",
            [](const serow::BasePoseGroundTruth& self) { return quaternion_to_numpy(self.orientation); },
            [](serow::BasePoseGroundTruth& self, const py::array_t<double>& arr) {
                self.orientation = numpy_to_quaternion(arr);
            },
            "Orientation as a quaternion (w, x, y, z)");

    // Binding for ContactEKF
    py::class_<serow::ContactEKF>(m, "ContactEKF", "Extended Kalman Filter for humanoid robot state estimation")
        .def(py::init<>(), "Default constructor")
        .def("init", &serow::ContactEKF::init, 
             py::arg("state"), py::arg("contacts_frame"), py::arg("point_feet"), 
             py::arg("g"), py::arg("imu_rate"), py::arg("outlier_detection") = false,
             "Initializes the EKF with the initial robot state and parameters")
        .def("predict", &serow::ContactEKF::predict, 
             py::arg("state"), py::arg("imu"), py::arg("kin"),
             "Predicts the robot's state forward based on IMU and kinematic measurements")
        .def(
            "update",
            [](serow::ContactEKF& self, serow::BaseState& state,
               const serow::KinematicMeasurement& kin, py::object odom, py::object terrain) {
                if (odom.is_none()) {
                    if (terrain.is_none()) {
                        self.update(state, kin, std::nullopt, nullptr);
                    } else {
                        self.update(state, kin, std::nullopt,
                                    terrain.cast<std::shared_ptr<serow::TerrainElevation>>());
                    }
                } else {
                    if (terrain.is_none()) {
                        self.update(state, kin, odom.cast<serow::OdometryMeasurement>(), nullptr);
                    } else {
                        self.update(state, kin, odom.cast<serow::OdometryMeasurement>(),
                                    terrain.cast<std::shared_ptr<serow::TerrainElevation>>());
                    }
                }
            },
            py::arg("state"), py::arg("kin"), py::arg("odom") = py::none(),
            py::arg("terrain_estimator") = py::none(),
            "Updates the robot's state based on kinematic measurements, optional odometry, and terrain data")
        .def("set_action", &serow::ContactEKF::setAction, 
             py::arg("contact_frame"), py::arg("action"),
             "Sets the action for the EKF for a given contact frame")
        .def("get_contact_position_innovation", 
             [](serow::ContactEKF& self, const std::string& contact_frame) {
                 Eigen::Vector3d innovation = Eigen::Vector3d::Zero();
                 Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
                 bool success = self.getContactPositionInnovation(contact_frame, innovation, covariance);
                 return std::make_tuple(success, innovation, covariance);
             },
             py::arg("contact_frame"),
             "Gets the contact position innovation and covariance for a given contact frame")
        .def("get_contact_orientation_innovation", 
             [](serow::ContactEKF& self, const std::string& contact_frame) {
                 Eigen::Vector3d innovation = Eigen::Vector3d::Zero();
                 Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
                 bool success = self.getContactOrientationInnovation(contact_frame, innovation, covariance);
                 return std::make_tuple(success, innovation, covariance);
             },
             py::arg("contact_frame"),
             "Gets the contact orientation innovation and covariance for a given contact frame");

    // Binding for Serow
    py::class_<serow::Serow>(m, "Serow", "Main SEROW estimator class")
        .def(py::init<>(), "Default constructor")
        .def("initialize", &serow::Serow::initialize,
             py::arg("config"),
             py::arg("initial_state") = py::none(),
             "Initializes SEROW's configuration and internal state")
        .def("filter",
             [](serow::Serow& self, const serow::ImuMeasurement& imu,
                const std::map<std::string, serow::JointMeasurement>& joints,
                py::object ft, py::object odom, py::object contact_probabilities,
                py::object base_pose_ground_truth) {
                 // Convert Python objects to C++ optional types
                 std::optional<std::map<std::string, serow::ForceTorqueMeasurement>> ft_opt;
                 if (!ft.is_none()) {
                     ft_opt = ft.cast<std::map<std::string, serow::ForceTorqueMeasurement>>();
                 }

                 std::optional<serow::OdometryMeasurement> odom_opt;
                 if (!odom.is_none()) {
                     odom_opt = odom.cast<serow::OdometryMeasurement>();
                 }

                 std::optional<std::map<std::string, serow::ContactMeasurement>> contact_prob_opt;
                 if (!contact_probabilities.is_none()) {
                     contact_prob_opt = contact_probabilities.cast<std::map<std::string, serow::ContactMeasurement>>();
                 }

                 std::optional<serow::BasePoseGroundTruth> ground_truth_opt;
                 if (!base_pose_ground_truth.is_none()) {
                     ground_truth_opt = base_pose_ground_truth.cast<serow::BasePoseGroundTruth>();
                 }

                 self.filter(imu, joints, ft_opt, odom_opt, contact_prob_opt, ground_truth_opt);
             },
             py::arg("imu"),
             py::arg("joints"),
             py::arg("ft") = py::none(),
             py::arg("odom") = py::none(),
             py::arg("contact_probabilities") = py::none(),
             py::arg("base_pose_ground_truth") = py::none(),
             "Runs SEROW's estimator and updates the internal state")
        .def("get_base_state", &serow::Serow::getBaseState,
             py::arg("allow_invalid") = false,
             "Gets the base state of the robot")
        .def("get_contact_state", &serow::Serow::getContactState,
             py::arg("allow_invalid") = false,
             "Gets the contact state of the robot")
        .def("get_state", &serow::Serow::getState,
             py::arg("allow_invalid") = false,
             "Gets the complete state of the robot")
        .def("set_action", &serow::Serow::setAction,
             py::arg("contact_frame"),
             py::arg("action"),
             "Sets the action for a given contact frame")
        .def("is_initialized", &serow::Serow::isInitialized,
             "Returns true if SEROW is initialized");

    // Binding for CentroidalState
    py::class_<serow::CentroidalState>(m, "CentroidalState", "Represents the centroidal state of the robot")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::CentroidalState::timestamp, "Timestamp of the state")
        .def_readwrite("com_position", &serow::CentroidalState::com_position, "Center of mass position (3D vector)")
        .def_readwrite("com_linear_velocity", &serow::CentroidalState::com_linear_velocity, "Center of mass linear velocity (3D vector)")
        .def_readwrite("external_forces", &serow::CentroidalState::external_forces, "External forces at the CoM (3D vector)")
        .def_readwrite("cop_position", &serow::CentroidalState::cop_position, "Center of pressure position (3D vector)")
        .def_readwrite("com_linear_acceleration", &serow::CentroidalState::com_linear_acceleration, "Center of mass linear acceleration (3D vector)")
        .def_readwrite("angular_momentum", &serow::CentroidalState::angular_momentum, "Angular momentum around the CoM (3D vector)")
        .def_readwrite("angular_momentum_derivative", &serow::CentroidalState::angular_momentum_derivative, "Angular momentum derivative around the CoM (3D vector)")
        .def_readwrite("com_position_cov", &serow::CentroidalState::com_position_cov, "Center of mass position covariance (3x3 matrix)")
        .def_readwrite("com_linear_velocity_cov", &serow::CentroidalState::com_linear_velocity_cov, "Center of mass linear velocity covariance (3x3 matrix)")
        .def_readwrite("external_forces_cov", &serow::CentroidalState::external_forces_cov, "External forces covariance (3x3 matrix)");

    // Binding for JointState
    py::class_<serow::JointState>(m, "JointState", "Represents the joint state of the robot")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::JointState::timestamp, "Timestamp of the state")
        .def_readwrite("joints_position", &serow::JointState::joints_position, "Map of joint positions (string to double)")
        .def_readwrite("joints_velocity", &serow::JointState::joints_velocity, "Map of joint velocities (string to double)");

    // Binding for State
    py::class_<serow::State>(m, "State", "Represents the overall state of the robot")
        .def(py::init<>(), "Default constructor")
        .def(py::init<std::set<std::string>, bool>(), 
             py::arg("contacts_frame"), 
             py::arg("point_feet"),
             "Constructor with contact frames and point feet flag")
        .def("get_base_pose", &serow::State::getBasePose, "Returns the base pose as a rigid transformation")
        .def("get_base_position", &serow::State::getBasePosition, "Returns the base position")
        .def("get_base_orientation", [](const serow::State& self) {
            return quaternion_to_numpy(self.getBaseOrientation());
        }, "Returns the base orientation")
        .def("get_base_linear_velocity", &serow::State::getBaseLinearVelocity, "Returns the base linear velocity")
        .def("get_base_angular_velocity", &serow::State::getBaseAngularVelocity, "Returns the base angular velocity")
        .def("get_imu_linear_acceleration_bias", &serow::State::getImuLinearAccelerationBias, "Returns the IMU linear acceleration bias")
        .def("get_imu_angular_velocity_bias", &serow::State::getImuAngularVelocityBias, "Returns the IMU angular velocity bias")
        .def("get_contacts_frame", &serow::State::getContactsFrame, "Returns the active contact frame names")
        .def("get_contact_position", &serow::State::getContactPosition, "Returns the contact position for a given frame")
        .def("get_contact_orientation", &serow::State::getContactOrientation, "Returns the contact orientation for a given frame")
        .def("get_contact_pose", &serow::State::getContactPose, "Returns the contact pose for a given frame")
        .def("get_contact_status", &serow::State::getContactStatus, "Returns the contact status for a given frame")
        .def("get_contact_force", &serow::State::getContactForce, "Returns the contact force for a given frame")
        .def("get_foot_position", &serow::State::getFootPosition, "Returns the foot position for a given frame")
        .def("get_foot_orientation", &serow::State::getFootOrientation, "Returns the foot orientation for a given frame")
        .def("get_foot_pose", &serow::State::getFootPose, "Returns the foot pose for a given frame")
        .def("get_foot_linear_velocity", &serow::State::getFootLinearVelocity, "Returns the foot linear velocity for a given frame")
        .def("get_foot_angular_velocity", &serow::State::getFootAngularVelocity, "Returns the foot angular velocity for a given frame")
        .def("get_com_position", &serow::State::getCoMPosition, "Returns the center of mass position")
        .def("get_com_linear_velocity", &serow::State::getCoMLinearVelocity, "Returns the center of mass linear velocity")
        .def("get_com_external_forces", &serow::State::getCoMExternalForces, "Returns the center of mass external forces")
        .def("get_com_angular_momentum", &serow::State::getCoMAngularMomentum, "Returns the center of mass angular momentum")
        .def("get_com_angular_momentum_rate", &serow::State::getCoMAngularMomentumRate, "Returns the center of mass angular momentum rate")
        .def("get_com_linear_acceleration", &serow::State::getCoMLinearAcceleration, "Returns the center of mass linear acceleration")
        .def("get_cop_position", &serow::State::getCOPPosition, "Returns the center of pressure position")
        .def("get_base_pose_cov", &serow::State::getBasePoseCov, "Returns the base pose covariance")
        .def("get_base_velocity_cov", &serow::State::getBaseVelocityCov, "Returns the base velocity covariance")
        .def("get_base_position_cov", &serow::State::getBasePositionCov, "Returns the base position covariance")
        .def("get_base_orientation_cov", &serow::State::getBaseOrientationCov, "Returns the base orientation covariance")
        .def("get_base_linear_velocity_cov", &serow::State::getBaseLinearVelocityCov, "Returns the base linear velocity covariance")
        .def("get_base_angular_velocity_cov", &serow::State::getBaseAngularVelocityCov, "Returns the base angular velocity covariance")
        .def("get_imu_linear_acceleration_bias_cov", &serow::State::getImuLinearAccelerationBiasCov, "Returns the IMU linear acceleration bias covariance")
        .def("get_imu_angular_velocity_bias_cov", &serow::State::getImuAngularVelocityBiasCov, "Returns the IMU angular velocity bias covariance")
        .def("get_contact_pose_cov", &serow::State::getContactPoseCov, "Returns the contact pose covariance for a given frame")
        .def("get_contact_position_cov", &serow::State::getContactPositionCov, "Returns the contact position covariance for a given frame")
        .def("get_contact_orientation_cov", &serow::State::getContactOrientationCov, "Returns the contact orientation covariance for a given frame")
        .def("get_com_position_cov", &serow::State::getCoMPositionCov, "Returns the center of mass position covariance")
        .def("get_com_linear_velocity_cov", &serow::State::getCoMLinearVelocityCov, "Returns the center of mass linear velocity covariance")
        .def("get_com_external_forces_cov", &serow::State::getCoMExternalForcesCov, "Returns the center of mass external forces covariance")
        .def("get_mass", &serow::State::getMass, "Returns the mass of the robot")
        .def("get_num_leg_ee", &serow::State::getNumLegEE, "Returns the number of leg end-effectors")
        .def("is_point_feet", &serow::State::isPointFeet, "Returns whether the robot has point feet")
        .def("set_base_state", &serow::State::setBaseState, py::arg("base_state"), "Sets the base state of the robot")
        .def("set_contact_state", &serow::State::setContactState, py::arg("contact_state"), "Sets the contact state of the robot")
        .def("set_centroidal_state", &serow::State::setCentroidalState, py::arg("centroidal_state"), "Sets the centroidal state of the robot")
        .def("set_joint_state", &serow::State::setJointState, py::arg("joint_state"), "Sets the joint state of the robot");
}

