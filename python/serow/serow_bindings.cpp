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

    // Binding for ImuMeasurement
    py::class_<serow::ImuMeasurement>(m, "ImuMeasurement", "Represents IMU sensor measurements")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::ImuMeasurement::timestamp, "Timestamp of the measurement")
        .def_readwrite("angular_velocity", &serow::ImuMeasurement::angular_velocity, "Angular velocity (3D vector)")
        .def_readwrite("linear_acceleration", &serow::ImuMeasurement::linear_acceleration, "Linear acceleration (3D vector)")
        .def_property(
            "orientation",
            [](const serow::ImuMeasurement& self) { return quaternion_to_numpy(self.orientation); },
            [](serow::ImuMeasurement& self, const py::array_t<double>& arr) {
                self.orientation = numpy_to_quaternion(arr);
            },
            "Orientation as a quaternion (w, x, y, z)")
        .def_readwrite("angular_velocity_cov", &serow::ImuMeasurement::angular_velocity_cov, "Angular velocity covariance (3x3 matrix)")
        .def_readwrite("linear_acceleration_cov", &serow::ImuMeasurement::linear_acceleration_cov, "Linear acceleration covariance (3x3 matrix)")
        .def_readwrite("angular_velocity_bias_cov", &serow::ImuMeasurement::angular_velocity_bias_cov, "Angular velocity bias covariance (3x3 matrix)")
        .def_readwrite("linear_acceleration_bias_cov", &serow::ImuMeasurement::linear_acceleration_bias_cov, "Linear acceleration bias covariance (3x3 matrix)")
        .def_readwrite("angular_acceleration", &serow::ImuMeasurement::angular_acceleration, "Angular acceleration (3D vector)");

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
}
