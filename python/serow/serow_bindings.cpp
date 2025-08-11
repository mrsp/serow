#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <tuple>

#include "ContactEKF.hpp"
#include "LocalTerrainMapper.hpp"
#include "Measurement.hpp"
#include "Serow.hpp"
#include "State.hpp"
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
    m.doc() =
        "Python bindings for Serow library classes (Serow, ContactEKF, State, and Measurements)";

    // Binding for BaseState
    py::class_<serow::BaseState>(m, "BaseState", "Represents the state of a humanoid robot")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::BaseState::timestamp, "Timestamp of the state")
        .def_readwrite("base_position", &serow::BaseState::base_position,
                       "Base position (3D vector)")
        .def_property(
            "base_orientation",
            [](const serow::BaseState& self) { return quaternion_to_numpy(self.base_orientation); },
            [](serow::BaseState& self, const py::array_t<double>& arr) {
                self.base_orientation = numpy_to_quaternion(arr);
            },
            "Base orientation as a quaternion (w, x, y, z)")
        .def_readwrite("base_linear_velocity", &serow::BaseState::base_linear_velocity,
                       "Base linear velocity (3D vector)")
        .def_readwrite("base_angular_velocity", &serow::BaseState::base_angular_velocity,
                       "Base angular velocity (3D vector)")
        .def_readwrite("base_linear_acceleration", &serow::BaseState::base_linear_acceleration,
                       "Base linear acceleration (3D vector)")
        .def_readwrite("base_angular_acceleration", &serow::BaseState::base_angular_acceleration,
                       "Base angular acceleration (3D vector)")
        .def_readwrite("imu_angular_velocity_bias", &serow::BaseState::imu_angular_velocity_bias,
                       "IMU angular velocity bias (3D vector)")
        .def_readwrite("imu_linear_acceleration_bias",
                       &serow::BaseState::imu_linear_acceleration_bias,
                       "IMU linear acceleration bias (3D vector)")
        .def_readwrite("contacts_position", &serow::BaseState::contacts_position,
                       "Map of contact positions (string to 3D vector)")
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
            [](serow::BaseState& self,
               const std::map<std::string, py::array_t<double>>& orientations) {
                std::map<std::string, Eigen::Quaterniond> eigen_orientations;
                for (const auto& [name, arr] : orientations) {
                    eigen_orientations[name] = numpy_to_quaternion(arr);
                }
                self.contacts_orientation = eigen_orientations;
            },
            "Map of contact orientations (string to quaternion)")
        .def_readwrite("base_position_cov", &serow::BaseState::base_position_cov,
                       "Base position covariance (3x3 matrix)")
        .def_readwrite("base_orientation_cov", &serow::BaseState::base_orientation_cov,
                       "Base orientation covariance (3x3 matrix)")
        .def_readwrite("base_linear_velocity_cov", &serow::BaseState::base_linear_velocity_cov,
                       "Base linear velocity covariance (3x3 matrix)")
        .def_readwrite("base_angular_velocity_cov", &serow::BaseState::base_angular_velocity_cov,
                       "Base angular velocity covariance (3x3 matrix)")
        .def_readwrite("imu_angular_velocity_bias_cov",
                       &serow::BaseState::imu_angular_velocity_bias_cov,
                       "IMU angular velocity bias covariance (3x3 matrix)")
        .def_readwrite("imu_linear_acceleration_bias_cov",
                       &serow::BaseState::imu_linear_acceleration_bias_cov,
                       "IMU linear acceleration bias covariance (3x3 matrix)")
        .def_readwrite("contacts_position_cov", &serow::BaseState::contacts_position_cov,
                       "Map of contact position covariances (string to 3x3 matrix)")
        .def_readwrite("contacts_orientation_cov", &serow::BaseState::contacts_orientation_cov,
                       "Map of contact orientation covariances (string to 3x3 matrix)")
        .def_readwrite("feet_position", &serow::BaseState::feet_position,
                       "Map of feet positions (string to 3D vector)")
        .def_property(
            "feet_orientation",
            [](const serow::BaseState& self) {
                std::map<std::string, py::array_t<double>> result;
                for (const auto& [name, quat] : self.feet_orientation) {
                    result[name] = quaternion_to_numpy(quat);
                }
                return result;
            },
            [](serow::BaseState& self,
               const std::map<std::string, py::array_t<double>>& orientations) {
                std::map<std::string, Eigen::Quaterniond> eigen_orientations;
                for (const auto& [name, arr] : orientations) {
                    eigen_orientations[name] = numpy_to_quaternion(arr);
                }
                self.feet_orientation = eigen_orientations;
            },
            "Map of feet orientations (string to quaternion)")
        .def_readwrite("feet_linear_velocity", &serow::BaseState::feet_linear_velocity,
                       "Map of feet linear velocities (string to 3D vector)")
        .def_readwrite("feet_angular_velocity", &serow::BaseState::feet_angular_velocity,
                       "Map of feet angular velocities (string to 3D vector)")
        // Add pickle support
        .def(py::pickle(
            [](const serow::BaseState& state) {  // __getstate__
                // Handle optional contacts_orientation
                std::map<std::string, py::array_t<double>> contacts_orientation_serialized;
                if (state.contacts_orientation) {
                    for (const auto& [name, quat] : *state.contacts_orientation) {
                        contacts_orientation_serialized[name] = quaternion_to_numpy(quat);
                    }
                }

                // Convert feet_orientation to numpy arrays
                std::map<std::string, py::array_t<double>> feet_orientation_serialized;
                for (const auto& [name, quat] : state.feet_orientation) {
                    feet_orientation_serialized[name] = quaternion_to_numpy(quat);
                }

                return py::make_tuple(
                    state.timestamp, state.base_position,
                    quaternion_to_numpy(state.base_orientation), state.base_linear_velocity,
                    state.base_angular_velocity, state.base_linear_acceleration,
                    state.base_angular_acceleration, state.imu_angular_velocity_bias,
                    state.imu_linear_acceleration_bias, state.contacts_position,
                    contacts_orientation_serialized, state.base_position_cov,
                    state.base_orientation_cov, state.base_linear_velocity_cov,
                    state.base_angular_velocity_cov, state.imu_angular_velocity_bias_cov,
                    state.imu_linear_acceleration_bias_cov, state.contacts_position_cov,
                    state.contacts_orientation_cov, state.feet_position,
                    feet_orientation_serialized, state.feet_linear_velocity,
                    state.feet_angular_velocity);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 23)
                    throw std::runtime_error("Invalid state for BaseState!");

                serow::BaseState state;
                state.timestamp = t[0].cast<double>();
                state.base_position = t[1].cast<decltype(state.base_position)>();
                state.base_orientation = numpy_to_quaternion(t[2].cast<py::array_t<double>>());
                state.base_linear_velocity = t[3].cast<decltype(state.base_linear_velocity)>();
                state.base_angular_velocity = t[4].cast<decltype(state.base_angular_velocity)>();
                state.base_linear_acceleration =
                    t[5].cast<decltype(state.base_linear_acceleration)>();
                state.base_angular_acceleration =
                    t[6].cast<decltype(state.base_angular_acceleration)>();
                state.imu_angular_velocity_bias =
                    t[7].cast<decltype(state.imu_angular_velocity_bias)>();
                state.imu_linear_acceleration_bias =
                    t[8].cast<decltype(state.imu_linear_acceleration_bias)>();
                state.contacts_position = t[9].cast<decltype(state.contacts_position)>();

                // Handle optional contacts_orientation
                auto contacts_orientation_serialized =
                    t[10].cast<std::map<std::string, py::array_t<double>>>();
                if (!contacts_orientation_serialized.empty()) {
                    std::map<std::string, Eigen::Quaterniond> contacts_orientation;
                    for (const auto& [name, arr] : contacts_orientation_serialized) {
                        contacts_orientation[name] = numpy_to_quaternion(arr);
                    }
                    state.contacts_orientation = contacts_orientation;
                }

                state.base_position_cov = t[11].cast<decltype(state.base_position_cov)>();
                state.base_orientation_cov = t[12].cast<decltype(state.base_orientation_cov)>();
                state.base_linear_velocity_cov =
                    t[13].cast<decltype(state.base_linear_velocity_cov)>();
                state.base_angular_velocity_cov =
                    t[14].cast<decltype(state.base_angular_velocity_cov)>();
                state.imu_angular_velocity_bias_cov =
                    t[15].cast<decltype(state.imu_angular_velocity_bias_cov)>();
                state.imu_linear_acceleration_bias_cov =
                    t[16].cast<decltype(state.imu_linear_acceleration_bias_cov)>();
                state.contacts_position_cov = t[17].cast<decltype(state.contacts_position_cov)>();
                state.contacts_orientation_cov =
                    t[18].cast<decltype(state.contacts_orientation_cov)>();
                state.feet_position = t[19].cast<decltype(state.feet_position)>();

                // Convert feet_orientation from numpy arrays
                auto feet_orientation_serialized =
                    t[20].cast<std::map<std::string, py::array_t<double>>>();
                for (const auto& [name, arr] : feet_orientation_serialized) {
                    state.feet_orientation[name] = numpy_to_quaternion(arr);
                }

                state.feet_linear_velocity = t[21].cast<decltype(state.feet_linear_velocity)>();
                state.feet_angular_velocity = t[22].cast<decltype(state.feet_angular_velocity)>();

                return state;
            }));

    // Binding for ContactState
    py::class_<serow::ContactState>(m, "ContactState",
                                    "Represents the contact state of a humanoid robot")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::ContactState::timestamp,
                       "Timestamp of the contact state")
        .def_readwrite("contacts_status", &serow::ContactState::contacts_status,
                       "Map of contact statuses (string to bool)")
        .def_readwrite("contacts_probability", &serow::ContactState::contacts_probability,
                       "Map of contact probabilities (string to double)")
        .def_readwrite("contacts_force", &serow::ContactState::contacts_force,
                       "Map of contact forces (string to 3D vector)")
        .def_readwrite("contacts_torque", &serow::ContactState::contacts_torque,
                       "Map of contact torques (string to 3D vector)")
        // Add pickle support
        .def(py::pickle(
            [](const serow::ContactState& contact) {  // __getstate__
                return py::make_tuple(contact.timestamp, contact.contacts_status,
                                      contact.contacts_probability, contact.contacts_force,
                                      contact.contacts_torque);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 5)
                    throw std::runtime_error("Invalid state for ContactState!");

                serow::ContactState contact;
                contact.timestamp = t[0].cast<double>();
                contact.contacts_status = t[1].cast<decltype(contact.contacts_status)>();
                contact.contacts_probability = t[2].cast<decltype(contact.contacts_probability)>();
                contact.contacts_force = t[3].cast<decltype(contact.contacts_force)>();
                contact.contacts_torque = t[4].cast<decltype(contact.contacts_torque)>();
                return contact;
            }));

    // Binding for JointMeasurement
    py::class_<serow::JointMeasurement>(m, "JointMeasurement", "Represents a joint measurement")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::JointMeasurement::timestamp,
                       "Timestamp of the measurement (s)")
        .def_readwrite("position", &serow::JointMeasurement::position,
                       "Joint position measurement (rad)")
        .def_readwrite("velocity", &serow::JointMeasurement::velocity,
                       "Optional joint velocity measurement (rad/s)")
        // Add pickle support
        .def(py::pickle(
            [](const serow::JointMeasurement& joint) {  // __getstate__
                return py::make_tuple(joint.timestamp, joint.position, joint.velocity);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state for JointMeasurement!");

                serow::JointMeasurement joint;
                joint.timestamp = t[0].cast<double>();
                joint.position = t[1].cast<double>();
                joint.velocity = t[2].cast<decltype(joint.velocity)>();

                return joint;
            }));

    // Binding for ImuMeasurement
    py::class_<serow::ImuMeasurement>(m, "ImuMeasurement", "Represents IMU measurements")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::ImuMeasurement::timestamp,
                       "Timestamp of the measurement (s)")
        .def_readwrite("linear_acceleration", &serow::ImuMeasurement::linear_acceleration,
                       "Linear acceleration measured by IMU (m/s^2)")
        .def_readwrite("angular_velocity", &serow::ImuMeasurement::angular_velocity,
                       "Angular velocity measured by IMU (rad/s)")
        .def_property(
            "orientation",
            [](const serow::ImuMeasurement& self) { return quaternion_to_numpy(self.orientation); },
            [](serow::ImuMeasurement& self, const py::array_t<double>& arr) {
                self.orientation = numpy_to_quaternion(arr);
            },
            "Orientation measured by IMU (quaternion)")
        .def_readwrite("angular_acceleration", &serow::ImuMeasurement::angular_acceleration,
                       "Angular acceleration measured by IMU (rad/s^2)")
        .def_readwrite("angular_velocity_cov", &serow::ImuMeasurement::angular_velocity_cov,
                       "Covariance matrix of angular velocity (rad^2/s^2)")
        .def_readwrite("linear_acceleration_cov", &serow::ImuMeasurement::linear_acceleration_cov,
                       "Covariance matrix of linear acceleration (m^2/s^4)")
        .def_readwrite("angular_velocity_bias_cov",
                       &serow::ImuMeasurement::angular_velocity_bias_cov,
                       "Covariance matrix of angular velocity bias (rad^2/s^2)")
        .def_readwrite("linear_acceleration_bias_cov",
                       &serow::ImuMeasurement::linear_acceleration_bias_cov,
                       "Covariance matrix of linear acceleration bias (m^2/s^4)")
        .def_readwrite("orientation_cov", &serow::ImuMeasurement::orientation_cov,
                       "Covariance matrix of orientation (rad^2)")
        // Add pickle support
        .def(py::pickle(
            [](const serow::ImuMeasurement& imu) {  // __getstate__
                return py::make_tuple(
                    imu.timestamp, imu.linear_acceleration, imu.angular_velocity,
                    quaternion_to_numpy(imu.orientation),  // Convert quaternion to numpy array
                    imu.angular_acceleration, imu.angular_velocity_cov, imu.linear_acceleration_cov,
                    imu.angular_velocity_bias_cov, imu.linear_acceleration_bias_cov,
                    imu.orientation_cov);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 10)
                    throw std::runtime_error("Invalid state for ImuMeasurement!");

                serow::ImuMeasurement imu;
                imu.timestamp = t[0].cast<double>();
                imu.linear_acceleration = t[1].cast<decltype(imu.linear_acceleration)>();
                imu.angular_velocity = t[2].cast<decltype(imu.angular_velocity)>();
                imu.orientation = numpy_to_quaternion(t[3].cast<py::array_t<double>>());
                imu.angular_acceleration = t[4].cast<decltype(imu.angular_acceleration)>();
                imu.angular_velocity_cov = t[5].cast<decltype(imu.angular_velocity_cov)>();
                imu.linear_acceleration_cov = t[6].cast<decltype(imu.linear_acceleration_cov)>();
                imu.angular_velocity_bias_cov =
                    t[7].cast<decltype(imu.angular_velocity_bias_cov)>();
                imu.linear_acceleration_bias_cov =
                    t[8].cast<decltype(imu.linear_acceleration_bias_cov)>();
                imu.orientation_cov = t[9].cast<decltype(imu.orientation_cov)>();

                return imu;
            }));

    // Binding for ForceTorqueMeasurement
    py::class_<serow::ForceTorqueMeasurement>(m, "ForceTorqueMeasurement",
                                              "Represents force-torque sensor measurements")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::ForceTorqueMeasurement::timestamp,
                       "Timestamp of the measurement (s)")
        .def_readwrite("force", &serow::ForceTorqueMeasurement::force,
                       "Force measured by force-torque sensor (N)")
        .def_readwrite("cop", &serow::ForceTorqueMeasurement::cop,
                       "Center of pressure (COP) measured by force-torque sensor (m)")
        .def_readwrite("torque", &serow::ForceTorqueMeasurement::torque,
                       "Optional torque measured by force-torque sensor (Nm)")
        // Add pickle support
        .def(py::pickle(
            [](const serow::ForceTorqueMeasurement& ft) {  // __getstate__
                return py::make_tuple(ft.timestamp, ft.force, ft.cop, ft.torque);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state for ForceTorqueMeasurement!");

                serow::ForceTorqueMeasurement ft;
                ft.timestamp = t[0].cast<double>();
                ft.force = t[1].cast<decltype(ft.force)>();
                ft.cop = t[2].cast<decltype(ft.cop)>();
                ft.torque = t[3].cast<decltype(ft.torque)>();

                return ft;
            }));

    // Binding for KinematicMeasurement
    py::class_<serow::KinematicMeasurement>(m, "KinematicMeasurement",
                                            "Represents kinematic measurements")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::KinematicMeasurement::timestamp,
                       "Timestamp of the measurement")
        .def_readwrite("base_linear_velocity", &serow::KinematicMeasurement::base_linear_velocity,
                       "Base linear velocity (3D vector)")
        .def_readwrite("contacts_status", &serow::KinematicMeasurement::contacts_status,
                       "Map of contact statuses (string to bool)")
        .def_readwrite("contacts_probability", &serow::KinematicMeasurement::contacts_probability,
                       "Map of contact probabilities (string to double)")
        .def_readwrite("is_new_contact", &serow::KinematicMeasurement::is_new_contact,
                       "Map of new contact statuses (string to bool)")
        .def_readwrite("contacts_position", &serow::KinematicMeasurement::contacts_position,
                       "Map of contact positions (string to 3D vector)")
        .def_readwrite("base_to_foot_positions",
                       &serow::KinematicMeasurement::base_to_foot_positions,
                       "Map of base-to-foot positions (string to 3D vector)")
        .def_property(
            "base_to_foot_orientations",
            [](const serow::KinematicMeasurement& self) {
                std::map<std::string, py::array_t<double>> result;
                for (const auto& [name, quat] : self.base_to_foot_orientations) {
                    result[name] = quaternion_to_numpy(quat);
                }
                return result;
            },
            [](serow::KinematicMeasurement& self,
               const std::map<std::string, py::array_t<double>>& orientations) {
                std::map<std::string, Eigen::Quaterniond> eigen_orientations;
                for (const auto& [name, arr] : orientations) {
                    eigen_orientations[name] = numpy_to_quaternion(arr);
                }
                self.base_to_foot_orientations = eigen_orientations;
            },
            "Map of base-to-foot orientations (string to quaternion)")
        .def_readwrite("base_to_foot_linear_velocities",
                       &serow::KinematicMeasurement::base_to_foot_linear_velocities,
                       "Map of base-to-foot linear velocities (string to 3D vector)")
        .def_readwrite("base_to_foot_angular_velocities",
                       &serow::KinematicMeasurement::base_to_foot_angular_velocities,
                       "Map of base-to-foot angular velocities (string to 3D vector)")
        .def_readwrite("contacts_position_noise",
                       &serow::KinematicMeasurement::contacts_position_noise,
                       "Map of contact position noise (string to 3x3 matrix)")
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
            [](serow::KinematicMeasurement& self,
               const std::map<std::string, py::array_t<double>>& orientations) {
                std::map<std::string, Eigen::Quaterniond> eigen_orientations;
                for (const auto& [name, arr] : orientations) {
                    eigen_orientations[name] = numpy_to_quaternion(arr);
                }
                self.contacts_orientation = eigen_orientations;
            },
            "Map of contact orientations (string to quaternion)")
        .def_readwrite("contacts_orientation_noise",
                       &serow::KinematicMeasurement::contacts_orientation_noise,
                       "Map of contact orientation noise (string to 3x3 matrix)")
        .def_readwrite("com_angular_momentum_derivative",
                       &serow::KinematicMeasurement::com_angular_momentum_derivative,
                       "Center of mass angular momentum derivative (3D vector)")
        .def_readwrite("com_position", &serow::KinematicMeasurement::com_position,
                       "Center of mass position (3D vector)")
        .def_readwrite("com_linear_acceleration",
                       &serow::KinematicMeasurement::com_linear_acceleration,
                       "Center of mass linear acceleration (3D vector)")
        .def_readwrite("position_slip_cov", &serow::KinematicMeasurement::position_slip_cov,
                       "Position slip covariance (3x3 matrix)")
        .def_readwrite("orientation_slip_cov", &serow::KinematicMeasurement::orientation_slip_cov,
                       "Orientation slip covariance (3x3 matrix)")
        .def_readwrite("position_cov", &serow::KinematicMeasurement::position_cov,
                       "Position covariance (3x3 matrix)")
        .def_readwrite("orientation_cov", &serow::KinematicMeasurement::orientation_cov,
                       "Orientation covariance (3x3 matrix)")
        .def_readwrite("com_position_process_cov",
                       &serow::KinematicMeasurement::com_position_process_cov,
                       "Center of mass position process covariance (3x3 matrix)")
        .def_readwrite("com_linear_velocity_process_cov",
                       &serow::KinematicMeasurement::com_linear_velocity_process_cov,
                       "Center of mass linear velocity process covariance (3x3 matrix)")
        .def_readwrite("external_forces_process_cov",
                       &serow::KinematicMeasurement::external_forces_process_cov,
                       "External forces process covariance (3x3 matrix)")
        .def_readwrite("com_position_cov", &serow::KinematicMeasurement::com_position_cov,
                       "Center of mass position covariance (3x3 matrix)")
        .def_readwrite("com_linear_acceleration_cov",
                       &serow::KinematicMeasurement::com_linear_acceleration_cov,
                       "Center of mass linear acceleration covariance (3x3 matrix)")
        // Add pickle support
        .def(py::pickle(
            [](const serow::KinematicMeasurement& kin) {  // __getstate__
                // Handle optional contacts_orientation
                std::map<std::string, py::array_t<double>> contacts_orientation_serialized;
                if (kin.contacts_orientation) {
                    for (const auto& [name, quat] : *kin.contacts_orientation) {
                        contacts_orientation_serialized[name] = quaternion_to_numpy(quat);
                    }
                }

                // Convert base_to_foot_orientations to numpy arrays
                std::map<std::string, py::array_t<double>> base_to_foot_orientations_serialized;
                for (const auto& [name, quat] : kin.base_to_foot_orientations) {
                    base_to_foot_orientations_serialized[name] = quaternion_to_numpy(quat);
                }

                return py::make_tuple(
                    kin.timestamp, kin.base_linear_velocity, kin.contacts_status,
                    kin.contacts_probability, kin.is_new_contact, kin.contacts_position,
                    kin.base_to_foot_positions, base_to_foot_orientations_serialized,
                    kin.base_to_foot_linear_velocities, kin.base_to_foot_angular_velocities,
                    kin.contacts_position_noise, contacts_orientation_serialized,
                    kin.contacts_orientation_noise, kin.com_angular_momentum_derivative,
                    kin.com_position, kin.com_linear_acceleration, kin.base_linear_velocity_cov,
                    kin.position_slip_cov, kin.orientation_slip_cov, kin.position_cov,
                    kin.orientation_cov, kin.com_position_process_cov,
                    kin.com_linear_velocity_process_cov, kin.external_forces_process_cov,
                    kin.com_position_cov, kin.com_linear_acceleration_cov);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 26)
                    throw std::runtime_error("Invalid state for KinematicMeasurement!");

                serow::KinematicMeasurement kin;
                kin.timestamp = t[0].cast<double>();
                kin.base_linear_velocity = t[1].cast<decltype(kin.base_linear_velocity)>();
                kin.contacts_status = t[2].cast<decltype(kin.contacts_status)>();
                kin.contacts_probability = t[3].cast<decltype(kin.contacts_probability)>();
                kin.is_new_contact = t[4].cast<decltype(kin.is_new_contact)>();
                kin.contacts_position = t[5].cast<decltype(kin.contacts_position)>();
                kin.base_to_foot_positions = t[6].cast<decltype(kin.base_to_foot_positions)>();

                // Convert base_to_foot_orientations from numpy arrays
                auto base_to_foot_orientations_serialized =
                    t[7].cast<std::map<std::string, py::array_t<double>>>();
                for (const auto& [name, arr] : base_to_foot_orientations_serialized) {
                    kin.base_to_foot_orientations[name] = numpy_to_quaternion(arr);
                }

                kin.base_to_foot_linear_velocities =
                    t[8].cast<decltype(kin.base_to_foot_linear_velocities)>();
                kin.base_to_foot_angular_velocities =
                    t[9].cast<decltype(kin.base_to_foot_angular_velocities)>();
                kin.contacts_position_noise = t[10].cast<decltype(kin.contacts_position_noise)>();

                // Handle optional contacts_orientation
                auto contacts_orientation_serialized =
                    t[11].cast<std::map<std::string, py::array_t<double>>>();
                if (!contacts_orientation_serialized.empty()) {
                    std::map<std::string, Eigen::Quaterniond> contacts_orientation;
                    for (const auto& [name, arr] : contacts_orientation_serialized) {
                        contacts_orientation[name] = numpy_to_quaternion(arr);
                    }
                    kin.contacts_orientation = contacts_orientation;
                }

                kin.contacts_orientation_noise =
                    t[12].cast<decltype(kin.contacts_orientation_noise)>();
                kin.com_angular_momentum_derivative =
                    t[13].cast<decltype(kin.com_angular_momentum_derivative)>();
                kin.com_position = t[14].cast<decltype(kin.com_position)>();
                kin.com_linear_acceleration = t[15].cast<decltype(kin.com_linear_acceleration)>();
                kin.base_linear_velocity_cov = t[16].cast<decltype(kin.base_linear_velocity_cov)>();
                kin.position_slip_cov = t[17].cast<decltype(kin.position_slip_cov)>();
                kin.orientation_slip_cov = t[18].cast<decltype(kin.orientation_slip_cov)>();
                kin.position_cov = t[19].cast<decltype(kin.position_cov)>();
                kin.orientation_cov = t[20].cast<decltype(kin.orientation_cov)>();
                kin.com_position_process_cov = t[21].cast<decltype(kin.com_position_process_cov)>();
                kin.com_linear_velocity_process_cov =
                    t[22].cast<decltype(kin.com_linear_velocity_process_cov)>();
                kin.external_forces_process_cov =
                    t[23].cast<decltype(kin.external_forces_process_cov)>();
                kin.com_position_cov = t[24].cast<decltype(kin.com_position_cov)>();
                kin.com_linear_acceleration_cov =
                    t[25].cast<decltype(kin.com_linear_acceleration_cov)>();

                return kin;
            }));

    // Binding for OdometryMeasurement
    py::class_<serow::OdometryMeasurement>(m, "OdometryMeasurement",
                                           "Represents odometry measurements")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::OdometryMeasurement::timestamp,
                       "Timestamp of the measurement")
        .def_readwrite("base_position", &serow::OdometryMeasurement::base_position,
                       "Base position (3D vector)")
        .def_property(
            "base_orientation",
            [](const serow::OdometryMeasurement& self) {
                return quaternion_to_numpy(self.base_orientation);
            },
            [](serow::OdometryMeasurement& self, const py::array_t<double>& arr) {
                self.base_orientation = numpy_to_quaternion(arr);
            },
            "Base orientation as a quaternion (w, x, y, z)")
        .def_readwrite("base_position_cov", &serow::OdometryMeasurement::base_position_cov,
                       "Base position covariance (3x3 matrix)")
        .def_readwrite("base_orientation_cov", &serow::OdometryMeasurement::base_orientation_cov,
                       "Base orientation covariance (3x3 matrix)")
        // Add pickle support
        .def(py::pickle(
            [](const serow::OdometryMeasurement& odom) {  // __getstate__
                return py::make_tuple(odom.timestamp, odom.base_position,
                                      quaternion_to_numpy(odom.base_orientation),
                                      odom.base_position_cov, odom.base_orientation_cov);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 5)
                    throw std::runtime_error("Invalid state for OdometryMeasurement!");

                serow::OdometryMeasurement odom;
                odom.timestamp = t[0].cast<double>();
                odom.base_position = t[1].cast<decltype(odom.base_position)>();
                odom.base_orientation = numpy_to_quaternion(t[2].cast<py::array_t<double>>());
                odom.base_position_cov = t[3].cast<decltype(odom.base_position_cov)>();
                odom.base_orientation_cov = t[4].cast<decltype(odom.base_orientation_cov)>();

                return odom;
            }));

    // Binding for BasePoseGroundTruth
    py::class_<serow::BasePoseGroundTruth>(m, "BasePoseGroundTruth",
                                           "Represents ground truth base pose")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::BasePoseGroundTruth::timestamp,
                       "Timestamp of the measurement")
        .def_readwrite("position", &serow::BasePoseGroundTruth::position, "Position (3D vector)")
        .def_property(
            "orientation",
            [](const serow::BasePoseGroundTruth& self) {
                return quaternion_to_numpy(self.orientation);
            },
            [](serow::BasePoseGroundTruth& self, const py::array_t<double>& arr) {
                self.orientation = numpy_to_quaternion(arr);
            },
            "Orientation as a quaternion (w, x, y, z)")
        // Add pickle support
        .def(py::pickle(
            [](const serow::BasePoseGroundTruth& base_pose) {  // __getstate__
                return py::make_tuple(base_pose.timestamp, base_pose.position,
                                      quaternion_to_numpy(base_pose.orientation));
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state for BasePoseGroundTruth!");

                serow::BasePoseGroundTruth base_pose;
                base_pose.timestamp = t[0].cast<double>();
                base_pose.position = t[1].cast<decltype(base_pose.position)>();
                base_pose.orientation = numpy_to_quaternion(t[2].cast<py::array_t<double>>());

                return base_pose;
            }));

    // Binding for ContactEKF
    py::class_<serow::ContactEKF>(m, "ContactEKF",
                                  "Extended Kalman Filter for humanoid robot state estimation")
        .def(py::init<>(), "Default constructor")
        .def("init", &serow::ContactEKF::init, py::arg("state"), py::arg("contacts_frame"),
             py::arg("point_feet"), py::arg("g"), py::arg("imu_rate"),
             py::arg("outlier_detection") = false,
             "Initializes the EKF with the initial robot state and parameters")
        .def("predict", &serow::ContactEKF::predict, py::arg("state"), py::arg("imu"),
             py::arg("kin"),
             "Predicts the robot's state forward based on IMU and kinematic measurements")
        .def(
            "update",
            [](serow::ContactEKF& self, serow::BaseState& state, const serow::ImuMeasurement& imu,
               const serow::KinematicMeasurement& kin, py::object odom, py::object terrain) {
                if (odom.is_none()) {
                    if (terrain.is_none()) {
                        self.update(state, imu, kin, std::nullopt, nullptr);
                    } else {
                        self.update(state, imu, kin, std::nullopt,
                                    terrain.cast<std::shared_ptr<serow::TerrainElevation>>());
                    }
                } else {
                    if (terrain.is_none()) {
                        self.update(state, imu, kin, odom.cast<serow::OdometryMeasurement>(),
                                    nullptr);
                    } else {
                        self.update(state, imu, kin, odom.cast<serow::OdometryMeasurement>(),
                                    terrain.cast<std::shared_ptr<serow::TerrainElevation>>());
                    }
                }
            },
            py::arg("state"), py::arg("imu"), py::arg("kin"), py::arg("odom") = py::none(),
            py::arg("terrain_estimator") = py::none(),
            "Updates the robot's state based on kinematic measurements, optional odometry, and "
            "terrain data");

    // Binding for Serow
    py::class_<serow::Serow>(m, "Serow", "Main SEROW estimator class")
        .def(py::init<>(), "Default constructor")
        .def("initialize", &serow::Serow::initialize, py::arg("config"),
             "Initializes SEROW's configuration and internal state")
        .def(
            "filter",
            [](serow::Serow& self, const serow::ImuMeasurement& imu,
               const std::map<std::string, serow::JointMeasurement>& joints, py::object ft,
               py::object odom, py::object contact_probabilities,
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

                std::optional<std::map<std::string, double>> contact_prob_opt;
                if (!contact_probabilities.is_none()) {
                    contact_prob_opt = contact_probabilities.cast<std::map<std::string, double>>();
                }

                std::optional<serow::BasePoseGroundTruth> ground_truth_opt;
                if (!base_pose_ground_truth.is_none()) {
                    ground_truth_opt = base_pose_ground_truth.cast<serow::BasePoseGroundTruth>();
                }

                return self.filter(imu, joints, ft_opt, odom_opt, contact_prob_opt,
                                   ground_truth_opt);
            },
            py::arg("imu"), py::arg("joints"), py::arg("ft") = py::none(),
            py::arg("odom") = py::none(), py::arg("contact_probabilities") = py::none(),
            py::arg("base_pose_ground_truth") = py::none(),
            "Runs SEROW's estimator and updates the internal state")
        .def("get_base_state", &serow::Serow::getBaseState, py::arg("allow_invalid") = false,
             "Gets the base state of the robot")
        .def("get_contact_state", &serow::Serow::getContactState, py::arg("allow_invalid") = false,
             "Gets the contact state of the robot")
        .def("get_state", &serow::Serow::getState, py::arg("allow_invalid") = false,
             "Gets the complete state of the robot")
        .def("is_initialized", &serow::Serow::isInitialized, "Returns true if SEROW is initialized")
        .def("set_state", &serow::Serow::setState, py::arg("state"), "Sets the state of the robot");

    // Binding for CentroidalState
    py::class_<serow::CentroidalState>(m, "CentroidalState",
                                       "Represents the centroidal state of the robot")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::CentroidalState::timestamp, "Timestamp of the state")
        .def_readwrite("com_position", &serow::CentroidalState::com_position,
                       "Center of mass position (3D vector)")
        .def_readwrite("com_linear_velocity", &serow::CentroidalState::com_linear_velocity,
                       "Center of mass linear velocity (3D vector)")
        .def_readwrite("external_forces", &serow::CentroidalState::external_forces,
                       "External forces at the CoM (3D vector)")
        .def_readwrite("cop_position", &serow::CentroidalState::cop_position,
                       "Center of pressure position (3D vector)")
        .def_readwrite("com_linear_acceleration", &serow::CentroidalState::com_linear_acceleration,
                       "Center of mass linear acceleration (3D vector)")
        .def_readwrite("angular_momentum", &serow::CentroidalState::angular_momentum,
                       "Angular momentum around the CoM (3D vector)")
        .def_readwrite("angular_momentum_derivative",
                       &serow::CentroidalState::angular_momentum_derivative,
                       "Angular momentum derivative around the CoM (3D vector)")
        .def_readwrite("com_position_cov", &serow::CentroidalState::com_position_cov,
                       "Center of mass position covariance (3x3 matrix)")
        .def_readwrite("com_linear_velocity_cov", &serow::CentroidalState::com_linear_velocity_cov,
                       "Center of mass linear velocity covariance (3x3 matrix)")
        .def_readwrite("external_forces_cov", &serow::CentroidalState::external_forces_cov,
                       "External forces covariance (3x3 matrix)");

    // Binding for JointState
    py::class_<serow::JointState>(m, "JointState", "Represents the joint state of the robot")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::JointState::timestamp, "Timestamp of the state")
        .def_readwrite("joints_position", &serow::JointState::joints_position,
                       "Map of joint positions (string to double)")
        .def_readwrite("joints_velocity", &serow::JointState::joints_velocity,
                       "Map of joint velocities (string to double)")
        // Add pickle support
        .def(py::pickle(
            [](const serow::JointState& joint) {  // __getstate__
                return py::make_tuple(joint.timestamp, joint.joints_position,
                                      joint.joints_velocity);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state for JointState!");

                serow::JointState joint;
                joint.timestamp = t[0].cast<double>();
                joint.joints_position = t[1].cast<decltype(joint.joints_position)>();
                joint.joints_velocity = t[2].cast<decltype(joint.joints_velocity)>();

                return joint;
            }));

    // Binding for State
    py::class_<serow::State>(m, "State", "Represents the overall state of the robot")
        .def(py::init<>(), "Default constructor")
        .def(py::init<std::set<std::string>, bool, std::string>(), py::arg("contacts_frame"),
             py::arg("point_feet"), py::arg("base_frame") = "base_link",
             "Constructor with contact frames, point feet flag, and base frame")
        .def("get_base_pose", &serow::State::getBasePose,
             "Returns the base pose as a rigid transformation")
        .def("get_base_position", &serow::State::getBasePosition, "Returns the base position")
        .def(
            "get_base_orientation",
            [](const serow::State& self) { return quaternion_to_numpy(self.getBaseOrientation()); },
            "Returns the base orientation")
        .def("get_base_linear_velocity", &serow::State::getBaseLinearVelocity,
             "Returns the base linear velocity")
        .def("get_base_angular_velocity", &serow::State::getBaseAngularVelocity,
             "Returns the base angular velocity")
        .def("get_imu_linear_acceleration_bias", &serow::State::getImuLinearAccelerationBias,
             "Returns the IMU linear acceleration bias")
        .def("get_imu_angular_velocity_bias", &serow::State::getImuAngularVelocityBias,
             "Returns the IMU angular velocity bias")
        .def("get_contacts_frame", &serow::State::getContactsFrame,
             "Returns the active contact frame names")
        .def("get_contact_position", &serow::State::getContactPosition,
             "Returns the contact position for a given frame")
        .def("get_contact_orientation", &serow::State::getContactOrientation,
             "Returns the contact orientation for a given frame")
        .def("get_contact_pose", &serow::State::getContactPose,
             "Returns the contact pose for a given frame")
        .def("get_contact_status", &serow::State::getContactStatus,
             "Returns the contact status for a given frame")
        .def("get_contact_force", &serow::State::getContactForce,
             "Returns the contact force for a given frame")
        .def("get_foot_position", &serow::State::getFootPosition,
             "Returns the foot position for a given frame")
        .def("get_foot_orientation", &serow::State::getFootOrientation,
             "Returns the foot orientation for a given frame")
        .def("get_foot_pose", &serow::State::getFootPose, "Returns the foot pose for a given frame")
        .def("get_foot_linear_velocity", &serow::State::getFootLinearVelocity,
             "Returns the foot linear velocity for a given frame")
        .def("get_foot_angular_velocity", &serow::State::getFootAngularVelocity,
             "Returns the foot angular velocity for a given frame")
        .def("get_com_position", &serow::State::getCoMPosition,
             "Returns the center of mass position")
        .def("get_com_linear_velocity", &serow::State::getCoMLinearVelocity,
             "Returns the center of mass linear velocity")
        .def("get_com_external_forces", &serow::State::getCoMExternalForces,
             "Returns the center of mass external forces")
        .def("get_com_angular_momentum", &serow::State::getCoMAngularMomentum,
             "Returns the center of mass angular momentum")
        .def("get_com_angular_momentum_rate", &serow::State::getCoMAngularMomentumRate,
             "Returns the center of mass angular momentum rate")
        .def("get_com_linear_acceleration", &serow::State::getCoMLinearAcceleration,
             "Returns the center of mass linear acceleration")
        .def("get_cop_position", &serow::State::getCOPPosition,
             "Returns the center of pressure position")
        .def("get_base_pose_cov", &serow::State::getBasePoseCov, "Returns the base pose covariance")
        .def("get_base_velocity_cov", &serow::State::getBaseVelocityCov,
             "Returns the base velocity covariance")
        .def("get_base_position_cov", &serow::State::getBasePositionCov,
             "Returns the base position covariance")
        .def("get_base_orientation_cov", &serow::State::getBaseOrientationCov,
             "Returns the base orientation covariance")
        .def("get_base_linear_velocity_cov", &serow::State::getBaseLinearVelocityCov,
             "Returns the base linear velocity covariance")
        .def("get_base_angular_velocity_cov", &serow::State::getBaseAngularVelocityCov,
             "Returns the base angular velocity covariance")
        .def("get_imu_linear_acceleration_bias_cov", &serow::State::getImuLinearAccelerationBiasCov,
             "Returns the IMU linear acceleration bias covariance")
        .def("get_imu_angular_velocity_bias_cov", &serow::State::getImuAngularVelocityBiasCov,
             "Returns the IMU angular velocity bias covariance")
        .def("get_contact_pose_cov", &serow::State::getContactPoseCov,
             "Returns the contact pose covariance for a given frame")
        .def("get_contact_position_cov", &serow::State::getContactPositionCov,
             "Returns the contact position covariance for a given frame")
        .def("get_contact_orientation_cov", &serow::State::getContactOrientationCov,
             "Returns the contact orientation covariance for a given frame")
        .def("get_com_position_cov", &serow::State::getCoMPositionCov,
             "Returns the center of mass position covariance")
        .def("get_com_linear_velocity_cov", &serow::State::getCoMLinearVelocityCov,
             "Returns the center of mass linear velocity covariance")
        .def("get_com_external_forces_cov", &serow::State::getCoMExternalForcesCov,
             "Returns the center of mass external forces covariance")
        .def("get_mass", &serow::State::getMass, "Returns the mass of the robot")
        .def("get_num_leg_ee", &serow::State::getNumLegEE,
             "Returns the number of leg end-effectors")
        .def("is_point_feet", &serow::State::isPointFeet,
             "Returns whether the robot has point feet")
        .def("set_base_state", &serow::State::setBaseState, py::arg("base_state"),
             "Sets the base state of the robot")
        .def("set_contact_state", &serow::State::setContactState, py::arg("contact_state"),
             "Sets the contact state of the robot")
        .def("set_centroidal_state", &serow::State::setCentroidalState, py::arg("centroidal_state"),
             "Sets the centroidal state of the robot")
        .def("set_joint_state", &serow::State::setJointState, py::arg("joint_state"),
             "Sets the joint state of the robot")
        .def("get_base_state", &serow::State::getBaseState, "Returns the base state of the robot")
        .def("get_contact_state", &serow::State::getContactState,
             "Returns the contact state of the robot")
        .def("get_centroidal_state", &serow::State::getCentroidalState,
             "Returns the centroidal state of the robot")
        .def("get_joint_state", &serow::State::getJointState,
             "Returns the joint state of the robot");
}
