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
#include "NaiveLocalTerrainMapper.hpp"
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
        .def_readwrite("base_local_linear_velocity", &serow::BaseState::base_local_linear_velocity,
                       "Base local linear velocity (3D vector)")
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
        .def_readwrite("base_linear_acceleration_cov",
                       &serow::BaseState::base_linear_acceleration_cov,
                       "Base linear acceleration covariance (3x3 matrix)")
        .def_readwrite("base_angular_acceleration_cov",
                       &serow::BaseState::base_angular_acceleration_cov,
                       "Base angular acceleration covariance (3x3 matrix)")
        .def_readwrite("imu_angular_velocity_bias_cov",
                       &serow::BaseState::imu_angular_velocity_bias_cov,
                       "IMU angular velocity bias covariance (3x3 matrix)")
        .def_readwrite("imu_linear_acceleration_bias_cov",
                       &serow::BaseState::imu_linear_acceleration_bias_cov,
                       "IMU linear acceleration bias covariance (3x3 matrix)")
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
                    quaternion_to_numpy(state.base_orientation), state.base_local_linear_velocity,
                    state.base_linear_velocity, state.base_angular_velocity,
                    state.base_linear_acceleration, state.base_angular_acceleration,
                    state.imu_angular_velocity_bias, state.imu_linear_acceleration_bias,
                    state.contacts_position, contacts_orientation_serialized,
                    state.base_position_cov, state.base_orientation_cov,
                    state.base_linear_velocity_cov, state.base_angular_velocity_cov,
                    state.base_linear_acceleration_cov, state.base_angular_acceleration_cov,
                    state.imu_angular_velocity_bias_cov, state.imu_linear_acceleration_bias_cov,
                    state.feet_position, feet_orientation_serialized, state.feet_linear_velocity,
                    state.feet_angular_velocity);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 24)
                    throw std::runtime_error("Invalid state for BaseState!");

                serow::BaseState state;
                state.timestamp = t[0].cast<double>();
                state.base_position = t[1].cast<decltype(state.base_position)>();
                state.base_orientation = numpy_to_quaternion(t[2].cast<py::array_t<double>>());
                state.base_local_linear_velocity =
                    t[3].cast<decltype(state.base_local_linear_velocity)>();
                state.base_linear_velocity = t[4].cast<decltype(state.base_linear_velocity)>();
                state.base_angular_velocity = t[5].cast<decltype(state.base_angular_velocity)>();
                state.base_linear_acceleration =
                    t[6].cast<decltype(state.base_linear_acceleration)>();
                state.base_angular_acceleration =
                    t[7].cast<decltype(state.base_angular_acceleration)>();
                state.imu_angular_velocity_bias =
                    t[8].cast<decltype(state.imu_angular_velocity_bias)>();
                state.imu_linear_acceleration_bias =
                    t[9].cast<decltype(state.imu_linear_acceleration_bias)>();
                state.contacts_position = t[10].cast<decltype(state.contacts_position)>();

                // Handle optional contacts_orientation
                auto contacts_orientation_serialized =
                    t[11].cast<std::map<std::string, py::array_t<double>>>();
                if (!contacts_orientation_serialized.empty()) {
                    std::map<std::string, Eigen::Quaterniond> contacts_orientation;
                    for (const auto& [name, arr] : contacts_orientation_serialized) {
                        contacts_orientation[name] = numpy_to_quaternion(arr);
                    }
                    state.contacts_orientation = contacts_orientation;
                }

                state.base_position_cov = t[12].cast<decltype(state.base_position_cov)>();
                state.base_orientation_cov = t[13].cast<decltype(state.base_orientation_cov)>();
                state.base_linear_velocity_cov =
                    t[14].cast<decltype(state.base_linear_velocity_cov)>();
                state.base_angular_velocity_cov =
                    t[15].cast<decltype(state.base_angular_velocity_cov)>();
                state.base_linear_acceleration_cov =
                    t[16].cast<decltype(state.base_linear_acceleration_cov)>();
                state.base_angular_acceleration_cov =
                    t[17].cast<decltype(state.base_angular_acceleration_cov)>();
                state.imu_angular_velocity_bias_cov =
                    t[18].cast<decltype(state.imu_angular_velocity_bias_cov)>();
                state.imu_linear_acceleration_bias_cov =
                    t[19].cast<decltype(state.imu_linear_acceleration_bias_cov)>();
                state.feet_position = t[20].cast<decltype(state.feet_position)>();

                // Convert feet_orientation from numpy arrays
                auto feet_orientation_serialized =
                    t[21].cast<std::map<std::string, py::array_t<double>>>();
                for (const auto& [name, arr] : feet_orientation_serialized) {
                    state.feet_orientation[name] = numpy_to_quaternion(arr);
                }

                state.feet_linear_velocity = t[22].cast<decltype(state.feet_linear_velocity)>();
                state.feet_angular_velocity = t[23].cast<decltype(state.feet_angular_velocity)>();

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
                       "Optional map of contact torques (string to 3D vector)")
        // Add pickle support
        .def(py::pickle(
            [](const serow::ContactState& contact) {  // __getstate__
                // Convert optional contacts_torque
                py::object contacts_torque_serialized = py::none();
                if (contact.contacts_torque.has_value()) {
                    contacts_torque_serialized = py::cast(contact.contacts_torque.value());
                }
                return py::make_tuple(contact.timestamp, contact.contacts_status,
                                      contact.contacts_probability, contact.contacts_force,
                                      contacts_torque_serialized);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 5)
                    throw std::runtime_error("Invalid state for ContactState!");

                serow::ContactState contact;
                contact.timestamp = t[0].cast<double>();
                contact.contacts_status = t[1].cast<decltype(contact.contacts_status)>();
                contact.contacts_probability = t[2].cast<decltype(contact.contacts_probability)>();
                contact.contacts_force = t[3].cast<decltype(contact.contacts_force)>();

                // Handle optional contacts_torque
                if (!t[4].is_none()) {
                    contact.contacts_torque = t[4].cast<std::map<std::string, Eigen::Vector3d>>();
                }

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

    // Binding for GroundReactionForceMeasurement
    py::class_<serow::GroundReactionForceMeasurement>(
        m, "GroundReactionForceMeasurement",
        "Represents ground reaction force measurements including force and center of pressure")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::GroundReactionForceMeasurement::timestamp,
                       "Timestamp of the measurement (s)")
        .def_readwrite("force", &serow::GroundReactionForceMeasurement::force,
                       "Ground reaction force (N)")
        .def_readwrite("cop", &serow::GroundReactionForceMeasurement::cop,
                       "Center of pressure (COP) (m)")
        .def(py::pickle(
            [](const serow::GroundReactionForceMeasurement& grf) {  // __getstate__
                return py::make_tuple(grf.timestamp, grf.force, grf.cop);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state for GroundReactionForceMeasurement!");

                serow::GroundReactionForceMeasurement grf;
                grf.timestamp = t[0].cast<double>();
                grf.force = t[1].cast<decltype(grf.force)>();
                grf.cop = t[2].cast<decltype(grf.cop)>();

                return grf;
            }));

    // Binding for KinematicMeasurement
    py::class_<serow::KinematicMeasurement>(m, "KinematicMeasurement",
                                            "Represents kinematic measurements")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &serow::KinematicMeasurement::timestamp,
                       "Timestamp of the measurement")
        .def_readwrite("base_position", &serow::KinematicMeasurement::base_position,
                       "Base position (3D vector)")
        .def_readwrite("base_linear_velocity", &serow::KinematicMeasurement::base_linear_velocity,
                       "Base linear velocity (3D vector)")
        .def_readwrite("base_linear_velocity_cov",
                       &serow::KinematicMeasurement::base_linear_velocity_cov,
                       "Covariance matrix of base linear velocity (m^2/s^2)")
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
        .def_readwrite("contacts_linear_velocity_noise",
                       &serow::KinematicMeasurement::contacts_linear_velocity_noise,
                       "Map of contact linear velocity noise (string to 3x3 matrix)")
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
        .def_readwrite("com_position", &serow::KinematicMeasurement::com_position,
                       "Center of mass position relative to base frame (3D vector)")
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
                       "Center of mass position covariance relative to base frame (3x3 matrix)")
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
                    kin.timestamp, kin.base_position, kin.base_linear_velocity,
                    kin.base_linear_velocity_cov, kin.contacts_status, kin.contacts_probability,
                    kin.is_new_contact, kin.contacts_position, kin.base_to_foot_positions,
                    base_to_foot_orientations_serialized, kin.base_to_foot_linear_velocities,
                    kin.base_to_foot_angular_velocities, kin.contacts_position_noise,
                    kin.contacts_linear_velocity_noise, contacts_orientation_serialized,
                    kin.com_position, kin.com_position_process_cov,
                    kin.com_linear_velocity_process_cov, kin.external_forces_process_cov,
                    kin.com_position_cov);
            },
            [](py::tuple t) {  // __setstate__
                const std::size_t n = t.size();
                if (n != 20 && n != 27)
                    throw std::runtime_error("Invalid state for KinematicMeasurement!");

                serow::KinematicMeasurement kin;
                kin.timestamp = t[0].cast<double>();
                kin.base_position = t[1].cast<decltype(kin.base_position)>();
                kin.base_linear_velocity = t[2].cast<decltype(kin.base_linear_velocity)>();
                kin.base_linear_velocity_cov = t[3].cast<decltype(kin.base_linear_velocity_cov)>();
                kin.contacts_status = t[4].cast<decltype(kin.contacts_status)>();
                kin.contacts_probability = t[5].cast<decltype(kin.contacts_probability)>();
                kin.is_new_contact = t[6].cast<decltype(kin.is_new_contact)>();
                kin.contacts_position = t[7].cast<decltype(kin.contacts_position)>();
                kin.base_to_foot_positions = t[8].cast<decltype(kin.base_to_foot_positions)>();

                // Convert base_to_foot_orientations from numpy arrays
                auto base_to_foot_orientations_serialized =
                    t[9].cast<std::map<std::string, py::array_t<double>>>();
                for (const auto& [name, arr] : base_to_foot_orientations_serialized) {
                    kin.base_to_foot_orientations[name] = numpy_to_quaternion(arr);
                }

                kin.base_to_foot_linear_velocities =
                    t[10].cast<decltype(kin.base_to_foot_linear_velocities)>();
                kin.base_to_foot_angular_velocities =
                    t[11].cast<decltype(kin.base_to_foot_angular_velocities)>();
                kin.contacts_position_noise = t[12].cast<decltype(kin.contacts_position_noise)>();
                kin.contacts_linear_velocity_noise =
                    t[13].cast<decltype(kin.contacts_linear_velocity_noise)>();

                // Handle optional contacts_orientation
                auto contacts_orientation_serialized =
                    t[14].cast<std::map<std::string, py::array_t<double>>>();
                if (!contacts_orientation_serialized.empty()) {
                    std::map<std::string, Eigen::Quaterniond> contacts_orientation;
                    for (const auto& [name, arr] : contacts_orientation_serialized) {
                        contacts_orientation[name] = numpy_to_quaternion(arr);
                    }
                    kin.contacts_orientation = contacts_orientation;
                }

                if (n == 20) {
                    kin.com_position = t[15].cast<decltype(kin.com_position)>();
                    kin.com_position_process_cov =
                        t[16].cast<decltype(kin.com_position_process_cov)>();
                    kin.com_linear_velocity_process_cov =
                        t[17].cast<decltype(kin.com_linear_velocity_process_cov)>();
                    kin.external_forces_process_cov =
                        t[18].cast<decltype(kin.external_forces_process_cov)>();
                    kin.com_position_cov = t[19].cast<decltype(kin.com_position_cov)>();
                } else {
                    // Legacy 27-tuple layout (pre-Measurement.hpp slimming)
                    kin.com_position = t[16].cast<decltype(kin.com_position)>();
                    kin.com_position_process_cov =
                        t[22].cast<decltype(kin.com_position_process_cov)>();
                    kin.com_linear_velocity_process_cov =
                        t[23].cast<decltype(kin.com_linear_velocity_process_cov)>();
                    kin.external_forces_process_cov =
                        t[24].cast<decltype(kin.external_forces_process_cov)>();
                    kin.com_position_cov = t[25].cast<decltype(kin.com_position_cov)>();
                }

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
             py::arg("g"), py::arg("imu_rate"), py::arg("kin_rate"), py::arg("eps") = 0.05,
             py::arg("point_feet") = true, py::arg("use_imu_orientation") = false,
             py::arg("verbose") = false,
             "Initializes the EKF with the initial robot state and parameters")
        .def("predict", &serow::ContactEKF::predict, py::arg("state"), py::arg("imu"),
             "Predicts the robot's state forward based on IMU measurements")
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
            "terrain data")
        .def("set_state", &serow::ContactEKF::setState, py::arg("state"),
             "Sets the state of the EKF")
        .def("update_with_base_linear_velocity", &serow::ContactEKF::updateWithBaseLinearVelocity,
             py::arg("state"), py::arg("base_linear_velocity"), py::arg("base_linear_velocity_cov"),
             py::arg("timestamp"),
             "Updates the robot's state based on base linear velocity measurements")
        .def("update_with_imu_orientation", &serow::ContactEKF::updateWithIMUOrientation,
             py::arg("state"), py::arg("imu_orientation"), py::arg("imu_orientation_cov"),
             py::arg("timestamp"),
             "Updates the robot's state based on IMU orientation measurements");

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
        .def("set_state", &serow::Serow::setState, py::arg("state"), "Sets the state of the robot")
        .def("get_terrain_estimator", &serow::Serow::getTerrainEstimator,
             "Returns the terrain estimator object")
        .def(
            "process_measurements",
            [](serow::Serow& self, const serow::ImuMeasurement& imu,
               const std::map<std::string, serow::JointMeasurement>& joints,
               py::object force_torque, py::object contacts_probability) {
                std::optional<std::map<std::string, serow::ForceTorqueMeasurement>> ft_opt;
                if (!force_torque.is_none()) {
                    ft_opt =
                        force_torque.cast<std::map<std::string, serow::ForceTorqueMeasurement>>();
                }

                std::optional<std::map<std::string, serow::ContactMeasurement>> contact_prob_opt;
                if (!contacts_probability.is_none()) {
                    contact_prob_opt =
                        contacts_probability
                            .cast<std::map<std::string, serow::ContactMeasurement>>();
                }

                return self.processMeasurements(imu, joints, ft_opt, contact_prob_opt);
            },
            py::arg("imu"), py::arg("joints"), py::arg("force_torque") = py::none(),
            py::arg("contacts_probability") = py::none(),
            "Processes the measurements and returns a tuple of IMU, kinematic, and force-torque "
            "measurements")
        .def("base_estimator_predict_step", &serow::Serow::baseEstimatorPredictStep, py::arg("imu"),
             py::arg("kin"), "Runs the base estimator's predict step")
        .def("base_estimator_update_with_imu_orientation",
             &serow::Serow::baseEstimatorUpdateWithImuOrientation, py::arg("imu"),
             "Runs the base estimator's update step with the IMU orientation")
        .def("base_estimator_update_with_base_linear_velocity",
             &serow::Serow::baseEstimatorUpdateWithBaseLinearVelocity, py::arg("kin"),
             "Runs the base estimator's update step with base linear velocity")
        .def("base_estimator_finish_update", &serow::Serow::baseEstimatorFinishUpdate,
             py::arg("imu"), py::arg("kin"),
             "Concludes the base estimator's update step with the IMU measurement")
        .def("reset", &serow::Serow::reset, "Resets the state of SEROW");

    // Binding for ElevationCell
    py::class_<serow::ElevationCell>(m, "ElevationCell",
                                     "Represents a single cell in the terrain elevation map")
        .def(py::init<>(), "Default constructor")
        .def(py::init<float, float>(), py::arg("height"), py::arg("variance"),
             "Constructor with height and variance")
        .def_readwrite("height", &serow::ElevationCell::height, "Terrain height (m)")
        .def_readwrite("variance", &serow::ElevationCell::variance, "Terrain height variance (m^2)")
        .def_readwrite("contact", &serow::ElevationCell::contact,
                       "Whether this cell is a contact point")
        .def_readwrite("updated", &serow::ElevationCell::updated,
                       "Whether this cell has been updated");

    // Binding for TerrainElevation::Params
    py::class_<serow::TerrainElevation::Params>(m, "TerrainElevationParams",
                                                "Parameters for the terrain elevation mapper")
        .def(py::init<>(), "Default constructor with sensible defaults")
        .def(py::init<float, float, float, float, float, float, size_t, float, float, float,
                      float>(),
             py::arg("resolution"), py::arg("radius"), py::arg("dist_variance_gain"),
             py::arg("power"), py::arg("min_variance"), py::arg("max_recenter_distance"),
             py::arg("max_contact_points"), py::arg("min_contact_probability"),
             py::arg("min_stable_contact_probability") = 0.95f,
             py::arg("min_stable_foot_angular_velocity") = 0.03f,
             py::arg("min_stable_foot_linear_velocity") = 0.03f, "Constructor with all parameters")
        .def_readwrite("resolution", &serow::TerrainElevation::Params::resolution,
                       "Map resolution (m/cell)")
        .def_readwrite("resolution_inv", &serow::TerrainElevation::Params::resolution_inv,
                       "Inverse of map resolution (cell/m)")
        .def_readwrite("radius", &serow::TerrainElevation::Params::radius,
                       "Radius of inflation per contact point (m)")
        .def_readwrite("radius_cells", &serow::TerrainElevation::Params::radius_cells,
                       "Radius of inflation in cells")
        .def_readwrite("dist_variance_gain", &serow::TerrainElevation::Params::dist_variance_gain,
                       "Gain to scale the variance based on distance")
        .def_readwrite("power", &serow::TerrainElevation::Params::power,
                       "Power parameter for inverse distance weighting")
        .def_readwrite("min_variance", &serow::TerrainElevation::Params::min_variance,
                       "Minimum terrain height variance (m^2)")
        .def_readwrite("max_recenter_distance",
                       &serow::TerrainElevation::Params::max_recenter_distance,
                       "Maximum distance to trigger recentering (m)")
        .def_readwrite("max_contact_points", &serow::TerrainElevation::Params::max_contact_points,
                       "Maximum number of stored contact points")
        .def_readwrite("min_contact_probability",
                       &serow::TerrainElevation::Params::min_contact_probability,
                       "Minimum contact probability for terrain estimation")
        .def_readwrite("min_stable_contact_probability",
                       &serow::TerrainElevation::Params::min_stable_contact_probability,
                       "Minimum stable contact probability for terrain estimation")
        .def_readwrite("min_stable_foot_angular_velocity",
                       &serow::TerrainElevation::Params::min_stable_foot_angular_velocity,
                       "Minimum stable foot angular velocity (rad/s)")
        .def_readwrite("min_stable_foot_linear_velocity",
                       &serow::TerrainElevation::Params::min_stable_foot_linear_velocity,
                       "Minimum stable foot linear velocity (m/s)");

    // Binding for TerrainElevation (abstract base class)
    py::class_<serow::TerrainElevation, std::shared_ptr<serow::TerrainElevation>>(
        m, "TerrainElevation", "Abstract base class for terrain elevation mapping")
        .def("print_map_information", &serow::TerrainElevation::printMapInformation,
             "Prints terrain map metadata to stdout")
        .def("get_map_origin", &serow::TerrainElevation::getMapOrigin,
             "Returns the map origin as [x, y]")
        .def("recenter", &serow::TerrainElevation::recenter, py::arg("location"),
             "Recenters the local map around the given [x, y] location")
        .def("initialize_local_map", &serow::TerrainElevation::initializeLocalMap,
             py::arg("height"), py::arg("variance"),
             py::arg("params") = serow::TerrainElevation::Params(),
             "Initializes the local map with a default height and variance")
        .def("update", &serow::TerrainElevation::update, py::arg("loc"), py::arg("height"),
             py::arg("variance"), py::arg("normal") = std::nullopt,
             "Updates the elevation at the given [x, y] location")
        .def("set_elevation", &serow::TerrainElevation::setElevation, py::arg("loc"),
             py::arg("elevation"), "Sets the elevation cell at the given [x, y] location")
        .def("get_elevation", &serow::TerrainElevation::getElevation, py::arg("loc"),
             "Returns the elevation cell at the given [x, y] location, or None if outside")
        .def("inside",
             static_cast<bool (serow::TerrainElevation::*)(const std::array<float, 2>&) const>(
                 &serow::TerrainElevation::inside),
             py::arg("location"), "Checks if a [x, y] location is inside the local map")
        .def("location_to_hash_id", &serow::TerrainElevation::locationToHashId, py::arg("loc"),
             "Converts a [x, y] location to a hash ID")
        .def("hash_id_to_location", &serow::TerrainElevation::hashIdToLocation, py::arg("hash_id"),
             "Converts a hash ID to a [x, y] location")
        .def("get_elevation_map", &serow::TerrainElevation::getElevationMap,
             "Returns the full elevation map as an array of ElevationCells")
        .def("get_local_map_info", &serow::TerrainElevation::getLocalMapInfo,
             "Returns (origin, bound_max, bound_min) of the local map")
        .def("add_contact_point", &serow::TerrainElevation::addContactPoint, py::arg("point"),
             "Adds a contact point [x, y] to the contact point buffer")
        .def("get_max_recenter_distance", &serow::TerrainElevation::getMaxRecenterDistance,
             "Returns the maximum recenter distance (m)")
        .def("get_resolution", &serow::TerrainElevation::getResolution,
             "Returns the map resolution (m/cell)")
        .def("get_min_contact_probability", &serow::TerrainElevation::getMinContactProbability,
             "Returns the minimum contact probability threshold")
        .def("get_min_stable_contact_probability",
             &serow::TerrainElevation::getMinStableContactProbability,
             "Returns the minimum stable contact probability threshold")
        .def("get_min_stable_foot_angular_velocity",
             &serow::TerrainElevation::getMinStableFootAngularVelocity,
             "Returns the minimum stable foot angular velocity (rad/s)")
        .def("get_min_stable_foot_linear_velocity",
             &serow::TerrainElevation::getMinStableFootLinearVelocity,
             "Returns the minimum stable foot linear velocity (m/s)")
        .def("clear_contact_points", &serow::TerrainElevation::clearContactPoints,
             "Clears all stored contact points")
        .def("interpolate_contact_points", &serow::TerrainElevation::interpolateContactPoints,
             "Interpolates elevation between stored contact points using inverse distance "
             "weighting");

    // Binding for LocalTerrainMapper (concrete implementation)
    py::class_<serow::LocalTerrainMapper, serow::TerrainElevation,
               std::shared_ptr<serow::LocalTerrainMapper>>(
        m, "LocalTerrainMapper", "Fast local terrain elevation mapper using hash-based indexing")
        .def(py::init<>(), "Default constructor");

    // Binding for NaiveLocalTerrainMapper (concrete implementation)
    py::class_<serow::NaiveLocalTerrainMapper, serow::TerrainElevation,
               std::shared_ptr<serow::NaiveLocalTerrainMapper>>(
        m, "NaiveLocalTerrainMapper", "Naive local terrain elevation mapper")
        .def(py::init<>(), "Default constructor");

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
                       "External forces covariance (3x3 matrix)")
        .def(py::pickle(
            [](const serow::CentroidalState& cs) {  // __getstate__
                return py::make_tuple(cs.timestamp, cs.com_position, cs.com_linear_velocity,
                                      cs.external_forces, cs.cop_position,
                                      cs.com_linear_acceleration, cs.angular_momentum,
                                      cs.angular_momentum_derivative, cs.com_position_cov,
                                      cs.com_linear_velocity_cov, cs.external_forces_cov);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 11)
                    throw std::runtime_error("Invalid state for CentroidalState!");

                serow::CentroidalState cs;
                cs.timestamp = t[0].cast<double>();
                cs.com_position = t[1].cast<decltype(cs.com_position)>();
                cs.com_linear_velocity = t[2].cast<decltype(cs.com_linear_velocity)>();
                cs.external_forces = t[3].cast<decltype(cs.external_forces)>();
                cs.cop_position = t[4].cast<decltype(cs.cop_position)>();
                cs.com_linear_acceleration = t[5].cast<decltype(cs.com_linear_acceleration)>();
                cs.angular_momentum = t[6].cast<decltype(cs.angular_momentum)>();
                cs.angular_momentum_derivative =
                    t[7].cast<decltype(cs.angular_momentum_derivative)>();
                cs.com_position_cov = t[8].cast<decltype(cs.com_position_cov)>();
                cs.com_linear_velocity_cov = t[9].cast<decltype(cs.com_linear_velocity_cov)>();
                cs.external_forces_cov = t[10].cast<decltype(cs.external_forces_cov)>();

                return cs;
            }));

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
        .def("get_joint_positions", &serow::State::getJointPositions, "Returns the joint positions")
        .def("get_joint_velocities", &serow::State::getJointVelocities,
             "Returns the joint velocities")
        .def("get_timestamp", &serow::State::getTimestamp, py::arg("state_type") = "base",
             "Returns the timestamp of the state")
        .def(
            "get_base_pose", [](const serow::State& self) { return self.getBasePose().matrix(); },
            "Returns the base pose as a 4x4 transformation matrix")
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
        .def(
            "get_contact_orientation",
            [](const serow::State& self,
               const std::string& frame_name) -> std::optional<py::array_t<double>> {
                auto q = self.getContactOrientation(frame_name);
                if (q) {
                    return quaternion_to_numpy(*q);
                }
                return std::nullopt;
            },
            py::arg("frame_name"), "Returns the contact orientation for a given frame")
        .def(
            "get_contact_pose",
            [](const serow::State& self,
               const std::string& frame_name) -> std::optional<Eigen::Matrix4d> {
                auto pose = self.getContactPose(frame_name);
                if (pose) {
                    return pose->matrix();
                }
                return std::nullopt;
            },
            py::arg("frame_name"), "Returns the contact pose as a 4x4 transformation matrix")
        .def("get_contact_status", &serow::State::getContactStatus,
             "Returns the contact status for a given frame")
        .def("get_contact_force", &serow::State::getContactForce,
             "Returns the contact force for a given frame")
        .def("get_contact_torque", &serow::State::getContactTorque,
             "Returns the contact torque for a given frame")
        .def("get_contact_probability", &serow::State::getContactProbability,
             "Returns the contact probability for a given frame")
        .def("get_foot_position", &serow::State::getFootPosition,
             "Returns the foot position for a given frame")
        .def(
            "get_foot_orientation",
            [](const serow::State& self, const std::string& frame_name) {
                return quaternion_to_numpy(self.getFootOrientation(frame_name));
            },
            py::arg("frame_name"), "Returns the foot orientation for a given frame")
        .def(
            "get_foot_pose",
            [](const serow::State& self, const std::string& frame_name) {
                return self.getFootPose(frame_name).matrix();
            },
            py::arg("frame_name"), "Returns the foot pose as a 4x4 transformation matrix")
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
        .def("is_valid", &serow::State::isValid, "Returns whether the state is valid")
        .def("is_initialized", &serow::State::isInitialized,
             "Returns whether the state is initialized")
        .def("set_valid", &serow::State::setValid, py::arg("valid"),
             "Sets whether the state is valid")
        .def("set_initialized", &serow::State::setInitialized, py::arg("initialized"),
             "Sets whether the state is initialized")
        .def("get_base_frame", &serow::State::getBaseFrame, "Returns the base frame name")
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
