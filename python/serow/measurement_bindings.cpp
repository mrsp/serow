#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "Measurement.hpp"

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

PYBIND11_MODULE(measurement, m) {
    m.doc() = "Python bindings for Measurement class";
        
    py::class_<serow::ImuMeasurement>(m, "ImuMeasurement")
        .def(py::init<>())
        .def_readwrite("timestamp", &serow::ImuMeasurement::timestamp)
        .def_readwrite("angular_velocity", &serow::ImuMeasurement::angular_velocity)
        .def_readwrite("linear_acceleration", &serow::ImuMeasurement::linear_acceleration)
        .def_readwrite("angular_velocity_cov", &serow::ImuMeasurement::angular_velocity_cov)
        .def_readwrite("linear_acceleration_cov", &serow::ImuMeasurement::linear_acceleration_cov)
        .def_readwrite("angular_velocity_bias_cov",
                       &serow::ImuMeasurement::angular_velocity_bias_cov)
        .def_readwrite("linear_acceleration_bias_cov",
                       &serow::ImuMeasurement::linear_acceleration_bias_cov)
        .def_readwrite("angular_acceleration", &serow::ImuMeasurement::angular_acceleration);

    py::class_<serow::KinematicMeasurement>(m, "KinematicMeasurement")
        .def(py::init<>())
        .def_readwrite("timestamp", &serow::KinematicMeasurement::timestamp)
        .def_readwrite("base_linear_velocity", &serow::KinematicMeasurement::base_linear_velocity)
        .def_property(
            "base_orientation",
            [](const serow::KinematicMeasurement& self) {
                return quaternion_to_numpy(self.base_orientation);
            },
            [](serow::KinematicMeasurement& self, const py::array_t<double>& arr) {
                self.base_orientation = numpy_to_quaternion(arr);
            })
        .def_readwrite("contacts_status", &serow::KinematicMeasurement::contacts_status)
        .def_readwrite("contacts_probability", &serow::KinematicMeasurement::contacts_probability)
        .def_readwrite("contacts_position", &serow::KinematicMeasurement::contacts_position)
        .def_readwrite("base_to_foot_positions", &serow::KinematicMeasurement::base_to_foot_positions)
        .def_readwrite("contacts_position_noise", &serow::KinematicMeasurement::contacts_position_noise)
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
            })
        .def_readwrite("contacts_orientation_noise", &serow::KinematicMeasurement::contacts_orientation_noise)
        .def_readwrite("com_angular_momentum_derivative", &serow::KinematicMeasurement::com_angular_momentum_derivative)
        .def_readwrite("com_position", &serow::KinematicMeasurement::com_position)
        .def_readwrite("com_linear_acceleration", &serow::KinematicMeasurement::com_linear_acceleration)
        .def_readwrite("base_linear_velocity_cov", &serow::KinematicMeasurement::base_linear_velocity_cov)
        .def_readwrite("base_orientation_cov", &serow::KinematicMeasurement::base_orientation_cov)
        .def_readwrite("position_slip_cov", &serow::KinematicMeasurement::position_slip_cov)
        .def_readwrite("orientation_slip_cov", &serow::KinematicMeasurement::orientation_slip_cov)
        .def_readwrite("position_cov", &serow::KinematicMeasurement::position_cov)
        .def_readwrite("orientation_cov", &serow::KinematicMeasurement::orientation_cov)
        .def_readwrite("com_position_process_cov", &serow::KinematicMeasurement::com_position_process_cov)
        .def_readwrite("com_linear_velocity_process_cov", &serow::KinematicMeasurement::com_linear_velocity_process_cov)
        .def_readwrite("external_forces_process_cov", &serow::KinematicMeasurement::external_forces_process_cov)
        .def_readwrite("com_position_cov", &serow::KinematicMeasurement::com_position_cov)
        .def_readwrite("com_linear_acceleration_cov", &serow::KinematicMeasurement::com_linear_acceleration_cov);

    py::class_<serow::OdometryMeasurement>(m, "OdometryMeasurement")
        .def(py::init<>())
        .def_readwrite("timestamp", &serow::OdometryMeasurement::timestamp)
        .def_readwrite("base_position", &serow::OdometryMeasurement::base_position)
        .def_property(
            "base_orientation",
            [](const serow::OdometryMeasurement& self) {
                return quaternion_to_numpy(self.base_orientation);
            },
            [](serow::OdometryMeasurement& self, const py::array_t<double>& arr) {
                self.base_orientation = numpy_to_quaternion(arr);
            })
        .def_readwrite("base_position_cov", &serow::OdometryMeasurement::base_position_cov)
        .def_readwrite("base_orientation_cov", &serow::OdometryMeasurement::base_orientation_cov);
}
