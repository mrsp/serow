#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "Measurement.hpp"
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

PYBIND11_MODULE(state, m) {
    m.doc() = "Python bindings for BaseState class";

    py::class_<serow::BaseState>(m, "BaseState")
        .def(py::init<>())
        .def_readwrite("timestamp", &serow::BaseState::timestamp)
        .def_readwrite("base_position", &serow::BaseState::base_position)
        .def_property(
            "base_orientation",
            [](const serow::BaseState& self) { return quaternion_to_numpy(self.base_orientation); },
            [](serow::BaseState& self, const py::array_t<double>& arr) {
                self.base_orientation = numpy_to_quaternion(arr);
            })
        .def_readwrite("base_linear_velocity", &serow::BaseState::base_linear_velocity)
        .def_readwrite("imu_angular_velocity_bias", &serow::BaseState::imu_angular_velocity_bias)
        .def_readwrite("imu_linear_acceleration_bias",
                       &serow::BaseState::imu_linear_acceleration_bias)
        .def_readwrite("contacts_position", &serow::BaseState::contacts_position)
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
            })
        .def_readwrite("base_position_cov", &serow::BaseState::base_position_cov)
        .def_readwrite("base_orientation_cov", &serow::BaseState::base_orientation_cov)
        .def_readwrite("base_linear_velocity_cov", &serow::BaseState::base_linear_velocity_cov)
        .def_readwrite("imu_angular_velocity_bias_cov",
                       &serow::BaseState::imu_angular_velocity_bias_cov)
        .def_readwrite("imu_linear_acceleration_bias_cov",
                       &serow::BaseState::imu_linear_acceleration_bias_cov)
        .def_readwrite("contacts_position_cov", &serow::BaseState::contacts_position_cov)
        .def_readwrite("contacts_orientation_cov", &serow::BaseState::contacts_orientation_cov);
}
