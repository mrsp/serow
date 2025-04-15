#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ContactEKF.hpp"
#include "Measurement.hpp"
#include "State.hpp"
#include "common.hpp"

namespace py = pybind11;

PYBIND11_MODULE(contact_ekf, m) {
    m.doc() = "Python bindings for ContactEKF class";

    py::class_<serow::ContactEKF>(m, "ContactEKF")
        .def(py::init<>())
        .def("init", &serow::ContactEKF::init, py::arg("state"), py::arg("contacts_frame"),
             py::arg("point_feet"), py::arg("g"), py::arg("imu_rate"),
             py::arg("outlier_detection") = false)
        .def("predict", &serow::ContactEKF::predict, py::arg("state"), py::arg("imu"),
             py::arg("kin"))
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
            py::arg("terrain_estimator") = py::none())
        .def("set_action", &serow::ContactEKF::setAction, py::arg("action"))
        .def("get_contact_position_innovation", 
            [](serow::ContactEKF& self, const std::string& contact_frame) {
                Eigen::Vector3d innovation = Eigen::Vector3d::Zero();
                Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
                bool success = self.getContactPositionInnovation(contact_frame, innovation, covariance);
                return std::make_tuple(success, innovation, covariance);
            },
            py::arg("contact_frame"))
        .def("get_contact_orientation_innovation", 
            [](serow::ContactEKF& self, const std::string& contact_frame) {
                Eigen::Vector3d innovation = Eigen::Vector3d::Zero();
                Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
                bool success = self.getContactOrientationInnovation(contact_frame, innovation, covariance);
                return std::make_tuple(success, innovation, covariance);
            },
            py::arg("contact_frame"));
}
