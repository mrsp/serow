/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif

#include <iostream>
#include <map>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <string>
#include <vector>

namespace serow {

class RobotKinematics {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RobotKinematics(const std::string& model_name, bool verbose = false) {
        // Create the model
        pmodel_ = std::make_unique<pinocchio::Model>();

        // Create the kinematic tree
        pinocchio::urdf::buildModel(model_name, *pmodel_, verbose);

        // Create the data
        data_ = std::make_unique<pinocchio::Data>(*pmodel_);

        int names_size = pmodel_->names.size();
        jnames_.reserve(names_size);

        for (int i = 0; i < names_size; i++) {
            const std::string& jname = pmodel_->names[i];
            // Do not insert "universe" joint
            if (jname != "universe") {
                jnames_.push_back(jname);
            }
        }

        qmin_.resize(jnames_.size());
        qmax_.resize(jnames_.size());
        dqmax_.resize(jnames_.size());
        qn_.resize(jnames_.size());
        qmin_ = pmodel_->lowerPositionLimit;
        qmax_ = pmodel_->upperPositionLimit;
        dqmax_ = pmodel_->velocityLimit;

        // Continuous joints are given spurious values per default, set those values
        // to arbitrary ones
        for (int i = 0; i < qmin_.size(); i++) {
            double d = qmax_[i] - qmin_[i];
            // If wrong values or if difference less than 0.05 deg (0.001 rad)
            if ((d < 0) || (std::fabs(d) < 0.001)) {
                qmin_[i] = -50.0;
                qmax_[i] = 50.0;
                dqmax_[i] = 200.0;
            }
        }
        std::cout << "Joint Names " << std::endl;
        printJointNames();
        std::cout << "with " << ndofActuated() << " actuated joints" << std::endl;
        std::cout << "Model loaded: " << model_name << std::endl;
    }

    int ndof() const { return pmodel_->nq; }

    int ndofActuated() const { return pmodel_->nq; }

    void updateJointConfig(const std::map<std::string, double>& qmap,
                           const std::map<std::string, double>& qdotmap, double joint_std) {
        mapJointNamesIDs(qmap, qdotmap);
        pinocchio::framesForwardKinematics(*pmodel_, *data_, q_);
        pinocchio::computeJointJacobians(*pmodel_, *data_, q_);

        qn_.setOnes();
        qn_ *= joint_std;
    }

    Eigen::Vector3d getLinearVelocityNoise(const std::string& frame_name) const {
        return linearJacobian(frame_name) * qn_;
    }

    Eigen::Vector3d getAngularVelocityNoise(const std::string& frame_name) const {
        return angularJacobian(frame_name) * qn_;
    }

    void mapJointNamesIDs(const std::map<std::string, double>& qmap,
                          const std::map<std::string, double>& qdotmap) {
        q_.setZero(pmodel_->nq);
        qdot_.setZero(pmodel_->nv);

        for (int i = 0; i < jnames_.size(); i++) {
            int jidx = pmodel_->getJointId(jnames_[i]);
            int qidx = pmodel_->idx_qs[jidx];
            int vidx = pmodel_->idx_vs[jidx];

            // Model value is equal to 2 for continuous joints
            if (pmodel_->nqs[jidx] == 2) {
                q_[qidx] = cos(qmap.at(jnames_[i]));
                q_[qidx + 1] = sin(qmap.at(jnames_[i]));
                qdot_[vidx] = qdotmap.at(jnames_[i]);
            } else {
                q_[qidx] = qmap.at(jnames_[i]);
                qdot_[vidx] = qdotmap.at(jnames_[i]);
            }
        }
    }

    Eigen::MatrixXd geometricJacobian(const std::string& frame_name) const {
        try {
            pinocchio::Data::Matrix6x J(6, pmodel_->nv);
            J.fill(0);
            // Jacobian in pinocchio::LOCAL (link) frame
            pinocchio::Model::FrameIndex link_number = pmodel_->getFrameId(frame_name);

            pinocchio::getFrameJacobian(*pmodel_, *data_, link_number, pinocchio::LOCAL, J);

            // Transform Jacobians from pinocchio::LOCAL frame to base frame
            J.topRows(3) = (data_->oMf[link_number].rotation()) * J.topRows(3);
            J.bottomRows(3) = (data_->oMf[link_number].rotation()) * J.bottomRows(3);
            return J;
        } catch (std::exception& e) {
            std::cerr << "WARNING: Link name " << frame_name << " is invalid! ... "
                      << "Returning zeros" << std::endl;
            return Eigen::MatrixXd::Zero(6, ndofActuated());
        }
    }

    Eigen::Vector3d getLinearVelocity(const std::string& frame_name) const {
        return (linearJacobian(frame_name) * qdot_);
    }

    Eigen::Vector3d getAngularVelocity(const std::string& frame_name) const {
        return (angularJacobian(frame_name) * qdot_);
    }

    Eigen::Vector3d linkPosition(const std::string& frame_name) const {
        try {
            pinocchio::Model::FrameIndex link_number = pmodel_->getFrameId(frame_name);

            return data_->oMf[link_number].translation();
        } catch (std::exception& e) {
            std::cerr << "WARNING: Link name " << frame_name << " is invalid! ... "
                      << "Returning zeros" << std::endl;
            return Eigen::Vector3d::Zero();
        }
    }

    Eigen::Quaterniond linkOrientation(const std::string& frame_name) const {
        try {
            pinocchio::Model::FrameIndex link_number = pmodel_->getFrameId(frame_name);

            Eigen::Vector4d temp = rotationToQuaternion(data_->oMf[link_number].rotation());
            Eigen::Quaterniond tempQ;
            tempQ.w() = temp(0);
            tempQ.x() = temp(1);
            tempQ.y() = temp(2);
            tempQ.z() = temp(3);
            return tempQ;
        } catch (std::exception& e) {
            std::cerr << "WARNING: Frame name " << frame_name << " is invalid! ... "
                      << "Returning zeros" << std::endl;
            return Eigen::Quaterniond::Identity();
        }
    }

    Eigen::VectorXd linkPose(const std::string& frame_name) const {
        Eigen::VectorXd lpose(7);
        pinocchio::Model::FrameIndex link_number = pmodel_->getFrameId(frame_name);

        lpose.head(3) = data_->oMf[link_number].translation();
        lpose.tail(4) = rotationToQuaternion(data_->oMf[link_number].rotation());

        return lpose;
    }

    Eigen::MatrixXd linearJacobian(const std::string& frame_name) const {
        pinocchio::Data::Matrix6x J(6, pmodel_->nv);
        J.fill(0);
        pinocchio::Model::FrameIndex link_number = pmodel_->getFrameId(frame_name);

        pinocchio::getFrameJacobian(*pmodel_, *data_, link_number, pinocchio::LOCAL, J);
        try {
            // Transform Jacobian from pinocchio::LOCAL frame to base frame
            return (data_->oMf[link_number].rotation()) * J.topRows(3);
        } catch (std::exception& e) {
            std::cerr << "WARNING: Link name " << frame_name << " is invalid! ... "
                      << "Returning zeros" << std::endl;
            return Eigen::MatrixXd::Zero(3, ndofActuated());
        }
    }

    Eigen::MatrixXd angularJacobian(const std::string& frame_name) const {
        pinocchio::Data::Matrix6x J(6, pmodel_->nv);
        J.fill(0);
        pinocchio::Model::FrameIndex link_number = pmodel_->getFrameId(frame_name);
        try {
            // Jacobian in pinocchio::LOCAL frame
            pinocchio::getFrameJacobian(*pmodel_, *data_, link_number, pinocchio::LOCAL, J);

            // Transform Jacobian from pinocchio::LOCAL frame to base frame
            return (data_->oMf[link_number].rotation()) * J.bottomRows(3);
        } catch (std::exception& e) {
            std::cerr << "WARNING: Link name " << frame_name << " is invalid! ... "
                      << "Returning zeros" << std::endl;
            return Eigen::MatrixXd::Zero(3, ndofActuated());
        }
    }

    Eigen::VectorXd comPosition() const {
        pinocchio::centerOfMass(*pmodel_, *data_, q_);
        return data_->com[0];
    }

    Eigen::MatrixXd comJacobian() const {
        return pinocchio::jacobianCenterOfMass(*pmodel_, *data_, q_);
    }

    std::vector<std::string> jointNames() const { return jnames_; }

    Eigen::VectorXd jointMaxAngularLimits() const { return qmax_; }

    Eigen::VectorXd jointMinAngularLimits() const { return qmin_; }

    Eigen::VectorXd jointVelocityLimits() const { return dqmax_; }

    void printJointNames() const {
        for (const auto& jname : jnames_) {
            std::cout << jname << std::endl;
        };
    }

    void printJointLimits() const {
        if (!((jnames_.size() == qmin_.size()) && (jnames_.size() == qmax_.size()) &&
              (jnames_.size() == dqmax_.size()))) {
            std::cerr << "Joint names and joint limits size do not match!" << std::endl;
            return;
        }
        std::cout << "\nJoint Name\t qmin \t qmax \t dqmax" << std::endl;
        for (int i = 0; i < jnames_.size(); ++i)
            std::cout << jnames_[i] << "\t\t" << qmin_[i] << "\t" << qmax_[i] << "\t" << dqmax_[i]
                      << std::endl;
    }

    double sgn(const double& x) const {
        if (x >= 0) {
            return 1.0;
        } else {
            return -1.0;
        }
    }

    Eigen::Vector4d rotationToQuaternion(const Eigen::Matrix3d& R) const {
        double eps = 1e-6;
        Eigen::Vector4d quat;

        quat(0) = 0.5 * sqrt(R(0, 0) + R(1, 1) + R(2, 2) + 1.0);
        if (fabs(R(0, 0) - R(1, 1) - R(2, 2) + 1.0) < eps) {
            quat(1) = 0.0;
        } else {
            quat(1) = 0.5 * sgn(R(2, 1) - R(1, 2)) * sqrt(R(0, 0) - R(1, 1) - R(2, 2) + 1.0);
        }
        if (fabs(R(1, 1) - R(2, 2) - R(0, 0) + 1.0) < eps) {
            quat(2) = 0.0;
        } else {
            quat(2) = 0.5 * sgn(R(0, 2) - R(2, 0)) * sqrt(R(1, 1) - R(2, 2) - R(0, 0) + 1.0);
        }
        if (fabs(R(2, 2) - R(0, 0) - R(1, 1) + 1.0) < eps) {
            quat(3) = 0.0;
        } else {
            quat(3) = 0.5 * sgn(R(1, 0) - R(0, 1)) * sqrt(R(2, 2) - R(0, 0) - R(1, 1) + 1.0);
        }
        return quat;
    }

   private:
    std::unique_ptr<pinocchio::Model> pmodel_;
    std::unique_ptr<pinocchio::Data> data_;
    std::vector<std::string> jnames_;
    Eigen::VectorXd qmin_, qmax_, dqmax_, q_, qdot_, qn_;
};

}  // namespace serow
