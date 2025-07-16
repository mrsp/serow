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
/**
 * @file RobotKinematics.hpp
 * @brief Implements a class for robot kinematics using Pinocchio library
 * @author Stylianos Piperakis
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
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <string>
#include <vector>

namespace serow {

/**
 * @class RobotKinematics
 * @brief Provides kinematic calculations and operations for a robot model using Pinocchio library
 */
class RobotKinematics {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Constructor to initialize the robot model and kinematic data
     * @param model_name Name of the URDF model file or robot model
     * @param verbose Verbosity flag for model loading (default: false)
     */
    RobotKinematics(const std::string& model_name, double joint_position_variance,
                    bool verbose = false) {
        // Create the model
        pmodel_ = std::make_unique<pinocchio::Model>();

        // Build the kinematic model from URDF file
        pinocchio::urdf::buildModel(model_name, *pmodel_, verbose);

        // Create the data associated with the model
        data_ = std::make_unique<pinocchio::Data>(*pmodel_);

        // Initialize joint names excluding the "universe" joint
        int names_size = pmodel_->names.size();
        jnames_.reserve(names_size);
        for (int i = 0; i < names_size; i++) {
            const std::string& jname = pmodel_->names[i];
            if (jname != "universe") {
                jnames_.push_back(jname);
            }
        }

        // Initialize joint limits
        qmin_.resize(jnames_.size());
        qmax_.resize(jnames_.size());
        dqmax_.resize(jnames_.size());
        qn_.resize(jnames_.size());
        qmin_ = pmodel_->lowerPositionLimit;
        qmax_ = pmodel_->upperPositionLimit;
        dqmax_ = pmodel_->velocityLimit;
        qn_.setOnes();
        qn_ *= std::sqrt(joint_position_variance);

        // Set default values for continuous joints if limits are invalid
        for (int i = 0; i < qmin_.size(); i++) {
            double d = qmax_[i] - qmin_[i];
            if ((d < 0) || (std::fabs(d) < 0.001)) {
                qmin_[i] = -50.0;
                qmax_[i] = 50.0;
                dqmax_[i] = 200.0;
            }
        }

        computeTotalMass();

        for (const auto& frame : pmodel_->frames) {
            if (frame.type == pinocchio::FrameType::BODY) {
                frame_names_.push_back(frame.name);
            }
        }

        if (verbose) {
            // Output model information
            std::cout << "Joint Names: " << std::endl;
            printJointNames();
            std::cout << "with " << ndofActuated() << " actuated joints" << std::endl;
            std::cout << "Model loaded: " << model_name << std::endl;
            std::cout << "Frame Names: " << std::endl;
            for (const auto& frame : frame_names_) {
                std::cout << frame << std::endl;
            }
        }
    }

    /**
     * @brief Returns the number of degrees of freedom (DOF) of the model
     * @return Number of DOF
     */
    int ndof() const {
        return pmodel_->nq;
    }

    /**
     * @brief Returns the number of actuated degrees of freedom (DOF) of the model
     * @return Number of actuated DOF
     */
    int ndofActuated() const {
        return pmodel_->nq;
    }

    /**
     * @brief Updates the joint configuration and kinematic data
     * @param qmap Map of joint names to their positions
     * @param qdotmap Map of joint names to their velocities
     * @param joint_std Standard deviation for joint configuration noise
     */
    void updateJointConfig(const std::map<std::string, double>& qmap,
                           const std::map<std::string, double>& qdotmap) {
        mapJointNamesIDs(qmap, qdotmap);
        pinocchio::framesForwardKinematics(*pmodel_, *data_, q_);
        pinocchio::computeJointJacobians(*pmodel_, *data_, q_);
    }

    /**
     * @brief Computes the noise in linear velocity of a specific frame
     * @param frame_name Name of the frame
     * @return Noise in linear velocity
     */
    Eigen::Vector3d linearVelocityNoise(const std::string& frame_name) const {
        return linearJacobian(frame_name) * qn_;
    }

    /**
     * @brief Computes the noise in angular velocity of a specific frame
     * @param frame_name Name of the frame
     * @return Noise in angular velocity
     */
    Eigen::Vector3d angularVelocityNoise(const std::string& frame_name) const {
        return angularJacobian(frame_name) * qn_;
    }

    /**
     * @brief Maps joint names to their respective indices and assigns values
     * @param qmap Map of joint names to their positions
     * @param qdotmap Map of joint names to their velocities
     */
    void mapJointNamesIDs(const std::map<std::string, double>& qmap,
                          const std::map<std::string, double>& qdotmap) {
        q_.setZero(pmodel_->nq);
        qdot_.setZero(pmodel_->nv);

        for (size_t i = 0; i < jnames_.size(); i++) {
            size_t jidx = pmodel_->getJointId(jnames_[i]);
            size_t qidx = pmodel_->idx_qs[jidx];
            size_t vidx = pmodel_->idx_vs[jidx];

            // Assign values based on joint type (continuous or not)
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

    /**
     * @brief Computes the geometric Jacobian matrix of a frame
     * @param frame_name Name of the frame
     * @return Geometric Jacobian matrix
     */
    Eigen::MatrixXd geometricJacobian(const std::string& frame_name) const {
        pinocchio::Data::Matrix6x J(6, pmodel_->nv);
        J.fill(0);

        pinocchio::Model::FrameIndex link_number = pmodel_->getFrameId(frame_name);

        // Check if the frame index is invalid
        if (link_number >= static_cast<pinocchio::Model::FrameIndex>(pmodel_->nframes)) {
            std::cerr << "WARNING: Link name <" << frame_name << "> is invalid! ... "
                      << "Returning zeros" << std::endl;
            return Eigen::MatrixXd::Zero(6, ndofActuated());
        }

        pinocchio::getFrameJacobian(*pmodel_, *data_, link_number, pinocchio::LOCAL, J);

        // Transform Jacobian from local frame to base frame
        J.topRows(3) = (data_->oMf[link_number].rotation()) * J.topRows(3);
        J.bottomRows(3) = (data_->oMf[link_number].rotation()) * J.bottomRows(3);

        return J;
    }

    /**
     * @brief Computes the linear velocity of a frame
     * @param frame_name Name of the frame
     * @return Linear velocity
     */
    Eigen::Vector3d linearVelocity(const std::string& frame_name) const {
        return (linearJacobian(frame_name) * qdot_);
    }

    /**
     * @brief Computes the angular velocity of a frame
     * @param frame_name Name of the frame
     * @return Angular velocity
     */
    Eigen::Vector3d angularVelocity(const std::string& frame_name) const {
        return (angularJacobian(frame_name) * qdot_);
    }

    /**
     * @brief Retrieves the position of a frame in 3D space
     * @param frame_name Name of the frame
     * @return Position vector of the frame
     */
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

    /**
     * @brief Computes the orientation of a specified frame in the robot model using quaternion
     * representation.
     * @param frame_name Name of the frame whose orientation is to be computed.
     * @return Eigen::Quaterniond representing the orientation of the specified frame.
     * @throws std::exception if the frame_name is invalid in the robot model.
     * @note This function retrieves the orientation from pinocchio::Data and converts it to
     * Eigen::Quaterniond.
     */
    Eigen::Quaterniond linkOrientation(const std::string& frame_name) const {
        try {
            // Get the frame index from the model based on the frame name
            pinocchio::Model::FrameIndex link_number = pmodel_->getFrameId(frame_name);

            // Convert rotation matrix of the frame to quaternion representation
            Eigen::Vector4d temp = rotationToQuaternion(data_->oMf[link_number].rotation());
            Eigen::Quaterniond tempQ;
            tempQ.w() = temp(0);
            tempQ.x() = temp(1);
            tempQ.y() = temp(2);
            tempQ.z() = temp(3);

            return tempQ;
        } catch (std::exception& e) {
            // Handle invalid frame name exception
            std::cerr << "WARNING: Frame name " << frame_name << " is invalid! ... "
                      << "Returning identity quaternion." << std::endl;
            return Eigen::Quaterniond::Identity();
        }
    }

    /**
     * @brief Retrieves the position and orientation of a specified frame in the robot model.
     * @param frame_name Name of the frame whose pose (position and orientation) is to be retrieved.
     * @return Eigen::VectorXd containing the pose of the specified frame (3 for translation, 4 for
     * quaternion).
     * @throws std::exception if the frame_name is invalid in the robot model.
     * @note This function retrieves both translation and quaternion representation of the frame's
     * orientation.
     */
    Eigen::VectorXd linkPose(const std::string& frame_name) const {
        Eigen::VectorXd lpose(7);  // Vector to store pose (translation + quaternion)

        // Get the frame index from the model based on the frame name
        pinocchio::Model::FrameIndex link_number = pmodel_->getFrameId(frame_name);

        // Retrieve translation and quaternion representation of the frame's orientation
        lpose.head(3) = data_->oMf[link_number].translation();
        lpose.tail(4) = rotationToQuaternion(data_->oMf[link_number].rotation());

        return lpose;
    }

    /**
     * @brief Retrieves the rigid body transformation of a specified frame in the robot model.
     * @param frame_name Name of the frame whose rigid body transformation is to be retrieved.
     * @return Eigen::Isometry3d representing the rigid body transformation of the specified frame.
     * @throws std::exception if the frame_name is invalid in the robot model.
     * @note This function retrieves both translation and rotation of the frame's rigid body
     * transformation.
     */
    Eigen::Isometry3d linkTF(const std::string& frame_name) const {
        Eigen::Isometry3d lpose = Eigen::Isometry3d::Identity();

        // Get the frame index from the model based on the frame name
        pinocchio::Model::FrameIndex link_number = pmodel_->getFrameId(frame_name);

        // Retrieve translation and quaternion representation of the frame's orientation
        lpose.translation() = data_->oMf[link_number].translation();
        lpose.linear() = data_->oMf[link_number].rotation();

        return lpose;
    }

    /**
     * @brief Computes the linear Jacobian matrix of a specified frame in the robot model.
     * @param frame_name Name of the frame whose linear Jacobian matrix is to be computed.
     * @return Eigen::MatrixXd representing the linear Jacobian matrix of the specified frame.
     * @throws std::exception if the frame_name is invalid in the robot model.
     * @note This function computes and transforms the Jacobian matrix from pinocchio::LOCAL frame
     * to the base frame.
     */
    Eigen::MatrixXd linearJacobian(const std::string& frame_name) const {
        pinocchio::Data::Matrix6x J(6, pmodel_->nv);  // Jacobian matrix placeholder
        J.fill(0);

        // Get the frame index from the model based on the frame name
        pinocchio::Model::FrameIndex link_number = pmodel_->getFrameId(frame_name);

        // Compute the Jacobian matrix in the pinocchio::LOCAL frame
        pinocchio::getFrameJacobian(*pmodel_, *data_, link_number, pinocchio::LOCAL, J);

        try {
            // Transform the Jacobian from pinocchio::LOCAL frame to base frame using rotation
            // matrix
            return (data_->oMf[link_number].rotation()) * J.topRows(3);
        } catch (std::exception& e) {
            // Handle invalid frame name exception
            std::cerr << "WARNING: Link name " << frame_name << " is invalid! ... "
                      << "Returning zeros." << std::endl;
            return Eigen::MatrixXd::Zero(3, ndofActuated());
        }
    }

    /**
     * @brief Computes the angular Jacobian matrix of a specified frame in the robot model.
     * @param frame_name Name of the frame whose angular Jacobian matrix is to be computed.
     * @return Eigen::MatrixXd representing the angular Jacobian matrix of the specified frame.
     * @throws std::exception if the frame_name is invalid in the robot model.
     * @note This function computes and transforms the Jacobian matrix from pinocchio::LOCAL frame
     * to the base frame.
     */
    Eigen::MatrixXd angularJacobian(const std::string& frame_name) const {
        pinocchio::Data::Matrix6x J(6, pmodel_->nv);  // Jacobian matrix placeholder
        J.fill(0);

        // Get the frame index from the model based on the frame name
        pinocchio::Model::FrameIndex link_number = pmodel_->getFrameId(frame_name);

        try {
            // Compute the Jacobian matrix in the pinocchio::LOCAL frame
            pinocchio::getFrameJacobian(*pmodel_, *data_, link_number, pinocchio::LOCAL, J);

            // Transform the Jacobian from pinocchio::LOCAL frame to base frame using rotation
            // matrix
            return (data_->oMf[link_number].rotation()) * J.bottomRows(3);
        } catch (std::exception& e) {
            // Handle invalid frame name exception
            std::cerr << "WARNING: Link name " << frame_name << " is invalid! ... "
                      << "Returning zeros." << std::endl;
            return Eigen::MatrixXd::Zero(3, ndofActuated());
        }
    }

    /**
     * @brief Computes recursively the total mass of the robot based on the urdf description of the
     * robot
     * @note Store the result to the member total_mass_
     */
    void computeTotalMass() {
        total_mass_ = 0.0;
        // Start from 1 since index 0 is typically the universe/world frame
        for (int i = 0; i < pmodel_->nbodies; ++i) {
            total_mass_ += pmodel_->inertias[i].mass();
        }
    }

    /**
     * @brief Returns the total mass of the robot
     */
    double getTotalMass() const {
        return total_mass_;
    }

    /**
     * @brief Computes the position of the center of mass (CoM) of the robot model.
     * @return Eigen::VectorXd representing the position of the CoM.
     * @note This function computes the CoM position using pinocchio::centerOfMass.
     */
    Eigen::VectorXd comPosition() const {
        pinocchio::centerOfMass(*pmodel_, *data_, q_);
        return data_->com[0];
    }

    /**
     * @brief Computes the Jacobian matrix of the center of mass (CoM) of the robot model.
     * @return Eigen::MatrixXd representing the Jacobian matrix of the CoM.
     * @note This function computes the Jacobian matrix of the CoM using
     * pinocchio::jacobianCenterOfMass.
     */
    Eigen::MatrixXd comJacobian() const {
        return pinocchio::jacobianCenterOfMass(*pmodel_, *data_, q_);
    }

    /**
     * @brief Computes the angular momentum of the center of mass (CoM) of the robot model.
     * @return Eigen::VectorXd representing the angular momentum of the CoM.
     * @note This function computes the angular momentum of the CoM using
     * pinocchio::computeCentroidalMomentum.
     */
    Eigen::VectorXd comAngularMomentum() const {
        pinocchio::computeCentroidalMomentum(*pmodel_, *data_, q_, qdot_);
        return data_->hg.angular();
    }

    /**
     * @brief Retrieves the names of all joints in the robot model.
     * @return std::vector<std::string> containing the names of all joints.
     * @note This function returns the stored joint names in the member variable jnames_.
     */
    std::vector<std::string> jointNames() const {
        return jnames_;
    }

    /**
     * @brief Retrieves the maximum angular limits of all joints in the robot model.
     * @return Eigen::VectorXd containing the maximum angular limits of all joints.
     * @note This function returns the stored maximum joint angular limits in the member variable
     * qmax_.
     */
    Eigen::VectorXd jointMaxAngularLimits() const {
        return qmax_;
    }

    /**
     * @brief Retrieves the minimum angular limits of all joints in the robot model.
     * @return Eigen::VectorXd containing the minimum angular limits of all joints.
     * @note This function returns the stored minimum joint angular limits in the member variable
     * qmin_.
     */
    Eigen::VectorXd jointMinAngularLimits() const {
        return qmin_;
    }

    /**
     * @brief Retrieves the velocity limits of all joints in the robot model.
     * @return Eigen::VectorXd containing the velocity limits of all joints.
     * @note This function returns the stored joint velocity limits in the member variable dqmax_.
     */
    Eigen::VectorXd jointVelocityLimits() const {
        return dqmax_;
    }

    /**
     * @brief Prints the names of all joints in the robot model to standard output.
     * @note This function iterates through jnames_ and prints each joint name.
     */
    void printJointNames() const {
        for (const auto& jname : jnames_) {
            std::cout << jname << std::endl;
        }
    }

    /**
     * @brief Retrieves the names of all joints in the robot model.
     * @return std::vector<std::string> containing the names of all joints.
     * @note This function returns the stored joint names in the member variable jnames_.
     */
    std::vector<std::string> getJointNames() const {
        return jnames_;
    }

    /**
     * @brief Prints the names and limits of all joints in the robot model to standard output.
     * @note This function checks if the sizes of jnames_, qmin_, qmax_, and dqmax_ match.
     *       It then prints the joint names, minimum angular limits (qmin_), maximum angular limits
     * (qmax_), and velocity limits (dqmax_) in a tabular format.
     */
    void printJointLimits() const {
        if (!((jnames_.size() == static_cast<size_t>(qmin_.size())) &&
              (jnames_.size() == static_cast<size_t>(qmax_.size())) &&
              (jnames_.size() == static_cast<size_t>(dqmax_.size())))) {
            std::cerr << "Joint names and joint limits size do not match!" << std::endl;
            return;
        }
        std::cout << "\nJoint Name\t qmin \t qmax \t dqmax" << std::endl;
        for (size_t i = 0; i < jnames_.size(); ++i)
            std::cout << jnames_[i] << "\t\t" << qmin_[i] << "\t" << qmax_[i] << "\t" << dqmax_[i]
                      << std::endl;
    }

    /**
     * @brief Returns the sign of a given double value.
     * @param x The input double value.
     * @return 1.0 if x >= 0, otherwise -1.0.
     * @note This function determines the sign of the input value x.
     */
    double sgn(const double& x) const {
        if (x >= 0) {
            return 1.0;
        } else {
            return -1.0;
        }
    }

    /**
     * @brief Converts a rotation matrix R into quaternion representation.
     * @param R The 3x3 rotation matrix.
     * @return Eigen::Vector4d representing the quaternion [w, x, y, z].
     * @note This function computes the quaternion from the given rotation matrix R,
     *       using a specific method that handles edge cases where the matrix is nearly singular.
     */
    Eigen::Vector4d rotationToQuaternion(const Eigen::Matrix3d& R) const {
        double eps = 1e-6;     // Small value for numerical stability
        Eigen::Vector4d quat;  // Quaternion representation [w, x, y, z]

        // Compute quaternion components
        quat(0) = 0.5 * sqrt(R(0, 0) + R(1, 1) + R(2, 2) + 1.0);  // w component

        // Determine x, y, z components, handling special cases to avoid division by zero
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

    /**
     * @brief Retrieves the names of all frames in the robot model.
     * @return std::vector<std::string> containing the names of all frames.
     * @note This function returns the stored frame names in the member variable frame_names_.
     */
    std::vector<std::string> frameNames() const {
        return frame_names_;
    }

private:
    /// Pinocchio model
    std::unique_ptr<pinocchio::Model> pmodel_;
    /// Pinocchio data
    std::unique_ptr<pinocchio::Data> data_;
    /// Joint names
    std::vector<std::string> jnames_;
    /// Joint position lower limits
    Eigen::VectorXd qmin_;
    /// Joint position upper limits
    Eigen::VectorXd qmax_;
    /// Joint velocity limits
    Eigen::VectorXd dqmax_;
    /// Joint positions
    Eigen::VectorXd q_;
    /// Joint velocities
    Eigen::VectorXd qdot_;
    /// Joint position measurement noises
    Eigen::VectorXd qn_;
    /// Total mass of the robot
    double total_mass_;
    /// Frame names
    std::vector<std::string> frame_names_;
};

}  // namespace serow
