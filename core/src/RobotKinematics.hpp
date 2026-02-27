/**
 * Copyright (C) Stylianos Piperakis, Ownage Dynamics L.P.
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
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <stdexcept>
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
     * @param model_name Name of the model file for the robot
     * @param joint_position_variance Per-joint position measurement noise variance
     * @param verbose Verbosity flag for model loading (default: false)
     */
    RobotKinematics(const std::string& model_name, double joint_position_variance,
                    bool verbose = false) {
        // Create the model
        pmodel_ = std::make_unique<pinocchio::Model>();

        // Build the kinematic model from file
        try {
            std::cout << "Building model " << model_name << std::endl;
            if (model_name.find(".xml") != std::string::npos) {
                pinocchio::mjcf::buildModel(model_name, *pmodel_, verbose);
            } else if (model_name.find(".urdf") != std::string::npos) {
                pinocchio::urdf::buildModel(model_name, *pmodel_, verbose);
            } else {
                throw std::runtime_error(
                    "Failed to load model from '" + model_name +
                    "': Unsupported file type, supported types are .xml and .urdf");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load model from '" + model_name + "': " + e.what());
        }

        // Verify the model was loaded successfully
        if (pmodel_->nq == 0) {
            throw std::runtime_error(
                "Model loaded but has 0 degrees of freedom. Check if the file is valid: " +
                model_name);
        }

        // Enforce fixed-base behavior by default by locking any root free-flyer joint.
        // NOTE: data_ is intentionally created AFTER this call since buildReducedModel
        // replaces *pmodel_ with a new model object, invalidating any previously created Data.
        reduceRootFreeFlyerToFixedBase();

        // Additional safety checks
        if (pmodel_->lowerPositionLimit.size() == 0 || pmodel_->upperPositionLimit.size() == 0 ||
            pmodel_->velocityLimit.size() == 0) {
            throw std::runtime_error(
                "Model loaded but joint limits are empty. Check if the file is valid: " +
                model_name);
        }

        // Create the data associated with the (possibly reduced) model
        data_ = std::make_unique<pinocchio::Data>(*pmodel_);

        // Initialize joint names excluding the "universe" joint and floating-base variants
        jnames_.reserve(pmodel_->names.size());
        for (const auto& jname : pmodel_->names) {
            if (jname.find("universe") == std::string::npos &&
                jname.find("world") == std::string::npos &&
                jname.find("free") == std::string::npos &&
                jname.find("floating") == std::string::npos) {
                jnames_.push_back(jname);
            }
        }

        // Initialize the vectors with proper sizes before assignment
        qmin_.resize(jnames_.size());
        qmax_.resize(jnames_.size());
        dqmax_.resize(jnames_.size());

        // Map limits by joint ids to keep them aligned with filtered joint names
        for (size_t i = 0; i < jnames_.size(); i++) {
            const pinocchio::JointIndex jidx = pmodel_->getJointId(jnames_[i]);
            const pinocchio::JointIndex qidx = pmodel_->idx_qs[jidx];
            const pinocchio::JointIndex vidx = pmodel_->idx_vs[jidx];
            qmin_[i] = (pmodel_->nqs[jidx] > 0) ? pmodel_->lowerPositionLimit[qidx] : 0.0;
            qmax_[i] = (pmodel_->nqs[jidx] > 0) ? pmodel_->upperPositionLimit[qidx] : 0.0;
            dqmax_[i] = (pmodel_->nvs[jidx] > 0) ? pmodel_->velocityLimit[vidx] : 0.0;
        }

        qn_ = Eigen::VectorXd::Ones(jnames_.size()) * joint_position_variance;

        // Initialize state vectors
        q_.setZero(pmodel_->nq);
        qdot_.setZero(pmodel_->nv);

        // Set default values for continuous joints if limits are invalid
        for (int i = 0; i < qmin_.size(); i++) {
            const double d = qmax_[i] - qmin_[i];
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
     * @brief Destructor
     */
    ~RobotKinematics() = default;

    /**
     * @brief Returns the configuration space dimension of the model (nq).
     * @note For models with continuous joints nq > nv; prefer ndofActuated() for velocity-space
     * operations.
     */
    int ndof() const {
        return pmodel_->nq;
    }

    /**
     * @brief Returns the velocity space dimension of the model (nv), i.e. true actuated DoF.
     */
    int ndofActuated() const {
        return pmodel_->nv;
    }

    /**
     * @brief Updates the joint configuration and kinematic data
     * @param qmap Map of joint names to their positions
     * @param qdotmap Map of joint names to their velocities
     */
    void updateJointConfig(const std::map<std::string, double>& qmap,
                           const std::map<std::string, double>& qdotmap) {
        mapJointNamesIDs(qmap, qdotmap);
        pinocchio::framesForwardKinematics(*pmodel_, *data_, q_);
        pinocchio::computeJointJacobians(*pmodel_, *data_, q_);
    }

    /**
     * @brief Computes the linear velocity covariance of a frame from independent joint noise.
     * @param frame_name Name of the frame
     * @return 3x3 linear velocity covariance matrix
     */
    Eigen::Matrix3d linearVelocityCovariance(const std::string& frame_name) const {
        const Eigen::MatrixXd J = linearJacobian(frame_name);
        return J * qn_.asDiagonal() * J.transpose();
    }

    /**
     * @brief Computes the angular velocity covariance of a frame from independent joint noise.
     * @param frame_name Name of the frame
     * @return 3x3 angular velocity covariance matrix
     */
    Eigen::Matrix3d angularVelocityCovariance(const std::string& frame_name) const {
        const Eigen::MatrixXd J = angularJacobian(frame_name);
        return J * qn_.asDiagonal() * J.transpose();
    }

    /**
     * @brief Maps joint names to their respective indices and assigns values for q_ and qdot_
     * @param qmap Map of joint names to their positions
     * @param qdotmap Map of joint names to their velocities
     */
    void mapJointNamesIDs(const std::map<std::string, double>& qmap,
                          const std::map<std::string, double>& qdotmap) {
        q_.setZero(pmodel_->nq);
        qdot_.setZero(pmodel_->nv);

        for (size_t i = 0; i < jnames_.size(); i++) {
            const size_t jidx = pmodel_->getJointId(jnames_[i]);
            const size_t qidx = pmodel_->idx_qs[jidx];
            const size_t vidx = pmodel_->idx_vs[jidx];

            if (pmodel_->nqs[jidx] == 2) {
                // Continuous (revolute unbounded) joint stored as (cos, sin)
                const double angle = qmap.at(jnames_[i]);
                q_[qidx] = std::cos(angle);
                q_[qidx + 1] = std::sin(angle);
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
        pinocchio::Data::Matrix6x J = pinocchio::Data::Matrix6x::Zero(6, pmodel_->nv);

        const pinocchio::Model::FrameIndex fid = pmodel_->getFrameId(frame_name, pinocchio::BODY);
        if (fid >= static_cast<pinocchio::Model::FrameIndex>(pmodel_->nframes)) {
            std::cerr << "WARNING: Link name <" << frame_name << "> is invalid! "
                      << "Returning zeros." << std::endl;
            return Eigen::MatrixXd::Zero(6, pmodel_->nv);
        }

        pinocchio::getFrameJacobian(*pmodel_, *data_, fid, pinocchio::LOCAL, J);

        const Eigen::Matrix3d& R = data_->oMf[fid].rotation();
        J.topRows(3) = R * J.topRows(3);
        J.bottomRows(3) = R * J.bottomRows(3);
        return J;
    }

    /**
     * @brief Computes the linear velocity of a frame
     * @param frame_name Name of the frame
     * @return Linear velocity
     */
    Eigen::Vector3d linearVelocity(const std::string& frame_name) const {
        return linearJacobian(frame_name) * qdot_;
    }

    /**
     * @brief Computes the angular velocity of a frame
     * @param frame_name Name of the frame
     * @return Angular velocity
     */
    Eigen::Vector3d angularVelocity(const std::string& frame_name) const {
        return angularJacobian(frame_name) * qdot_;
    }

    /**
     * @brief Retrieves the position of a frame in 3D space
     * @param frame_name Name of the frame
     * @return Position vector of the frame
     */
    Eigen::Vector3d linkPosition(const std::string& frame_name) const {
        const pinocchio::Model::FrameIndex fid = pmodel_->getFrameId(frame_name, pinocchio::BODY);
        if (fid >= static_cast<pinocchio::Model::FrameIndex>(pmodel_->nframes)) {
            std::cerr << "WARNING: Link name " << frame_name << " is invalid! "
                      << "Returning zeros." << std::endl;
            return Eigen::Vector3d::Zero();
        }
        return data_->oMf[fid].translation();
    }

    /**
     * @brief Retrieves the orientation of a frame as a quaternion.
     * @param frame_name Name of the frame whose orientation is to be computed.
     * @return Eigen::Quaterniond (identity on invalid frame name)
     */
    Eigen::Quaterniond linkOrientation(const std::string& frame_name) const {
        const pinocchio::Model::FrameIndex fid = pmodel_->getFrameId(frame_name, pinocchio::BODY);
        if (fid >= static_cast<pinocchio::Model::FrameIndex>(pmodel_->nframes)) {
            std::cerr << "WARNING: Frame name " << frame_name << " is invalid! "
                      << "Returning identity quaternion." << std::endl;
            return Eigen::Quaterniond::Identity();
        }
        return Eigen::Quaterniond(data_->oMf[fid].rotation());
    }

    /**
     * @brief Retrieves the pose of a frame as a 7-vector [tx, ty, tz, qw, qx, qy, qz].
     * @param frame_name Name of the frame whose pose is to be computed.
     * @return Pose vector of the frame
     */
    Eigen::VectorXd linkPose(const std::string& frame_name) const {
        Eigen::VectorXd lpose(7);

        const pinocchio::Model::FrameIndex fid = pmodel_->getFrameId(frame_name, pinocchio::BODY);
        if (fid >= static_cast<pinocchio::Model::FrameIndex>(pmodel_->nframes)) {
            std::cerr << "WARNING: Frame name " << frame_name << " is invalid! "
                      << "Returning zero pose." << std::endl;
            lpose.setZero();
            lpose(3) = 1.0;  // identity quaternion w=1
            return lpose;
        }

        lpose.head(3) = data_->oMf[fid].translation();
        const Eigen::Quaterniond q(data_->oMf[fid].rotation());
        lpose(3) = q.w();
        lpose(4) = q.x();
        lpose(5) = q.y();
        lpose(6) = q.z();
        return lpose;
    }

    /**
     * @brief Retrieves the rigid-body transform of a frame as an Eigen::Isometry3d.
     * @param frame_name Name of the frame whose transform is to be computed.
     * @return Rigid-body transform of the frame
     */
    Eigen::Isometry3d linkTF(const std::string& frame_name) const {
        Eigen::Isometry3d lpose = Eigen::Isometry3d::Identity();

        const pinocchio::Model::FrameIndex fid = pmodel_->getFrameId(frame_name, pinocchio::BODY);
        if (fid >= static_cast<pinocchio::Model::FrameIndex>(pmodel_->nframes)) {
            std::cerr << "WARNING: Frame name " << frame_name << " is invalid! "
                      << "Returning identity transform." << std::endl;
            return lpose;
        }

        lpose.translation() = data_->oMf[fid].translation();
        lpose.linear() = data_->oMf[fid].rotation();
        return lpose;
    }

    /**
     * @brief Computes the linear (translational) Jacobian (3 x nv).
     * @param frame_name Name of the frame whose linear Jacobian is to be computed.
     * @return Linear Jacobian matrix
     */
    Eigen::MatrixXd linearJacobian(const std::string& frame_name) const {
        pinocchio::Data::Matrix6x J = pinocchio::Data::Matrix6x::Zero(6, pmodel_->nv);

        const pinocchio::Model::FrameIndex fid = pmodel_->getFrameId(frame_name, pinocchio::BODY);
        if (fid >= static_cast<pinocchio::Model::FrameIndex>(pmodel_->nframes)) {
            std::cerr << "WARNING: Link name " << frame_name << " is invalid! "
                      << "Returning zeros." << std::endl;
            return Eigen::MatrixXd::Zero(3, pmodel_->nv);
        }

        pinocchio::getFrameJacobian(*pmodel_, *data_, fid, pinocchio::LOCAL, J);
        return data_->oMf[fid].rotation() * J.topRows(3);
    }

    /**
     * @brief Computes the angular Jacobian (3 x nv).
     * @param frame_name Name of the frame whose angular Jacobian is to be computed.
     * @return Angular Jacobian matrix
     */
    Eigen::MatrixXd angularJacobian(const std::string& frame_name) const {
        pinocchio::Data::Matrix6x J = pinocchio::Data::Matrix6x::Zero(6, pmodel_->nv);

        const pinocchio::Model::FrameIndex fid = pmodel_->getFrameId(frame_name, pinocchio::BODY);
        if (fid >= static_cast<pinocchio::Model::FrameIndex>(pmodel_->nframes)) {
            std::cerr << "WARNING: Link name " << frame_name << " is invalid! "
                      << "Returning zeros." << std::endl;
            return Eigen::MatrixXd::Zero(3, pmodel_->nv);
        }

        pinocchio::getFrameJacobian(*pmodel_, *data_, fid, pinocchio::LOCAL, J);
        return data_->oMf[fid].rotation() * J.bottomRows(3);
    }

    /**
     * @brief Computes the total mass of the robot from all body inertias.
     */
    void computeTotalMass() {
        total_mass_ = 0.0;
        for (int i = 0; i < pmodel_->nbodies; ++i) {
            total_mass_ += pmodel_->inertias[i].mass();
        }
    }

    /**
     * @brief Gets the total mass of the robot.
     * @return Total mass of the robot
     */
    double getTotalMass() const {
        return total_mass_;
    }

    /**
     * @brief Computes the CoM position.
     * @return CoM position vector
     */
    Eigen::VectorXd comPosition() const {
        pinocchio::centerOfMass(*pmodel_, *data_, q_);
        return data_->com[0];
    }

    /**
     * @brief Computes the CoM Jacobian (3 x nv).
     * @return CoM Jacobian matrix
     */
    Eigen::MatrixXd comJacobian() const {
        return pinocchio::jacobianCenterOfMass(*pmodel_, *data_, q_);
    }

    /**
     * @brief Computes the CoM angular momentum and its covariance in a single Pinocchio pass.
     * @return {h_angular (3D), covariance (3x3)}
     */
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> comAngularMomentumAndCovariance() const {
        pinocchio::computeCentroidalMap(*pmodel_, *data_, q_);
        const Eigen::MatrixXd Ag_angular = data_->Ag.bottomRows(3);
        const Eigen::Vector3d h_angular = Ag_angular * qdot_;
        const Eigen::Matrix3d cov = Ag_angular * qn_.asDiagonal() * Ag_angular.transpose();
        return {h_angular, cov};
    }

    /**
     * @brief Computes the CoM angular momentum.
     * @note Prefer comAngularMomentumAndCovariance() if you also need the covariance.
     */
    Eigen::VectorXd comAngularMomentum() const {
        pinocchio::computeCentroidalMomentum(*pmodel_, *data_, q_, qdot_);
        return data_->hg.angular();
    }

    /**
     * @brief Computes the CoM angular momentum covariance from joint noise.
     * @note Prefer comAngularMomentumAndCovariance() if you also need the momentum.
     */
    Eigen::Matrix3d comAngularMomentumCovariance() const {
        pinocchio::computeCentroidalMap(*pmodel_, *data_, q_);
        const Eigen::MatrixXd Ag_angular = data_->Ag.bottomRows(3);
        return Ag_angular * qn_.asDiagonal() * Ag_angular.transpose();
    }

    /**
     * @brief Retrieves the names of the joints
     * @return Vector of joint names
     */
    std::vector<std::string> jointNames() const {
        return jnames_;
    }

    /**
     * @brief Retrieves the names of the frames
     * @return Vector of frame names
     */
    std::vector<std::string> frameNames() const {
        return frame_names_;
    }

    /**
     * @brief Retrieves the maximum angular limits of the joints
     * @return Vector of maximum angular limits
     */
    Eigen::VectorXd jointMaxAngularLimits() const {
        return qmax_;
    }

    /**
     * @brief Retrieves the minimum angular limits of the joints
     * @return Vector of minimum angular limits
     */
    Eigen::VectorXd jointMinAngularLimits() const {
        return qmin_;
    }

    /**
     * @brief Retrieves the velocity limits of the joints
     * @return Vector of velocity limits
     */
    Eigen::VectorXd jointVelocityLimits() const {
        return dqmax_;
    }

    /**
     * @brief Prints the names of the joints
     */
    void printJointNames() const {
        for (const auto& jname : jnames_) {
            std::cout << jname << std::endl;
        }
    }

    /**
     * @brief Prints the joint limits
     */
    void printJointLimits() const {
        if (jnames_.size() != static_cast<size_t>(qmin_.size()) ||
            jnames_.size() != static_cast<size_t>(qmax_.size()) ||
            jnames_.size() != static_cast<size_t>(dqmax_.size())) {
            std::cerr << "Joint names and joint limits size do not match!" << std::endl;
            return;
        }
        std::cout << "\nJoint Name\t qmin \t qmax \t dqmax" << std::endl;
        for (size_t i = 0; i < jnames_.size(); ++i) {
            std::cout << jnames_[i] << "\t\t" << qmin_[i] << "\t" << qmax_[i] << "\t" << dqmax_[i]
                      << std::endl;
        }
    }

private:
    /// Pinocchio model
    std::unique_ptr<pinocchio::Model> pmodel_;
    /// Pinocchio data (always consistent with *pmodel_)
    std::unique_ptr<pinocchio::Data> data_;
    /// Filtered joint names (excludes universe / floating-base joints)
    std::vector<std::string> jnames_;
    /// Joint position lower limits
    Eigen::VectorXd qmin_;
    /// Joint position upper limits
    Eigen::VectorXd qmax_;
    /// Joint velocity limits
    Eigen::VectorXd dqmax_;
    /// Joint positions (size nq)
    Eigen::VectorXd q_;
    /// Joint velocities (size nv)
    Eigen::VectorXd qdot_;
    /// Per-joint position noise variance (size == jnames_.size())
    Eigen::VectorXd qn_;
    /// Total robot mass
    double total_mass_{0.0};
    /// BODY frame names
    std::vector<std::string> frame_names_;

    /**
     * @brief Locks root free-flyer joints so the model is fixed-base by default.
     *
     * Iterates all joints whose parent is the universe (index 0) and whose
     * configuration/velocity dimensions match a free-flyer (nq=7, nv=6).
     * buildReducedModel replaces *pmodel_ in-place; data_ must be (re-)created
     * after this call, which the constructor guarantees.
     */
    void reduceRootFreeFlyerToFixedBase() {
        std::vector<pinocchio::JointIndex> joints_to_lock;
        joints_to_lock.reserve(pmodel_->njoints);  // upper bound; avoids misleading reserve(1)

        for (pinocchio::JointIndex jid = 1;
             jid < static_cast<pinocchio::JointIndex>(pmodel_->njoints); ++jid) {
            const bool is_root_joint = (pmodel_->parents[jid] == 0);
            // Prefer the joint type name to be robust against future convention changes.
            const bool is_free_flyer = (pmodel_->joints[jid].shortname() == "JointModelFreeFlyer");
            if (is_root_joint && is_free_flyer) {
                joints_to_lock.push_back(jid);
            }
        }

        if (!joints_to_lock.empty()) {
            const Eigen::VectorXd qref = pinocchio::neutral(*pmodel_);
            *pmodel_ = pinocchio::buildReducedModel(*pmodel_, joints_to_lock, qref);
        }
    }
};

}  // namespace serow
