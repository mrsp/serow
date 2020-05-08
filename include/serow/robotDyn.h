/*
 * SERoW - a complete state estimation scheme for humanoid robots
 *
 * Copyright 2018-2019 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH)
 *	 nor the names of its contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef __ROBOTDYN_H__
#define __ROBOTDYN_H__
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <string>
#include <vector>
#include <map>

#include <iostream>
#include <Eigen/Dense>


using namespace std;



namespace serow
{
    

    class robotDyn
    {
        
        
    private:
        pinocchio::Model *pmodel_;
        pinocchio::Data *data_;
        std::vector<std::string> jnames_;
        Eigen::VectorXd qmin_, qmax_, dqmax_, q_, qdot_, qn;
        bool has_floating_base_;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

          
        robotDyn(const std::string& model_name,
                 const bool& has_floating_base, const bool& verbose = false)
        {
            has_floating_base_ = has_floating_base;
            pmodel_ = new pinocchio::Model();
            
            if (has_floating_base)
                pinocchio::urdf::buildModel(model_name, pinocchio::JointModelFreeFlyer(),
                                      *pmodel_, verbose);
            else
                pinocchio::urdf::buildModel(model_name, *pmodel_, verbose);
            
            data_ = new pinocchio::Data(*pmodel_);
            
            jnames_.clear();
            int names_size=pmodel_->names.size();
            jnames_.reserve(names_size);
            
            for(int i=0;i<names_size;i++)
            {
                const std::string &jname=pmodel_->names[i];
                //Do not insert "universe" joint
                if (jname.compare("universe")!=0)
                {
                    jnames_.push_back(jname);
                }
            }                        
            
            qmin_.resize(jnames_.size());
            qmax_.resize(jnames_.size());
            dqmax_.resize(jnames_.size());
            qn.resize(jnames_.size());
            qmin_  = pmodel_->lowerPositionLimit;
            qmax_  = pmodel_->upperPositionLimit;
            dqmax_ = pmodel_->velocityLimit;
            
            
            
            // If free-floating base, eliminate the "root_joint" when displaying
            if (has_floating_base_)
            {
                jnames_.erase(jnames_.begin());
                Eigen::VectorXd tmp;
                tmp = qmin_; qmin_ = tmp.tail(jnames_.size());
                tmp = qmax_; qmax_ = tmp.tail(jnames_.size());
                tmp = dqmax_; dqmax_ = tmp.tail(jnames_.size());
            }
            
            // Continuous joints are given spurious values par default, set those values
            // to arbitrary ones
            for (int i=0; i<qmin_.size(); ++i)
            {
                double d = qmax_[i]-qmin_[i];
                // If wrong values or if difference less than 0.05 deg (0.001 rad)
                if ( (d<0) || (fabs(d)<0.001) )
                {
                    qmin_[i]  = -50.0;
                    qmax_[i]  =  50.0;
                    dqmax_[i] = 200.0;
                }
            }
            std::cout<<"Joint Names "<<std::endl;
            printJointNames();
	        std::cout<<"with "<<ndofActuated()<<" actuated joints"<<std::endl;
            std::cout << "Model loaded: " << model_name << std::endl;
	
	}
        
        
        int ndof()
        const
        {
            return pmodel_->nq;
        }

        int ndofActuated()
        const
        {
            if (has_floating_base_)
                // Eliminate the Cartesian position and orientation (quaternion)
                return pmodel_->nq-7;
            else
                return pmodel_->nq;
        }

        //void updateJointConfig(const Eigen::VectorXd& q)

        void updateJointConfig(std::map<std::string, double> qmap, std::map<std::string, double> qdotmap, double joint_std)
        {
            mapJointNamesIDs(qmap,qdotmap);
            if (has_floating_base_)
            {
                // Change quaternion order: in Pinocchio it is
                // (x,y,z,w)
                Eigen::VectorXd qpin; qpin = q_;
                qpin[3] = q_[4];
                qpin[4] = q_[5];
                qpin[5] = q_[6];
                qpin[6] = q_[3];
                //pinocchio::forwardKinematics(*pmodel_, *data_, qpin);
                pinocchio::framesForwardKinematics(*pmodel_, *data_, qpin);
                pinocchio::computeJointJacobians(*pmodel_, *data_, qpin);

            }
            else
            {                
                pinocchio::framesForwardKinematics(*pmodel_, *data_, q_);
                pinocchio::computeJointJacobians(*pmodel_, *data_, q_);
                
            }
             qn.setOnes();
             qn *= joint_std;
        }
        
        
        Eigen::Vector3d getLinearVelocityNoise(const std::string& frame_name)
        {
            return linearJacobian(frame_name) * qn;

        }
        Eigen::Vector3d getAngularVelocityNoise(const std::string& frame_name)
        {
            return angularJacobian(frame_name) * qn;

        }
        
        //TODO
        void mapJointNamesIDs(std::map<std::string, double> qmap, std::map<std::string, double> qdotmap)
        {
            q_.resize(pmodel_->nq);
            qdot_.resize(pmodel_->nv);

            for(int i = 0; i<jnames_.size();i++)
            {
                int jidx=pmodel_->getJointId(jnames_[i]);
                int qidx=pmodel_->idx_qs[jidx];
                int vidx=pmodel_->idx_vs[jidx];
                
                //this value is equal to 2 for continuous joints
                if(pmodel_->nqs[jidx]==2)
                {
                    q_[qidx] = cos(qmap[jnames_[i]]);
                    q_[qidx+1] = sin(qmap[jnames_[i]]);
                    qdot_[vidx] = qdotmap[jnames_[i]];  
                }
                else
                {
                    q_[qidx] = qmap[jnames_[i]];
                    qdot_[vidx] = qdotmap[jnames_[i]];
                }

            }         
        }
        
        
        Eigen::MatrixXd geometricJacobian(const std::string& frame_name)
        {
            try
            {
                pinocchio::Data::Matrix6x J(6,pmodel_->nv); J.fill(0);
                // Jacobian in pinocchio::LOCAL (link) frame
                pinocchio::Model::FrameIndex link_number =  pmodel_->getFrameId(frame_name);

                pinocchio::getFrameJacobian(*pmodel_, *data_, link_number, pinocchio::LOCAL, J);

                if (has_floating_base_)
                {
                    Eigen::MatrixXd Jg; Jg.resize(6, this->ndof()); Jg.setZero();
                    Eigen::Matrix3d Rbase; Rbase = quaternionToRotation(q_.segment(3,4));
                    Jg.topLeftCorner(3,3) =
                    Rbase.transpose()*data_->oMf[link_number].rotation()*J.block(0,0,3,3);
                    Jg.topRightCorner(3,ndofActuated()) =
                    (data_->oMf[link_number].rotation())*J.block(0,6,3,ndofActuated());
                    Jg.bottomRightCorner(3,ndofActuated()) =
                    (data_->oMf[link_number].rotation())*J.block(3,6,3,ndofActuated());
                    
                    Eigen::MatrixXd T; T.resize(3,4);
                    T <<
                    -2.0*q_(4),  2.0*q_(3), -2.0*q_(6),  2.0*q_(5),
                    -2.0*q_(5),  2.0*q_(6),  2.0*q_(3), -2.0*q_(4),
                    -2.0*q_(6), -2.0*q_(5),  2.0*q_(4),  2.0*q_(3);
                    Jg.block(0,3,3,4) =
                    data_->oMf[link_number].rotation()*J.block(0,3,3,3)*Rbase.transpose()*T;
                    Jg.block(3,3,3,4) = T;
                    
                            return Jg;
                        }
                    else
                    {
                    // Transform Jacobians from pinocchio::LOCAL frame to base frame
                    J.topRows(3) = (data_->oMf[link_number].rotation())*J.topRows(3);
                    J.bottomRows(3) = (data_->oMf[link_number].rotation())*J.bottomRows(3);
                    return J;
                }
            }
            catch (std::exception& e)
            {
                std::cerr << "WARNING: Link name " << frame_name << " is invalid! ... "
                <<  "Returning zeros" << std::endl;
                return Eigen::MatrixXd::Zero(6,ndofActuated());
            }
        }
        


  

        Eigen::Vector3d getLinearVelocity(const std::string& frame_name)
        {
            return  (linearJacobian(frame_name)* qdot_);
        }

        Eigen::Vector3d getAngularVelocity(const std::string& frame_name)
        {
            return  (angularJacobian(frame_name)* qdot_);
        }
        
     
        
      
        
        Eigen::Vector3d linkPosition(const std::string& frame_name)
        {
            try{ 
           	 pinocchio::Model::FrameIndex link_number =  pmodel_->getFrameId(frame_name);

           	 return data_->oMf[link_number].translation();
            }
            catch (std::exception& e)
            {
                std::cerr << "WARNING: Link name " << frame_name << " is invalid! ... "
                <<  "Returning zeros" << std::endl;
                return Eigen::Vector3d::Zero();
            }
        }
        
        
      
        
        Eigen::Quaterniond linkOrientation(const std::string& frame_name)
        {
            try{

                pinocchio::Model::FrameIndex link_number =  pmodel_->getFrameId(frame_name);

		Eigen::Vector4d temp = rotationToQuaternion(data_->oMf[link_number].rotation());
		    Eigen::Quaterniond tempQ;
		    tempQ.w() = temp(0);
		    tempQ.x() = temp(1);
		    tempQ.y() = temp(2);
		    tempQ.z() = temp(3);
		    return tempQ;
            }
            catch (std::exception& e)
            {
                std::cerr << "WARNING: Frame name " << frame_name << " is invalid! ... "
                <<  "Returning zeros" << std::endl;
                return Eigen::Quaterniond::Identity();
            }
        }
        
        
        Eigen::VectorXd linkPose(const std::string& frame_name)
        {
            Eigen::VectorXd lpose(7);
            pinocchio::Model::FrameIndex link_number =  pmodel_->getFrameId(frame_name);

            lpose.head(3) = data_->oMf[link_number].translation();
            lpose.tail(4) = rotationToQuaternion(data_->oMf[link_number].rotation());
            
            return lpose;
        }
        
        
        
        
       
        Eigen::MatrixXd linearJacobian(const std::string& frame_name)
        {
            pinocchio::Data::Matrix6x J(6, pmodel_->nv); J.fill(0);
            pinocchio::Model::FrameIndex link_number =  pmodel_->getFrameId(frame_name);

            pinocchio::getFrameJacobian(*pmodel_, *data_, link_number, pinocchio::LOCAL, J);
            try
            {
                if (has_floating_base_)
                {
                    // Structure of J is:
                    // [ Rworld_wrt_link*Rbase_wrt_world |
                    //   Rworld_wrt_link*skew(Pbase_wrt_world-Plink_wrt_world)*R_base_wrt_world |
                    //   Jq_wrt_link]
                    Eigen::MatrixXd Jlin; Jlin.resize(3, this->ndof()); Jlin.setZero();
                    Eigen::Matrix3d Rbase; Rbase = quaternionToRotation(q_.segment(3,4));
                    Jlin.leftCols(3) =
                    Rbase.transpose()*data_->oMf[link_number].rotation()*J.block(0,0,3,3);
                    Jlin.rightCols(ndofActuated()) =
                    (data_->oMf[link_number].rotation())*J.block(0,6,3,ndofActuated());
                    
                    Eigen::MatrixXd T; T.resize(3,4);
                    T <<
                    -2.0*q_(4),  2.0*q_(3), -2.0*q_(6),  2.0*q_(5),
                    -2.0*q_(5),  2.0*q_(6),  2.0*q_(3), -2.0*q_(4),
                    -2.0*q_(6), -2.0*q_(5),  2.0*q_(4),  2.0*q_(3);
                    Jlin.middleCols(3,4) =
                    data_->oMf[link_number].rotation()*J.block(0,3,3,3)*Rbase.transpose()*T;
                    
                    return Jlin;
                }
                else
                {
                    // Transform Jacobian from pinocchio::LOCAL frame to base frame
                    return (data_->oMf[link_number].rotation())*J.topRows(3);
                }
            }
            catch (std::exception& e)
            {
                std::cerr << "WARNING: Link name " << frame_name << " is invalid! ... "
                <<  "Returning zeros" << std::endl;
                return Eigen::MatrixXd::Zero(3,ndofActuated());
            }
        }
        

        Eigen::MatrixXd angularJacobian(const std::string& frame_name)
        {
            pinocchio::Data::Matrix6x J(6, pmodel_->nv); J.fill(0);
            pinocchio::Model::FrameIndex link_number =  pmodel_->getFrameId(frame_name);


            try
            {
                if (has_floating_base_)
                {

                    Eigen::MatrixXd Jang; 
                    Jang.resize(3, this->ndof()); Jang.setZero();
                    Eigen::Vector4d q;
                    
                    // Jacobian in global frame
                    pinocchio::getFrameJacobian(*pmodel_, *data_, link_number,pinocchio::LOCAL, J);
                    
                    // The structure of J is: [0 | Rot_ff_wrt_world | Jq_wrt_world]
                    Jang.rightCols(ndofActuated()) = J.block(3,6,3,ndofActuated());
                    q = rotationToQuaternion(J.block(3,3,3,3));
                    Jang.middleCols(3,4) <<
                    -2.0*q(1),  2.0*q(0), -2.0*q(3),  2.0*q(2),
                    -2.0*q(2),  2.0*q(3),  2.0*q(0), -2.0*q(1),
                    -2.0*q(3), -2.0*q(2),  2.0*q(1),  2.0*q(0);
                    return Jang;

                }
                else
                {

                    // Jacobian in pinocchio::LOCAL frame
                    pinocchio::getFrameJacobian(*pmodel_, *data_, link_number,pinocchio::LOCAL, J);

                    // Transform Jacobian from pinocchio::LOCAL frame to base frame
                    return (data_->oMf[link_number].rotation())*J.bottomRows(3);
                }
            }
            catch (std::exception& e)
            {
                std::cerr << "WARNING: Link name " << frame_name << " is invalid! ... "
                <<  "Returning zeros" << std::endl;
                return Eigen::MatrixXd::Zero(3,ndofActuated());
            }
        }
        
      
        
        
        Eigen::VectorXd comPosition()
        {
            Eigen::Vector3d com;
            
            if (has_floating_base_)
            {
                // Change quaternion order: in oscr it is (w,x,y,z) and in Pinocchio it is
                // (x,y,z,w)
                Eigen::VectorXd qpin;
                
                qpin = q_;
                qpin[3] = q_[4];
                qpin[4] = q_[5];
                qpin[5] = q_[6];
                qpin[6] = q_[3];
                //Eigen::Vector3d com = pinocchio::centerOfMass(*pmodel_, *data_, qpin);
                //std::cout << qpin.head(7).transpose() << std::endl;
                pinocchio::centerOfMass(*pmodel_, *data_, qpin);
                
                // Eigen::Matrix3d Rbase; Rbase = quaternionToRotation(q_.segment(3,4));
                // com = Rbase*data_->com[0];
                com = data_->com[0];
            }
            else
            {
                //Eigen::Vector3d com = pinocchio::centerOfMass(*pmodel_, *data_, q_);
                pinocchio::centerOfMass(*pmodel_, *data_, q_);
                com = data_->com[0];
            }
            
            return com;
            
        }
        
        
        Eigen::MatrixXd comJacobian()
        const
        {
            Eigen::MatrixXd Jcom;
            if (has_floating_base_)
            {
                // Change quaternion order: in oscr it is (w,x,y,z) and in Pinocchio it is
                // (x,y,z,w)
                Eigen::VectorXd qpin; qpin = q_;
                qpin[3] = q_[4];
                qpin[4] = q_[5];
                qpin[5] = q_[6];
                qpin[6] = q_[3];
                Jcom = pinocchio::jacobianCenterOfMass(*pmodel_, *data_, qpin);
            }
            else
            {
                Jcom = pinocchio::jacobianCenterOfMass(*pmodel_, *data_, q_);
            }
            return Jcom;
        }
        
        
        
        std::vector<std::string> jointNames()
        const
        {
            return jnames_;
        }
        
        
        Eigen::VectorXd jointMaxAngularLimits()
        const
        {
            return qmax_;
        }
        
        
        Eigen::VectorXd jointMinAngularLimits()
        const
        {
            return qmin_;
        }
        
        
        Eigen::VectorXd jointVelocityLimits()
        const
        {
            return dqmax_;
        }
        void printJointNames()
        const
        {
            // for (int i=0; i<jnames_.size(); ++i)
            //     std::cout << jnames_[i] <<std::endl;
                std::cout << *pmodel_ <<std::endl;


        }
        void printJointLimits()
        const
        {
            if (!( (jnames_.size() == qmin_.size()) &&
                  (jnames_.size() == qmax_.size()) &&
                  (jnames_.size() == dqmax_.size())) )
            {
                std::cerr << "Joint names and joint limits size do not match!"
                << std::endl;
                return;
            }
            std::cout << "\nJoint Name\t qmin \t qmax \t dqmax" << std::endl;
            for (int i=0; i<jnames_.size(); ++i)
                std::cout << jnames_[i] << "\t\t" << qmin_[i] << "\t" << qmax_[i] << "\t"
                << dqmax_[i] << std::endl;
        }
        
        
       
        
        double sgn(const double& x)
        {
            if (x>=0)
                return 1.0;
            else
                return -1.0;
        }
        
        Eigen::Vector4d rotationToQuaternion(const Eigen::Matrix3d& R)
        {
            double dEpsilon = 1e-6;
            Eigen::Vector4d quat;
            
            quat(0) = 0.5*sqrt(R(0,0)+R(1,1)+R(2,2)+1.0);
            if ( fabs(R(0,0)-R(1,1)-R(2,2)+1.0) < dEpsilon )
                quat(1) = 0.0;
            else
                quat(1) = 0.5*sgn(R(2,1)-R(1,2))*sqrt(R(0,0)-R(1,1)-R(2,2)+1.0);
            if ( fabs(R(1,1)-R(2,2)-R(0,0)+1.0) < dEpsilon )
                quat(2) = 0.0;
            else
                quat(2) = 0.5*sgn(R(0,2)-R(2,0))*sqrt(R(1,1)-R(2,2)-R(0,0)+1.0);
            if ( fabs(R(2,2)-R(0,0)-R(1,1)+1.0) < dEpsilon )
                quat(3) = 0.0;
            else
                quat(3) = 0.5*sgn(R(1,0)-R(0,1))*sqrt(R(2,2)-R(0,0)-R(1,1)+1.0);
            
            return quat;
        }
        
        
        Eigen::Matrix3d quaternionToRotation(const Eigen::Vector4d& q)
        {
            double normq = q.norm();
            if (fabs(normq-1.0)>0.001)
            {
                std::cerr << "WARNING: Input quaternion is not unitary! ... "
                << "Returning identity" << std::endl;
                return Eigen::Matrix3d::Identity();
            }
            Eigen::Matrix3d res;
            res(0,0) = 2.0*(q(0)*q(0)+q(1)*q(1))-1.0;
            res(0,1) = 2.0*(q(1)*q(2)-q(0)*q(3));
            res(0,2) = 2.0*(q(1)*q(3)+q(0)*q(2));
            res(1,0) = 2.0*(q(1)*q(2)+q(0)*q(3));
            res(1,1) = 2.0*(q(0)*q(0)+q(2)*q(2))-1.0;
            res(1,2) = 2.0*(q(2)*q(3)-q(0)*q(1));
            res(2,0) = 2.0*(q(1)*q(3)-q(0)*q(2));
            res(2,1) = 2.0*(q(2)*q(3)+q(0)*q(1));
            res(2,2) = 2.0*(q(0)*q(0)+q(3)*q(3))-1.0;
            
            return res;
        }
        
        
        
    };
}
#endif
