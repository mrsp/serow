/*
 * humanoid_state_estimation - a complete state estimation scheme for humanoid robots
 *
 * Copyright 2016-2017 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
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
 *     * Neither the name of the University of Freiburg nor the names of its
 *       contributors may be used to endorse or promote products derived from
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

#ifndef HUMANOID_EKF_H
#define HUMANOID_EKF_H

// ROS Headers
#include <ros/ros.h>

// Estimator Headers
#include "humanoid_state_estimation/IMUEKF.h"
#include "humanoid_state_estimation/CoMEKF.h"
#include "humanoid_state_estimation/JointSSKF.h"
#include "humanoid_state_estimation/LPF.h"
#include "humanoid_state_estimation/MovingAverageFilter.h"

#include <eigen3/Eigen/Dense>
// ROS Messages
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>

#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <hrl_kinematics/Kinematics.h>
#include <dynamic_reconfigure/server.h>
#include <humanoid_state_estimation/ParamControlConfig.h>



using namespace Eigen;
using namespace std;
using namespace hrl_kinematics;

class humanoid_ekf{
private:
	// ROS Standard Variables
	ros::NodeHandle n;

	ros::Publisher bodyPose_est_pub, bodyVel_est_pub, bodyAcc_est_pub,supportPose_est_pub,
	support_leg_pub, RLeg_est_pub, LLeg_est_pub, COP_pub,CoM_vel_pub,CoM_pos_pub, joint_filt_pub,
	external_force_filt_pub, odom_est_pub, leg_odom_pub,support_path_pub, odom_path_pub, leg_odom_path_pub,com_path_pub,cop_path_pub,
	ground_truth_odom_path_pub, ground_truth_com_path_pub, ground_truth_com_pub, CoM_odom_pub, ground_truth_odom_pub,ds_pub;
    ros::Subscriber imu_sub, joint_state_sub, pose_sub, lfsr_sub, rfsr_sub, odom_sub, copl_sub, copr_sub, ground_truth_odom_sub,ds_sub;
	
	double  freq, joint_freq, fsr_freq;
	ros::Time Tbs_stamp, Tbsw_stamp;
	bool fsr_inc, pose_inc, imu_inc, joint_inc, odom_inc, leg_odom_inc, support_inc, swing_inc, check_no_motion, ground_truth_odom_inc;
	bool firstOdom, firstUpdate, firstPose;
	int number_of_joints;
	bool firstGyrodot;
	bool useGyroLPF;
	int  maWindow;
	bool firstJoint;
	bool predictWithImu, predictWithCoM;
	bool no_motion_indicator;
	int no_motion_it, no_motion_it_threshold;
	double no_motion_threshold;
	Quaterniond  q_update;
	Vector3d  pos_update, CoM_gt;
	Affine3d T_B_I, T_B_P, Tib_gt;
	Quaterniond q_B_I, q_B_P, qib_gt;
	bool useJointKF, useCoMEKF, useLegOdom;
    bool visualize_with_rviz;
	//ROS Messages
	sensor_msgs::JointState joint_state_msg, joint_filt_msg;
	sensor_msgs::Imu imu_msg;
	nav_msgs::Odometry odom_msg, odom_msg_, odom_est_msg, leg_odom_msg, ground_truth_odom_msg, ground_truth_com_odom_msg, CoM_odom_msg;
	nav_msgs::Path odom_path_msg, leg_odom_path_msg, com_path_msg, cop_path_msg, support_path_msg, ground_truth_odom_path_msg, ground_truth_com_path_msg;
	geometry_msgs::PoseStamped pose_msg, pose_msg_, temp_pose_msg;
	std_msgs::String support_leg_msg;
	geometry_msgs::WrenchStamped RLeg_est_msg, LLeg_est_msg, lfsr_msg, rfsr_msg, external_force_filt_msg;
   
    geometry_msgs::PoseStamped bodyPose_est_msg, supportPose_est_msg;
	geometry_msgs::TwistStamped bodyVel_est_msg, CoM_vel_msg;
	sensor_msgs::Imu  bodyAcc_est_msg;
	std_msgs::Int32 is_in_ds_msg;
	geometry_msgs::PointStamped COP_msg, copl_msg, copr_msg, CoM_pos_msg;

	// Helper
	bool is_connected_, ground_truth;

	tf::TransformListener Tbs_listener, Tbsw_listener;
	tf::StampedTransform Tbs_tf, Tbsw_tf;
	Quaterniond qbs, qbsw, qwb, qwb_;
	string base_link_frame, swing_foot_frame, support_foot_frame, lfoot_frame, rfoot_frame;
	
	tf::TransformListener Tfsr_listener;
	tf::StampedTransform Tfsr_tf;


    boost::shared_ptr< dynamic_reconfigure::Server<humanoid_state_estimation::ParamControlConfig> > dynamic_recfg_;

	// get joint positions from state message
  	std::map<std::string, double> joint_map;
	tf::Point com;
	tf::Transform tf_right_foot, tf_left_foot;
	double mass;
	IMUEKF* imuEKF;
	Kinematics* kin;
	CoMEKF* nipmEKF;
	LPF* gyroLPF;
	MovingAverageFilter** gyroMAF;
	//Cuttoff Freqs for LPF
	double gyro_fx, gyro_fy, gyro_fz;
	Vector3d COP_fsr, GRF_fsr, CoM_enc, Gyrodot, Gyro_;
	double bias_fx,bias_fy,bias_fz;
	JointSSKF** JointKF;
	double jointFreq;

	string support_leg, swing_leg;

	Vector3d LLegGRF, RLegGRF;
  	Vector3d copl, copr;

	Affine3d Tws, Twh, Twb, Twb_; //From support s to world frame;
	Affine3d Tbs, Tsb, Tssw, Tbsw, Tbs_, Tbsw_;
	Vector3d no_motion_residual;
	/****/
	bool firstrun, legSwitch, firstContact;
	
	double LLegUpThres, LLegLowThres, LosingContact, StrikingContact;
	double bias_ax, bias_ay, bias_az, bias_gx, bias_gy, bias_gz;
	double g, m, I_xx, I_yy, I_zz;
	/** Real odometry Data **/
     string lfsr_topic,rfsr_topic,copl_topic,copr_topic;
	 string pose_topic;
	 string imu_topic;
	 string joint_state_topic;
	 string odom_topic;
	 string ground_truth_odom_topic, is_in_ds_topic;

	 bool usePoseUpdate;

	//Odometry, from supportleg to inertial, transformation from support leg to other leg
     void subscribeToIMU();
	 void subscribeToFSR();
	 void subscribeToJointState();
 	 void subscribeToPose();
	 void subscribeToOdom();
	 void subscribeToGroundTruth();
	 void subscribeToDS();
	 void is_in_dsCb(const std_msgs::Bool::ConstPtr& msg);
	 void ground_truth_odomCb(const nav_msgs::Odometry::ConstPtr& msg);
	 void imuCb(const sensor_msgs::Imu::ConstPtr& msg);
	 void poseCb(const geometry_msgs::PoseStamped::ConstPtr& msg);
	 void joint_stateCb(const sensor_msgs::JointState::ConstPtr& msg);
	 void odomCb(const nav_msgs::Odometry::ConstPtr& msg);
	 
	 void lfsrCb(const geometry_msgs::WrenchStamped::ConstPtr& msg);
	 void rfsrCb(const geometry_msgs::WrenchStamped::ConstPtr& msg);
	 void coplCb(const geometry_msgs::PointStamped::ConstPtr& msg);
	 void coprCb(const geometry_msgs::PointStamped::ConstPtr& msg);


	 void computeLGRF();
	 void computeRGRF();
	 void computeCOP(Affine3d Tis_, Affine3d Tssprime_);
	 void filterGyrodot();
	//private methods
	void init();

	void estimateWithCoMEKF();
	void estimateWithIMUEKF();



	void determineLegContact();


	void computeKinTFs();
	//publish functions
	void publishGRF();
	void publishJointEstimates();
	void publishCoMEstimates();
	void deAllocate();
	void publishSupportEstimates();
	void publishBodyEstimates();
	void publishContact();
	void publishCOP();
	// Advertise to ROS Topics
	void advertise();
	void subscribe();

	Affine3d isNear(Affine3d T_1, Affine3d T_2, float epsx, float epsy, float epsz){
		Affine3d T;
		T.translation() = T_1.translation();
		T.linear() = T_2.linear();

		if(abs(T_1.translation()(0) - T_2.translation()(0)) > epsx)
			T.translation()(0) = T_2.translation()(0);
		if(abs(T_1.translation()(1) - T_2.translation()(1)) > epsy)
			T.translation()(1) = T_2.translation()(1);
		if(abs(T_1.translation()(2) - T_2.translation()(2)) > epsz)
			T.translation()(2) = T_2.translation()(2);

		return T;
	}


public:




	// Constructor/Destructor
	humanoid_ekf();

	~humanoid_ekf();

	// Connect/Disconnet to ALProxies
	bool connect(const ros::NodeHandle nh);

	void disconnect();



	// Parameter Server
	void loadparams();
	void loadIMUEKFparams();
	void loadCoMEKFparams();
	void loadJointKFparams();
	// General Methods
	void reconfigureCB(humanoid_state_estimation::ParamControlConfig& config, uint32_t level);
	void run();

	bool connected();


};

#endif // HUMANOID_EKF_H
