/*
 * SEROW - a complete state estimation scheme for humanoid robots
 *
 * Copyright 2017-2018 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
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

#ifndef QUADRUPED_EKF_H
#define QUADRUPED_EKF_H

// ROS Headers
#include <ros/ros.h>

// Estimator Headers
#include <serow/IMUinEKF.h>
#include <serow/IMUEKF.h>
#include <serow/CoMEKF.h>
#include <serow/JointDF.h>
#include <serow/butterworthLPF.h>
#include <serow/MovingAverageFilter.h>

#include <eigen3/Eigen/Dense>
// ROS Messages
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>

#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>

#include <dynamic_reconfigure/server.h>
#include <serow/VarianceControlConfig.h>
#include <serow/mediator.h>
#include <serow/differentiator.h>
#include <serow/robotDyn.h>
#include <serow/Madgwick.h>
#include <serow/Mahony.h>
#include <serow/deadReckoningQuad.h>
#include <serow/ContactDetectionQuad.h>

using namespace Eigen;
using namespace std;

class quadruped_ekf{
private:
	// ROS Standard Variables
	ros::NodeHandle n;
	ros::Publisher supportPose_est_pub, bodyAcc_est_pub, LFLeg_odom_pub, LHLeg_odom_pub, RFLeg_odom_pub, RHLeg_odom_pub,
	support_leg_pub, RFLeg_est_pub, RHLeg_est_pub, LFLeg_est_pub, LHLeg_est_pub, COP_pub, joint_filt_pub, rel_CoMPose_pub,
	external_force_filt_pub, odom_est_pub, leg_odom_pub, ground_truth_com_pub, CoM_odom_pub, 
	CoM_leg_odom_pub, ground_truth_odom_pub, ds_pub, 
	rel_LFLegPose_pub, rel_LHLegPose_pub, rel_RFLegPose_pub, rel_RHLegPose_pub, comp_odom0_pub;
    
	ros::Subscriber imu_sub, joint_state_sub, LFft_sub, LHft_sub, RFft_sub, RHft_sub, odom_sub, 
	ground_truth_odom_sub, ds_sub, ground_truth_com_sub, support_idx_sub, compodom0_sub;
	
	Eigen::VectorXd joint_state_pos,joint_state_vel;

	Eigen::Vector3d omegabLF, omegabLH, omegabRF, omegabRH, vbLF, vbLH, vbRF, vbRH, vbLFn, vbRFn, vbLHn, vbRHn, vwb, omegawb, vwLF, vwLH, vwRF, vwRH, omegawLF, omegawLH, omegawRF, omegawRH, p_FT_LH, p_FT_RH, p_FT_LF, p_FT_RF;
	Eigen::Matrix3d JLFQnJLFt, JLHQnJLHt,  JRFQnJRFt, JRHQnJRHt;
	Affine3d TwLF, TwLH, TwRF, TwRH, TbLF, TbLH,  TbRF, TbRH;
	Affine3d T_B_A, T_B_G, T_B_P, T_FT_RH, T_FT_RF, T_FT_LF, T_FT_LH, T_B_GT;
	Quaterniond qbs, qbLF, qbLH, qbRF, qbRH, qwb, qwb_, qws, qwLF, qwLH, qwRF, qwRH;
	string base_link_frame, support_foot_frame, LFfoot_frame,  LHfoot_frame, RFfoot_frame, RHfoot_frame;


	//Wrapper of Pinnochio
	serow::robotDyn* rd;
	//Orientation Estimators based on IMU
    bool useMahony;
	serow::Madgwick* mw;
    serow::Mahony* mh;
    double Kp, Ki;

	//Leg Odometry Computation
	serow::deadReckoningQuad* dr;
	//Joint State Estimator 
  	std::map<std::string, double> joint_state_pos_map, joint_state_vel_map;
	JointDF** JointVF;
	double jointFreq,joint_cutoff_freq;

	double  Tau0, Tau1, VelocityThres;

	double  freq, joint_freq, fsr_freq;
	bool LFfsr_inc, LHfsr_inc, RHfsr_inc, RFfsr_inc, LFft_inc, LHft_inc, RFft_inc, RHft_inc;
	bool imu_inc, joint_inc, odom_inc, leg_odom_inc, leg_vel_inc, support_inc, check_no_motion, com_inc, ground_truth_odom_inc;
	bool firstOdom, firstUpdate, odom_divergence;
	int number_of_joints, outlier_count;
	bool firstGyrodot;
	bool useGyroLPF;
	int  maWindow;
	int medianWindow;
	bool firstJointStates;
	bool predictWithImu, predictWithCoM;
	bool no_motion_indicator, outlier;
	int no_motion_it, no_motion_it_threshold;
	double no_motion_threshold;
	Quaterniond  q_update, q_update_, q_leg_update, q_now, q_prev;
	Vector3d  pos_update, pos_update_, pos_leg_update, CoM_gt, temp, gt_odom;
	Quaterniond  q_B_P, q_B_GT, tempq, qoffsetGTCoM, tempq_, gt_odomq; 
	bool useCoMEKF, useLegOdom, firstGT,firstGTCoM, useOutlierDetection;
    bool debug_mode;

	//ROS Messages
	sensor_msgs::JointState joint_state_msg, joint_filt_msg;
	sensor_msgs::Imu imu_msg;
	nav_msgs::Odometry odom_msg, odom_msg_, odom_est_msg, leg_odom_msg, ground_truth_odom_msg, LFLeg_odom_msg, LHLeg_odom_msg, RFLeg_odom_msg, RHLeg_odom_msg,
	ground_truth_com_odom_msg, CoM_odom_msg, ground_truth_odom_msg_, ground_truth_odom_pub_msg;
	geometry_msgs::PoseStamped pose_msg, pose_msg_, temp_pose_msg, rel_supportPose_msg, rel_swingPose_msg;
	std_msgs::String support_leg_msg;
	geometry_msgs::WrenchStamped RFLeg_est_msg, LFLeg_est_msg, RHLeg_est_msg, LHLeg_est_msg, LFfsr_msg, LHfsr_msg, RFfsr_msg, RHfsr_msg, external_force_filt_msg;
   
    geometry_msgs::PoseStamped bodyPose_est_msg, supportPose_est_msg;
	geometry_msgs::TwistStamped bodyVel_est_msg, CoM_vel_msg;
	sensor_msgs::Imu  bodyAcc_est_msg;
	std_msgs::Int32 is_in_ds_msg, support_idx_msg;
	geometry_msgs::PointStamped COP_msg, copl_msg, copr_msg, CoM_pos_msg;

	//Madgwick gain
	double beta;
	// Helper
	bool is_connected_, ground_truth, support_idx_provided;



	

    boost::shared_ptr< dynamic_reconfigure::Server<serow::VarianceControlConfig> > dynamic_recfg_;

	double mass;
	IMUEKF* imuEKF;
	IMUinEKF* imuInEKF;
	bool useInIMUEKF;
	CoMEKF* nipmEKF;
	butterworthLPF** gyroLPF;
	MovingAverageFilter** gyroMAF;
	//Cuttoff Freqs for LPF
	double gyro_fx, gyro_fy, gyro_fz;
	Vector3d COP_fsr, GRF_fsr, CoM_enc, Gyrodot, Gyro_, CoM_leg_odom;
	double bias_fx,bias_fy,bias_fz;

	Mediator *LFmdf, *RFmdf, *RHmdf, *LHmdf;

	string support_leg;

	serow::ContactDetectionQuad* cd;
    Vector3d copwLF, copwLH, copwRF, copwRH;
	Vector3d copLF, copLH, copRF, copRH;

    double weightLF, weightLH, weightRF, weightRH;


	bool useGEM, ContactDetectionWithCOP, ContactDetectionWithKinematics;
	double foot_polygon_xmin, foot_polygon_xmax, foot_polygon_ymin, foot_polygon_ymax;
	double LFforce_sigma, LHforce_sigma, RFforce_sigma, RHforce_sigma, LFcop_sigma, LHcop_sigma, RFcop_sigma, RHcop_sigma, LFvnorm_sigma, LHvnorm_sigma, RFvnorm_sigma, RHvnorm_sigma, probabilisticContactThreshold;
	Vector3d LFLegGRF, LHLegGRF, RFLegGRF, RHLegGRF, LFLegGRT, LHLegGRT, RFLegGRT, RHLegGRT, offsetGT,offsetGTCoM;
	Affine3d Tws, Twb, Twb_; //From support s to world frame;
	Affine3d Tbs, Tsb, Tssw, Tbsw;
	Vector3d no_motion_residual;
	/****/
	bool  kinematicsInitialized, firstContact;
	double LFLegForceFilt, LHLegForceFilt, RFLegForceFilt, RHLegForceFilt;
	double LegHighThres, LegLowThres, LosingContact, StrikingContact;
	double bias_ax, bias_ay, bias_az, bias_gx, bias_gy, bias_gz;
	double g, m, I_xx, I_yy, I_zz;
	double joint_noise_density;

	bool comp_with, comp_odom0_inc, firstCO;
	std::string comp_with_odom0_topic;
	Vector3d offsetCO;
	Quaterniond qoffsetCO;
	nav_msgs::Odometry comp_odom0_msg;
	void subscribeToCompOdom();
	void compodom0Cb(const nav_msgs::Odometry::ConstPtr& msg);
	/** Real odometry Data **/
     string LFfsr_topic,LHfsr_topic, RFfsr_topic, RHfsr_topic;
	 string imu_topic;
	 string joint_state_topic;
	 string odom_topic;
	 string ground_truth_odom_topic, ground_truth_com_topic, support_idx_topic;
     string modelname;

	//Odometry, from supportleg to inertial, transformation from support leg to other leg
     void subscribeToIMU();
	 void subscribeToFSR();
	 void subscribeToJointState();
	 void subscribeToOdom();
	 void subscribeToGroundTruth();
	 void subscribeToGroundTruthCoM();
	 void ground_truth_comCb(const nav_msgs::Odometry::ConstPtr& msg);
	 void subscribeToSupportIdx();
	 void support_idxCb(const std_msgs::Int32::ConstPtr& msg);
	 void ground_truth_odomCb(const nav_msgs::Odometry::ConstPtr& msg);
	 void imuCb(const sensor_msgs::Imu::ConstPtr& msg);
	 void joint_stateCb(const sensor_msgs::JointState::ConstPtr& msg);
	 void odomCb(const nav_msgs::Odometry::ConstPtr& msg);
	 void LFfsrCb(const geometry_msgs::WrenchStamped::ConstPtr& msg);
	 void LHfsrCb(const geometry_msgs::WrenchStamped::ConstPtr& msg);
	 void RFfsrCb(const geometry_msgs::WrenchStamped::ConstPtr& msg);
	 void RHfsrCb(const geometry_msgs::WrenchStamped::ConstPtr& msg);


	void computeGlobalCOP(Affine3d TwLF_, Affine3d TwLH_, Affine3d TwRF_, Affine3d TwRH_);
	 void filterGyrodot();
	//private methods
	void init();

	void estimateWithCoMEKF();
	void estimateWithIMUEKF();
	void estimateWithInIMUEKF();




	void computeKinTFs();
	//publish functions
	void publishGRF();
	void publishJointEstimates();
	void publishCoMEstimates();
	void deAllocate();
	void publishLegEstimates();
	void publishSupportEstimates();

	void publishBodyEstimates();
	void publishContact();
	void publishCOP();
	// Advertise to ROS Topics
	void advertise();
	void subscribe();

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	// Constructor/Destructor
	quadruped_ekf();

	~quadruped_ekf();

	// Connect/Disconnet to ALProxies
	bool connect(const ros::NodeHandle nh);

	void disconnect();



	// Parameter Server
	void loadparams();
	void loadIMUEKFparams();
	void loadCoMEKFparams();
	void loadJointKFparams();
	// General Methods
	void reconfigureCB(serow::VarianceControlConfig& config, uint32_t level);
	void run();

	bool connected();


};

#endif // HUMANOID_EKF_H
