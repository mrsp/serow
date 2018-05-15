/*
 * humanoid_state_estimation - a complete state estimation scheme for humanoid robots
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


#include <iostream>
#include <algorithm>
#include "humanoid_state_estimation/humanoid_ekf.h"


void humanoid_ekf::loadparams() {

	ros::NodeHandle n_p("~");
	// Load Server Parameters
	n_p.param<std::string>("base_link",base_link_frame,"base_link");
	n_p.param<std::string>("lfoot",lfoot_frame,"l_sole");
	n_p.param<std::string>("rfoot",rfoot_frame,"r_sole");

	n_p.param<double>("imu_topic_freq",freq,100.0);
	n_p.param<double>("fsr_topic_freq",fsr_freq,100.0);
	n_p.param<double>("LLegUpThres", LLegUpThres,20.0);
	n_p.param<double>("LLegLowThres", LLegLowThres,15.0);
	n_p.param<double>("LosingContact", LosingContact,5.0);
	n_p.param<double>("StrikingContact", StrikingContact,10.0);
	n_p.param<bool>("useLegOdom",useLegOdom,false);
	n_p.param<bool>("usePoseUpdate",usePoseUpdate,false);
	n_p.param<bool>("visualize_with_rviz",visualize_with_rviz,false);
	n_p.param<bool>("ground_truth",ground_truth,false);
	n_p.param<bool>("debug_mode",debug_mode,false);
	n_p.param<bool>("support_idx_provided",support_idx_provided,false);
	if(support_idx_provided)
		n_p.param<std::string>("support_idx_topic", support_idx_topic,"support_idx");
	if(ground_truth){
		n_p.param<std::string>("ground_truth_odom_topic", ground_truth_odom_topic,"ground_truth");
		n_p.param<std::string>("ground_truth_com_topic", ground_truth_com_topic,"ground_truth_com");
		n_p.param<std::string>("is_in_ds_topic", is_in_ds_topic,"is_in_ds_topic");
	}
	n_p.param<bool>("comp_with",comp_with,false);
	if(comp_with){
		n_p.param<std::string>("comp_with_odom_topic", comp_with_odom_topic,"compare_with_odom");
	}
	if(!useLegOdom){
		std::vector<double> pose_list;
		n_p.getParam("T_B_P",pose_list);
		T_B_P(0,0) = pose_list[0];
		T_B_P(0,1) = pose_list[1];
		T_B_P(0,2) = pose_list[2];
		T_B_P(0,3) = pose_list[3];
		T_B_P(1,0) = pose_list[4];
		T_B_P(1,1) = pose_list[5];
		T_B_P(1,2) = pose_list[6];
		T_B_P(1,3) = pose_list[7];
		T_B_P(2,0) = pose_list[8];
		T_B_P(2,1) = pose_list[9];
		T_B_P(2,2) = pose_list[10];
		T_B_P(2,3) = pose_list[11];
		T_B_P(3,0) = pose_list[12];
		T_B_P(3,1) = pose_list[13];
		T_B_P(3,2) = pose_list[14];
		T_B_P(3,3) = pose_list[15];		
		q_B_P = Quaterniond(T_B_P.linear());
	}
	std::vector<double> imu_list;
	n_p.getParam("T_B_I",imu_list);
	T_B_I(0,0) = imu_list[0];
	T_B_I(0,1) = imu_list[1];
	T_B_I(0,2) = imu_list[2];
	T_B_I(0,3) = imu_list[3];
	T_B_I(1,0) = imu_list[4];
	T_B_I(1,1) = imu_list[5];
	T_B_I(1,2) = imu_list[6];
	T_B_I(1,3) = imu_list[7];
	T_B_I(2,0) = imu_list[8];
	T_B_I(2,1) = imu_list[9];
	T_B_I(2,2) = imu_list[10];
	T_B_I(2,3) = imu_list[11];
	T_B_I(3,0) = imu_list[12];
	T_B_I(3,1) = imu_list[13];
	T_B_I(3,2) = imu_list[14];
	T_B_I(3,3) = imu_list[15];		
	q_B_I = Quaterniond(T_B_I.linear());


	n_p.param<std::string>("pose_topic", pose_topic,"pose");
	n_p.param<std::string>("odom_topic", odom_topic,"odom");
	n_p.param<std::string>("imu_topic", imu_topic,"imu");
	n_p.param<std::string>("joint_state_topic", joint_state_topic,"joint_states");
	n_p.param<std::string>("lfoot_force_torque_topic",lfsr_topic,"LfsrFL");
	n_p.param<std::string>("rfoot_force_torque_topic",rfsr_topic,"LfsrFR");

	n_p.param<std::string>("copl_topic",copl_topic,"LfsrFL");
	n_p.param<std::string>("copr_topic",copr_topic,"LfsrFR");

	n_p.param<bool>("estimateCoM", useCoMEKF,false);
	n_p.param<bool>("estimateJoints", useJointKF,false);
	
}

void humanoid_ekf::loadJointKFparams()
{
	ros::NodeHandle n_p("~");
	n_p.param<double>("joint_topic_freq",joint_freq,100.0);	
}

void humanoid_ekf::loadIMUEKFparams()
{
	ros::NodeHandle n_p("~");
	n_p.param<double>("bias_ax", bias_ax,0.0);
	n_p.param<double>("bias_ay", bias_ay,0.0);
	n_p.param<double>("bias_az", bias_az,0.0);
	n_p.param<double>("bias_gx", bias_gx,0.0);
	n_p.param<double>("bias_gy", bias_gy,0.0);
	n_p.param<double>("bias_gz", bias_gz,0.0);

	n_p.param<double>("accelerometer_noise_density", imuEKF->acc_qx,0.001);
	n_p.param<double>("accelerometer_noise_density", imuEKF->acc_qy,0.001);
	n_p.param<double>("accelerometer_noise_density", imuEKF->acc_qz,0.001);

	n_p.param<double>("gyroscope_noise_density", imuEKF->gyr_qx,0.0001);
	n_p.param<double>("gyroscope_noise_density", imuEKF->gyr_qy,0.0001);
	n_p.param<double>("gyroscope_noise_density", imuEKF->gyr_qz,0.0001);

	n_p.param<double>("accelerometer_bias_random_walk", imuEKF->accb_qx,1.0e-04);
	n_p.param<double>("accelerometer_bias_random_walk", imuEKF->accb_qy,1.0e-04);
	n_p.param<double>("accelerometer_bias_random_walk", imuEKF->accb_qz,1.0e-04);
	n_p.param<double>("gyroscope_bias_random_walk", imuEKF->gyrb_qx,1.0e-05);
	n_p.param<double>("gyroscope_bias_random_walk", imuEKF->gyrb_qy,1.0e-05);
	n_p.param<double>("gyroscope_bias_random_walk", imuEKF->gyrb_qz,1.0e-05);


	n_p.param<double>("support_position_random_walk", imuEKF->support_qpx,5.0e-03);
	n_p.param<double>("support_position_random_walk", imuEKF->support_qpy,5.0e-03);
	n_p.param<double>("support_position_random_walk", imuEKF->support_qpz,5.0e-03);
	n_p.param<double>("support_orientation_random_walk", imuEKF->support_qax,5.0e-03);
	n_p.param<double>("support_orientation_random_walk", imuEKF->support_qay,5.0e-03);
	n_p.param<double>("support_orientation_random_walk", imuEKF->support_qaz,5.0e-03);


	n_p.param<double>("odom_position_noise_density", imuEKF->odom_px,1.0e-03);
	n_p.param<double>("odom_position_noise_density", imuEKF->odom_py,1.0e-03);
	n_p.param<double>("odom_position_noise_density", imuEKF->odom_pz,1.0e-03);
	n_p.param<double>("odom_orientation_noise_density", imuEKF->odom_ax,1.0e-02);
	n_p.param<double>("odom_orientation_noise_density", imuEKF->odom_ay,1.0e-02);
	n_p.param<double>("odom_orientation_noise_density", imuEKF->odom_az,1.0e-02);



	n_p.param<double>("support_position_noise_density", imuEKF->support_px,5.0e-05);
	n_p.param<double>("support_position_noise_density", imuEKF->support_py,5.0e-05);
	n_p.param<double>("support_position_noise_density", imuEKF->support_pz,5.0e-05);
	n_p.param<double>("support_orientation_noise_density", imuEKF->support_ax,5.0e-05);
	n_p.param<double>("support_orientation_noise_density", imuEKF->support_ay,5.0e-05);
	n_p.param<double>("support_orientation_noise_density", imuEKF->support_az,5.0e-05);
	n_p.param<double>("gravity", imuEKF->ghat,9.81);
	n_p.param<double>("gravity", g,9.81);
}



void humanoid_ekf::loadCoMEKFparams() {

	ros::NodeHandle n_p("~");
	n_p.param<double>("com_position_random_walk", nipmEKF->com_q, 1.0e-04);
	n_p.param<double>("com_velocity_random_walk", nipmEKF->comd_q, 1.0e-03);
	n_p.param<double>("external_force_random_walk", nipmEKF->fd_q, 1.0);
	n_p.param<double>("com_position_noise_density", nipmEKF->com_r, 1.0e-04);
	n_p.param<double>("com_acceleration_noise_density", nipmEKF->comdd_r, 5.0e-02);
	n_p.param<double>("mass", m, 5.14);
	n_p.param<double>("Ixx", I_xx,0.00000);
	n_p.param<double>("Iyy", I_yy,0.00000);
	n_p.param<double>("Izz", I_zz,0.00000);
	n_p.param<double>("bias_fx",bias_fx,0.0);
	n_p.param<double>("bias_fy",bias_fy,0.0);
	n_p.param<double>("bias_fz",bias_fz,0.0);
	n_p.param<bool>("useGyroLPF",useGyroLPF,false);
	n_p.param<double>("gyro_cut_off_freq",gyro_fx,7.0);
	n_p.param<double>("gyro_cut_off_freq",gyro_fy,7.0);
	n_p.param<double>("gyro_cut_off_freq",gyro_fz,7.0);
	n_p.param<int>("maWindow",maWindow,10);

}



humanoid_ekf::humanoid_ekf() 
{
	useJointKF = false;
	useCoMEKF = false;
	useLegOdom = false;
	firstUpdate = false;
	firstOdom = false;
	firstPose = false;
}

humanoid_ekf::~humanoid_ekf() {
	if (is_connected_)
		disconnect();
}

void humanoid_ekf::disconnect() {
	if (!is_connected_)
		return;
	
	is_connected_ = false;
}

bool humanoid_ekf::connect(const ros::NodeHandle nh) {
	// Initialize ROS nodes
	n = nh;
	// Load ROS Parameters
	loadparams();
	//Initialization
	init();

	dynamic_recfg_ = boost::make_shared< dynamic_reconfigure::Server<humanoid_state_estimation::VarianceControlConfig> >(n);
    dynamic_reconfigure::Server<humanoid_state_estimation::VarianceControlConfig>::CallbackType cb = boost::bind(&humanoid_ekf::reconfigureCB, this, _1, _2);
    dynamic_recfg_->setCallback(cb);

	// Load IMU parameters
	loadIMUEKFparams();
	imuEKF->setAccBias(Vector3d(bias_ax,bias_ay,bias_az));
	imuEKF->setGyroBias(Vector3d(bias_gx,bias_gy,bias_gz));

	if(useCoMEKF)
		loadCoMEKFparams();
	if(useJointKF)
		loadJointKFparams();


	//Subscribe/Publish ROS Topics/Services
	subscribe();
	advertise();

	is_connected_ = true;

	ROS_INFO_STREAM("Humanoid State Estimator Initialized");

	return true;
}



void humanoid_ekf::reconfigureCB(humanoid_state_estimation::VarianceControlConfig& config, uint32_t level)
{

      imuEKF->accb_qx = config.accb_qx;
      imuEKF->accb_qy = config.accb_qy;
      imuEKF->accb_qz = config.accb_qz;

      imuEKF->gyrb_qx = config.gyrb_qx;
      imuEKF->gyrb_qy = config.gyrb_qy;
      imuEKF->gyrb_qz = config.gyrb_qz;

      imuEKF->acc_qx = config.acc_qx;
      imuEKF->acc_qy = config.acc_qy;
      imuEKF->acc_qz = config.acc_qz;

      imuEKF->gyr_qx = config.gyr_qx;
      imuEKF->gyr_qy = config.gyr_qy;
      imuEKF->gyr_qz = config.gyr_qz;

      imuEKF->odom_px = config.odom_px; 
      imuEKF->odom_py = config.odom_py; 
      imuEKF->odom_pz = config.odom_pz; 

      imuEKF->odom_ax = config.odom_ax; 
      imuEKF->odom_ay = config.odom_ay; 
      imuEKF->odom_az = config.odom_az; 


      imuEKF->support_qpx = config.support_qpx; 
      imuEKF->support_qpy = config.support_qpy; 
      imuEKF->support_qpz = config.support_qpz; 

      imuEKF->support_qax = config.support_qax; 
      imuEKF->support_qay = config.support_qay; 
      imuEKF->support_qaz = config.support_qaz; 

      imuEKF->support_px = config.support_px; 
      imuEKF->support_py = config.support_py; 
      imuEKF->support_pz = config.support_pz; 
	  
      imuEKF->support_ax = config.support_ax; 
      imuEKF->support_ay = config.support_ay; 
      imuEKF->support_az = config.support_az; 

     if(useCoMEKF){
		nipmEKF->com_q = config.com_q;
		nipmEKF->comd_q = config.comd_q;
		nipmEKF->com_r = config.com_r;
		nipmEKF->comdd_r = config.comdd_r;
		nipmEKF->fd_q = config.fd_q;
	 }
}


bool humanoid_ekf::connected() {
	return is_connected_;
}

void humanoid_ekf::subscribe()
{

	subscribeToIMU();
	subscribeToFSR();

	firstJoint = true;
	subscribeToJointState();
	
	firstUpdate = true;
	if(!useLegOdom){
		if (usePoseUpdate){
			firstPose = true;
			subscribeToPose();
		}
		else{
			firstOdom = true;
			subscribeToOdom();
		}
	}

	if(ground_truth){
		subscribeToGroundTruth();
		subscribeToGroundTruthCoM();
		subscribeToDS();
	}
	if(comp_with)
		subscribeToCompOdom();

	if(support_idx_provided)
		subscribeToSupportIdx();

	ros::Duration(1.0).sleep();
}

void humanoid_ekf::init() {


	/** Initialize Variables **/
	//Kinematic TFs
	Tws = Affine3d::Identity();
	Twh = Affine3d::Identity();
	Twb = Affine3d::Identity();
	Twb_ = Twb;
	Tbs = Affine3d::Identity();
	Tbsw = Affine3d::Identity();
	Tbs_ = Tbs;
	Tbsw = Tbsw;
	Tsb = Affine3d::Identity();
	Tssw = Affine3d::Identity();
	LLegGRF = Vector3d::Zero();
	RLegGRF = Vector3d::Zero();
	LLegGRT = Vector3d::Zero();
	RLegGRT = Vector3d::Zero();
	copl = Vector3d::Zero();
	copr = Vector3d::Zero();
	no_motion_residual = Vector3d::Zero();
	firstrun = true;
	firstGyrodot = true;
	legSwitch = false;
	firstContact = true;

	// Initialize the IMU based EKF 
	imuEKF = new IMUEKF;
	imuEKF->init();
	if(useCoMEKF){
		if(useGyroLPF)
			gyroLPF = new LPF;
		else{
			gyroMAF = new MovingAverageFilter*[3];
			for(unsigned int i=0;i<3;i++)
				gyroMAF[i] = new MovingAverageFilter();			
		}
		nipmEKF = new CoMEKF;
		nipmEKF->init();
		kin = new Kinematics(base_link_frame,rfoot_frame,lfoot_frame);
	}
	imu_inc = false;
	fsr_inc = false;
	pose_inc = false;
	joint_inc = false;
	odom_inc = false;
	leg_odom_inc = false;
	support_inc = false;
	swing_inc = false;
    no_motion_indicator = false;
	no_motion_it = 0;
	no_motion_threshold = 5e-4;
	no_motion_it_threshold = 500;
	Tbs_stamp = ros::Time::now();
	Tbsw_stamp = ros::Time::now();
	lcount = 0;
	rcount = 0;
}



void humanoid_ekf::run() {
	
	static ros::Rate rate(1.05*freq);  //ROS Node Loop Rate
	while (ros::ok()){
		predictWithImu = false;
		predictWithCoM = false;

		if(fsr_inc){
			computeLGRF();
			computeRGRF();
			//Determine which foot is in contact with the Ground and is the current support foot
			determineLegContact();	
		}

		//Compute the required transformation matrices (tfs) with Kinematics
		if(joint_inc){
			computeKinTFs();
		}
		if(!firstrun){
			estimateWithIMUEKF();
			if(useCoMEKF)
				estimateWithCoMEKF();

			//Publish Data
			publishBodyEstimates();
			publishSupportEstimates();
			publishContact();
			publishGRF();
			
			if(useCoMEKF){
				publishCoMEstimates();
				publishCOP();
			}
			if(useJointKF)
				publishJointEstimates();
		}
		ros::spinOnce();
		rate.sleep();
	}
	//De-allocation of Heap
	deAllocate();
}

void humanoid_ekf::estimateWithIMUEKF()
{
		//Initialize the IMU EKF state
		if (imuEKF->firstrun) {
			imuEKF->setdt(1.0/freq);
			imuEKF->setBodyPos(Twb.translation());
			imuEKF->setBodyOrientation(Twb.linear());
			imuEKF->setSupportPos(Tws.translation());
			imuEKF->setSupportOrientation(Tws.linear());
			imuEKF->firstrun = false;
		}
		//Compute the attitude and posture with the IMU-Kinematics Fusion
		//Predict with the IMU gyro and acceleration
		if(imu_inc && !predictWithImu && !imuEKF->firstrun){
			imuEKF->predict( T_B_I.linear() * Vector3d(imu_msg.angular_velocity.x,imu_msg.angular_velocity.y,imu_msg.angular_velocity.z),
			T_B_I.linear()*Vector3d(imu_msg.linear_acceleration.x,imu_msg.linear_acceleration.y,imu_msg.linear_acceleration.z));
			imu_inc = false;
			predictWithImu = true;
		}

		//Check for no motion
		if(predictWithImu)
		{
			if(check_no_motion){
				no_motion_residual = Twb.translation()-Twb_.translation();
				if(no_motion_residual.norm() < no_motion_threshold)
					no_motion_it++;
				else{
					no_motion_indicator = false;
					no_motion_it = 0;
				}
				if(no_motion_it > no_motion_it_threshold)
				{
					no_motion_indicator = true;
					no_motion_it = 0;
				}
				check_no_motion = false;
			}

			//Update EKF
			if(firstUpdate){
				pos_update = Twb.translation();
				q_update = qwb;
				//First Update
				firstUpdate = false;
				imuEKF->updateWithOdom(pos_update, q_update);	
			}
			else{
			//Update with the odometry
				if(no_motion_indicator || useLegOdom){
						//cout<<"No motion"<<endl;
						//cout<<no_motion_indicator<<endl;
						if(leg_odom_inc){
								pos_update += Twb.translation()-Twb_.translation();
								Quaterniond q_now = qwb;
								Quaterniond q_prev = qwb_;
								q_update *=  q_now  * q_prev.inverse();
								leg_odom_inc = false;
								imuEKF->updateWithOdom(pos_update, q_update);	
						}
				}
				else
				{
					if(!usePoseUpdate)
					{
						if(odom_inc)
						{
								pos_update += T_B_P.linear() * Vector3d(odom_msg.pose.pose.position.x - odom_msg_.pose.pose.position.x,
								odom_msg.pose.pose.position.y - odom_msg_.pose.pose.position.y ,odom_msg.pose.pose.position.z - odom_msg_.pose.pose.position.z);

								Quaterniond q_now =  q_B_P * Quaterniond(odom_msg.pose.pose.orientation.w,odom_msg.pose.pose.orientation.x,
								odom_msg.pose.pose.orientation.y,odom_msg.pose.pose.orientation.z);

								Quaterniond q_prev = q_B_P * Quaterniond(odom_msg_.pose.pose.orientation.w,odom_msg_.pose.pose.orientation.x,
								odom_msg_.pose.pose.orientation.y,odom_msg_.pose.pose.orientation.z) ;

								q_update *=   ( q_now * q_prev.inverse());
								odom_inc = false;
								odom_msg_ = odom_msg;
								imuEKF->updateWithOdom(pos_update, q_update);	
					
						}
					}
					else
					{
						if(pose_inc)
						{
								pos_update += T_B_P.linear() * Vector3d(pose_msg.pose.position.x-pose_msg_.pose.position.x,
								pose_msg.pose.position.y-pose_msg_.pose.position.y,pose_msg.pose.position.z-pose_msg_.pose.position.z);
								Quaterniond q_now = q_B_P * Quaterniond(pose_msg.pose.orientation.w,pose_msg.pose.orientation.x,
								pose_msg.pose.orientation.y,pose_msg.pose.orientation.z);
								Quaterniond q_prev = q_B_P *  Quaterniond(pose_msg_.pose.orientation.w,pose_msg_.pose.orientation.x,
								pose_msg_.pose.orientation.y,pose_msg_.pose.orientation.z);
								q_update *=   ( q_now * q_prev.inverse());
								pose_inc = false;
								pose_msg_ = pose_msg;
								imuEKF->updateWithOdom(pos_update, q_update);	

						}
					}
				}


			}
		}

	
		//Update with the Support foot position and orientation
		if(support_inc && predictWithImu){
			imuEKF->updateWithSupport(Tbs.translation(),qbs);
			support_inc=false;
		}
	// 	cout<<"ax"<<imuEKF->bias_ax<<endl;
	// 	cout<<"ay"<<imuEKF->bias_ay<<endl;
	// 	cout<<"az"<<imuEKF->bias_az<<endl;
	//  cout<<"gx"<<imuEKF->bias_gx<<endl;
	// 	cout<<"gy"<<imuEKF->bias_gy<<endl;
	// 	cout<<"gz"<<imuEKF->bias_gz<<endl;
}


void humanoid_ekf::estimateWithCoMEKF()
{
	if(joint_inc){
			kin->computeCOM(joint_map, com, mass,tf_right_foot,  tf_left_foot);
			CoM_enc << com.x(), com.y(), com.z();



			if (nipmEKF->firstrun){
				nipmEKF->setdt(1.0/fsr_freq);
				nipmEKF->setParams(mass,I_xx,I_yy,g);
				nipmEKF->setCoMPos(Twb*CoM_enc);
				nipmEKF->setCoMExternalForce(Vector3d(bias_fx,bias_fy,bias_fz));
				nipmEKF->firstrun = false;
				if(useGyroLPF){
					gyroLPF->setdt(1.0/fsr_freq);
					gyroLPF->setCutOffFreq(gyro_fx,gyro_fy,gyro_fz);
				}
				else{
					for(unsigned int i=0;i<3;i++)
						gyroMAF[i]->setParams(maWindow);
				}
				
			}
	}
	
	//Compute the COP in the Inertial Frame
		if(fsr_inc && predictWithImu && !predictWithCoM && !nipmEKF->firstrun){
			computeCOP(imuEKF->Tis,Tssw);
			//Numerically compute the Gyro acceleration in the Inertial Frame and use a 3-Point Low-Pass filter
			filterGyrodot();
			DiagonalMatrix<double,3> Inertia(I_xx,I_yy,I_zz);
			nipmEKF->predict(COP_fsr,GRF_fsr,imuEKF->Rib*Inertia*Gyrodot);
			fsr_inc = false;
			predictWithCoM = true;
		}

		if(joint_inc && predictWithCoM){
			// nipmEKF->updateWithEnc(imuEKF->Tib*CoM_enc);
			// nipmEKF->updateWithImu(
			// 		Vector3d(imuEKF->accX, imuEKF->accY, imuEKF->accZ - g),
			// 		imuEKF->Tib*CoM_enc,
			// 		Vector3d(imuEKF->gyroX, imuEKF->gyroY, 0.0));
			nipmEKF->update(
					Vector3d(imuEKF->accX, imuEKF->accY, imuEKF->accZ - imuEKF->ghat),
					imuEKF->Tib*CoM_enc,
					Vector3d(imuEKF->gyroX, imuEKF->gyroY, imuEKF->gyroZ),Gyrodot);
			joint_inc = false;
		}

}

void humanoid_ekf::computeLGRF()
{
	LLegGRF(0) = lfsr_msg.wrench.force.x;
	LLegGRF(1) = lfsr_msg.wrench.force.y;
	LLegGRF(2) = lfsr_msg.wrench.force.z;
	LLegGRT(0) = lfsr_msg.wrench.torque.x;
	LLegGRT(1) = lfsr_msg.wrench.torque.y;
	LLegGRT(2) = lfsr_msg.wrench.torque.z;

}
void humanoid_ekf::computeRGRF()
{
	RLegGRF(0) = rfsr_msg.wrench.force.x;
	RLegGRF(1) = rfsr_msg.wrench.force.y;
	RLegGRF(2) = rfsr_msg.wrench.force.z;
	RLegGRT(0) = rfsr_msg.wrench.torque.x;
	RLegGRT(1) = rfsr_msg.wrench.torque.y;
	RLegGRT(2) = rfsr_msg.wrench.torque.z;
}

void humanoid_ekf::computeKinTFs() {

	try{
		Tbs_listener.lookupTransform(base_link_frame, support_foot_frame,  
						ros::Time(0), Tbs_tf);
		
		if(Tbs_tf.stamp_ !=	Tbs_stamp)
		{
			Tbs_.translation() << Tbs_tf.getOrigin().x(), Tbs_tf.getOrigin().y(), Tbs_tf.getOrigin().z();
			qbs = Quaterniond(Tbs_tf.getRotation().w(), Tbs_tf.getRotation().x(), Tbs_tf.getRotation().y(), Tbs_tf.getRotation().z());
			Tbs_.linear() = qbs.toRotationMatrix();
			Tbs = isNear(Tbs, Tbs_, 5e-3, 5e-3, 5e-3);
			Tbs_stamp = Tbs_tf.stamp_;
			support_inc = true;
		}
	}
	catch (tf::TransformException ex){
		ROS_ERROR("%s",ex.what());
	}
	
	//TF Initialization
	if (firstrun && support_inc){
			Tws.translation() << 0.00, Tbs.translation()(1), 0.00;
			Tws.linear() = Tbs.linear();
	}

	try{
		Tbsw_listener.lookupTransform(base_link_frame, swing_foot_frame,  
				ros::Time(0), Tbsw_tf);
		if(Tbsw_tf.stamp_ != Tbsw_stamp){
			Tbsw_.translation() << Tbsw_tf.getOrigin().x(), Tbsw_tf.getOrigin().y(), Tbsw_tf.getOrigin().z();
			qbsw = Quaterniond(Tbsw_tf.getRotation().w(), Tbsw_tf.getRotation().x(), Tbsw_tf.getRotation().y(), Tbsw_tf.getRotation().z());
			Tbsw_.linear() = qbsw.toRotationMatrix();
			Tbsw = isNear(Tbsw, Tbsw_, 5e-3, 5e-3, 5e-3);
			Tbsw_stamp = Tbsw_tf.stamp_;
			swing_inc = true;
		}
	}
	catch (tf::TransformException ex){
		ROS_ERROR("%s",ex.what());
	}

	if (support_inc && swing_inc)
	{
		Tsb = Tbs.inverse();
		Tssw = isNear(Tssw,Tsb * Tbsw, 1e-2,1e-2,5e-2); //needed by COP
		//Tssw = Tsb*Tbsw;
		qssw = Quaterniond(Tssw.linear());		
		
		if(legSwitch){
			//If Support foot changed update the support foot - world TF
			//Tws = Tws * Tssw.inverse();
			Tws = isNear(Tws, Tws * Tssw.inverse(), 1e-2, 1e-2, 5e-2); //needed by COP
			legSwitch=false;
		}
		Twb_ = Twb;
		Twb = isNear(Twb, Tws * Tsb, 5e-3, 5e-3, 5e-3);

		//Race Condition safe first run
		if(firstrun){
			Twb_ = Twb;
			firstrun=false;
		}
		qwb = Quaterniond(Twb.linear());
		qwb_ = Quaterniond(Twb_.linear());

		leg_odom_inc = true;
		check_no_motion = false;
		swing_inc = false;
	}

	


}




/* Schmidtt Trigger */
void humanoid_ekf::determineLegContact() {
	
	//Choose Initial Support Foot based on Contact Force
	if(firstContact){
		if(LLegGRF(2)>RLegGRF(2)){
				// Initial support leg 
					support_leg = "LLeg";
					swing_leg = "RLeg";
					support_foot_frame = lfoot_frame;
					swing_foot_frame = rfoot_frame;
		}
		else
		{
					support_leg = "RLeg";
					swing_leg = "LLeg";
					support_foot_frame = rfoot_frame;
					swing_foot_frame = lfoot_frame;
		}
		firstContact = false;
	}
	else{
		if(!support_idx_provided){
		//Determine if the Support Foot changed  
			if (support_leg == "RLeg")
			{
				if (LLegGRF(2) > LLegUpThres  && LLegGRF(2) < StrikingContact)
				{
					lcount++;
					if(lcount>3)
					{
						support_leg = "LLeg";
						support_foot_frame = lfoot_frame;
						swing_leg = "RLeg";
						swing_foot_frame = rfoot_frame;			
						legSwitch = true;
						lcount=0;
					}
				}
				else
					lcount = 0;
			}
			else{
				if (LLegGRF(2) < LLegLowThres && RLegGRF(2) < StrikingContact)
				{
					rcount++;
					if(rcount>3)
					{
						support_leg = "RLeg";
						support_foot_frame = rfoot_frame;
						swing_leg = "LLeg";
						swing_foot_frame = lfoot_frame;			
						legSwitch = true;
						rcount=0;
					}
				}
				else
					rcount = 0;
			}
		}
	}
}
  







void humanoid_ekf::deAllocate()
{
	if(useJointKF){
		for (unsigned int i = 0; i < number_of_joints; i++)
			delete[] JointKF[i];
		delete[] JointKF;
	}
	if(useCoMEKF){
		delete nipmEKF;
		if(useGyroLPF)
			delete gyroLPF;
		else
		for (unsigned int i = 0; i <3; i++)
			delete[] gyroMAF[i];
		delete[] gyroMAF;
	}
	delete imuEKF;
}





void humanoid_ekf::filterGyrodot() {
	if (!firstGyrodot) {
		//Compute numerical derivative
		Gyrodot = (Vector3d(imuEKF->gyroX, imuEKF->gyroY, imuEKF->gyroZ) - Gyro_)*fsr_freq;
		if(useGyroLPF){
			gyroLPF->filter(Gyrodot);
			Gyrodot(0) = gyroLPF->x;
			Gyrodot(1) = gyroLPF->y;
			Gyrodot(2) = gyroLPF->z;
		}
		else{
			gyroMAF[0]->filter(Gyrodot(0));
			gyroMAF[1]->filter(Gyrodot(1));
			gyroMAF[2]->filter(Gyrodot(2));

			Gyrodot(0)=gyroMAF[0]->x;
			Gyrodot(1)=gyroMAF[1]->x;
			Gyrodot(2)=gyroMAF[2]->x;
		}
	} 
	else {
		Gyrodot = Vector3d::Zero();
		firstGyrodot = false;
	}
	Gyro_ = Vector3d(imuEKF->gyroX, imuEKF->gyroY, imuEKF->gyroZ);
}


void humanoid_ekf::publishBodyEstimates() {
	bodyPose_est_msg.header.stamp = ros::Time::now();
	bodyPose_est_msg.header.frame_id = "odom";
	bodyPose_est_msg.pose.position.x = imuEKF->rX;
	bodyPose_est_msg.pose.position.y = imuEKF->rY;
	bodyPose_est_msg.pose.position.z = imuEKF->rZ;
	bodyPose_est_msg.pose.orientation.x = imuEKF->qib_.x();
	bodyPose_est_msg.pose.orientation.y = imuEKF->qib_.y();
	bodyPose_est_msg.pose.orientation.z = imuEKF->qib_.z();
	bodyPose_est_msg.pose.orientation.w = imuEKF->qib_.w();
	bodyPose_est_pub.publish(bodyPose_est_msg);

	bodyVel_est_msg.header.stamp = ros::Time::now();
	bodyVel_est_msg.header.frame_id = "odom";
	bodyVel_est_msg.twist.linear.x = imuEKF->velX;
	bodyVel_est_msg.twist.linear.y = imuEKF->velY;
	bodyVel_est_msg.twist.linear.z = imuEKF->velZ;
	bodyVel_est_msg.twist.angular.x = imuEKF->gyroX;
	bodyVel_est_msg.twist.angular.y = imuEKF->gyroY;
	bodyVel_est_msg.twist.angular.z = imuEKF->gyroZ;
	bodyVel_est_pub.publish(bodyVel_est_msg);

	bodyAcc_est_msg.header.stamp = ros::Time::now();
	bodyAcc_est_msg.header.frame_id = "odom";
	bodyAcc_est_msg.linear_acceleration.x = imuEKF->accX;
	bodyAcc_est_msg.linear_acceleration.y = imuEKF->accY;
	bodyAcc_est_msg.linear_acceleration.z = imuEKF->accZ;
	bodyAcc_est_msg.angular_velocity.x = imuEKF->gyroX;
	bodyAcc_est_msg.angular_velocity.y = imuEKF->gyroY;
	bodyAcc_est_msg.angular_velocity.z = imuEKF->gyroZ;
	bodyAcc_est_pub.publish(bodyAcc_est_msg);


	odom_est_msg.header.stamp=ros::Time::now();
	odom_est_msg.header.frame_id = "odom";
	odom_est_msg.pose.pose = bodyPose_est_msg.pose;
	odom_est_msg.twist.twist = bodyVel_est_msg.twist;
	//for(int i=0;i<36;i++)
    //odom_est_msg.pose.covariance[i] = 0;
	odom_est_pub.publish(odom_est_msg);


	if(debug_mode){
		leg_odom_msg.header.stamp=ros::Time::now();
		leg_odom_msg.header.frame_id = "odom";
		leg_odom_msg.pose.pose.position.x = Twb.translation()(0);
		leg_odom_msg.pose.pose.position.y = Twb.translation()(1);
		leg_odom_msg.pose.pose.position.z = Twb.translation()(2);
		leg_odom_msg.pose.pose.orientation.x = qwb.x();
		leg_odom_msg.pose.pose.orientation.y = qwb.y();
		leg_odom_msg.pose.pose.orientation.z = qwb.z();
		leg_odom_msg.pose.pose.orientation.w = qwb.w();
		leg_odom_pub.publish(leg_odom_msg);
	}

	if(ground_truth)
	{
			ground_truth_com_odom_msg.header.stamp = ros::Time::now();
			ground_truth_com_odom_msg.header.frame_id = "odom";
			ground_truth_com_pub.publish(ground_truth_com_odom_msg);

			ground_truth_odom_msg.header.stamp = ros::Time::now();
			ground_truth_odom_msg.header.frame_id = "odom";
			ground_truth_odom_pub.publish(ground_truth_odom_msg);

			ds_pub.publish(is_in_ds_msg);
	}
	if(comp_with){
			comp_odom_msg.header = odom_est_msg.header;
			comp_odom_pub.publish(comp_odom_msg);
	}
	if(visualize_with_rviz)
	{
		odom_path_msg.header = odom_est_msg.header;
		odom_path_msg.poses.push_back(bodyPose_est_msg);
		odom_path_pub.publish(odom_path_msg);

		temp_pose_msg.pose.position.x = Twb.translation()(0);
		temp_pose_msg.pose.position.y = Twb.translation()(1);
		temp_pose_msg.pose.position.z = Twb.translation()(2);
		temp_pose_msg.pose.orientation.x = qwb.x();
		temp_pose_msg.pose.orientation.y = qwb.y();
		temp_pose_msg.pose.orientation.z = qwb.z();
		temp_pose_msg.pose.orientation.w = qwb.w();
		leg_odom_path_msg.header = leg_odom_msg.header;
		leg_odom_path_msg.poses.push_back(temp_pose_msg);
		leg_odom_path_pub.publish(leg_odom_path_msg);



		if(ground_truth)
		{
			ground_truth_odom_path_msg.header.stamp = ros::Time::now();
			ground_truth_odom_path_msg.header.frame_id = "odom";
		
			temp_pose_msg.pose = ground_truth_odom_msg.pose.pose;
			ground_truth_odom_path_msg.poses.push_back(temp_pose_msg);
			ground_truth_odom_path_pub.publish(ground_truth_odom_path_msg);

			ground_truth_com_path_msg.header.stamp = ros::Time::now();
			ground_truth_com_path_msg.header.frame_id = "odom";
			temp_pose_msg.pose = ground_truth_com_odom_msg.pose.pose;
			ground_truth_com_path_msg.poses.push_back(temp_pose_msg);
			ground_truth_com_path_pub.publish(ground_truth_com_path_msg);


		}
		if(comp_with)
		{
			comp_odom_msg.header = odom_est_msg.header;
			comp_odom_pub.publish(comp_odom_msg);
			comp_odom_path_msg.header = odom_est_msg.header;
			temp_pose_msg.pose = comp_odom_msg.pose.pose;
			comp_odom_path_msg.poses.push_back(temp_pose_msg);
			comp_odom_path_pub.publish(comp_odom_path_msg);
		}
	}


}


void humanoid_ekf::publishSupportEstimates() {
	supportPose_est_msg.header.stamp = ros::Time::now();
	supportPose_est_msg.header.frame_id = "odom";
	supportPose_est_msg.pose.position.x = imuEKF->Tis.translation()(0);
	supportPose_est_msg.pose.position.y = imuEKF->Tis.translation()(1);
	supportPose_est_msg.pose.position.z = imuEKF->Tis.translation()(2);
	supportPose_est_msg.pose.orientation.x = imuEKF->qis_.x();
	supportPose_est_msg.pose.orientation.y = imuEKF->qis_.y();
	supportPose_est_msg.pose.orientation.z = imuEKF->qis_.z();
	supportPose_est_msg.pose.orientation.w = imuEKF->qis_.w();
	supportPose_est_pub.publish(supportPose_est_msg);


	if(visualize_with_rviz)
	{
		support_path_msg.header = supportPose_est_msg.header;
		support_path_msg.poses.push_back(supportPose_est_msg);
		support_path_pub.publish(support_path_msg);
	}

	if(debug_mode){
		rel_supportPose_msg.header.stamp = ros::Time::now();
		rel_supportPose_msg.header.frame_id = base_link_frame;
		rel_supportPose_msg.pose.position.x = Tbs.translation()(0);
		rel_supportPose_msg.pose.position.y = Tbs.translation()(1);
		rel_supportPose_msg.pose.position.z = Tbs.translation()(2);
		rel_supportPose_msg.pose.orientation.x = qbs.x();
		rel_supportPose_msg.pose.orientation.y = qbs.y();
		rel_supportPose_msg.pose.orientation.z = qbs.z();
		rel_supportPose_msg.pose.orientation.w = qbs.w();
		rel_supportPose_pub.publish(rel_supportPose_msg);

		rel_swingPose_msg.header.stamp = ros::Time::now();
		rel_swingPose_msg.header.frame_id = support_leg;
		rel_swingPose_msg.pose.position.x = Tssw.translation()(0);
		rel_swingPose_msg.pose.position.y = Tssw.translation()(1);
		rel_swingPose_msg.pose.position.z = Tssw.translation()(2);
		rel_swingPose_msg.pose.orientation.x = qssw.x();
		rel_swingPose_msg.pose.orientation.y = qssw.y();
		rel_swingPose_msg.pose.orientation.z = qssw.z();
		rel_swingPose_msg.pose.orientation.w = qssw.w();
		rel_swingPose_pub.publish(rel_swingPose_msg);
	}
}

void humanoid_ekf::publishContact() {
	support_leg_msg.data = support_leg;
	support_leg_pub.publish(support_leg_msg);
}

void humanoid_ekf::publishGRF() {

	if(debug_mode){
		LLeg_est_msg.wrench.force.x = LLegGRF(0);
		LLeg_est_msg.wrench.force.y = LLegGRF(1);
		LLeg_est_msg.wrench.force.z = LLegGRF(2);
		LLeg_est_msg.wrench.torque.x = LLegGRT(0);
		LLeg_est_msg.wrench.torque.y = LLegGRT(1);
		LLeg_est_msg.wrench.torque.z = LLegGRT(2);

		LLeg_est_msg.header.frame_id = lfoot_frame;
		LLeg_est_msg.header.stamp = ros::Time::now();
		LLeg_est_pub.publish(LLeg_est_msg);

		RLeg_est_msg.wrench.force.x = RLegGRF(0);
		RLeg_est_msg.wrench.force.y = RLegGRF(1);
		RLeg_est_msg.wrench.force.z = RLegGRF(2);
		RLeg_est_msg.wrench.torque.x = RLegGRT(0);
		RLeg_est_msg.wrench.torque.y = RLegGRT(1);
		RLeg_est_msg.wrench.torque.z = RLegGRT(2);
		RLeg_est_msg.header.frame_id = rfoot_frame;
		RLeg_est_msg.header.stamp = ros::Time::now();
		RLeg_est_pub.publish(RLeg_est_msg);
	}






}


void humanoid_ekf::computeCOP(Affine3d Tis_, Affine3d Tssprime_) {


	Vector3d cops, copsprime, fs, fsprime;
	double weights, weightsprime;

	fs = Vector3d::Zero();
	fsprime = Vector3d::Zero();
	copsprime = Vector3d::Zero();
	cops = Vector3d::Zero();

	// Computation of the CoP in the Local Coordinate Frame of the Foot
	copl = Vector3d(copl_msg.point.x, copl_msg.point.y,0);
	copr = Vector3d(copr_msg.point.x, copr_msg.point.y,0);

	if (support_leg == "RLeg")
	{
		cops = copr;
		copsprime = copl;
		weights = RLegGRF(2)/g;
		weightsprime = LLegGRF(2)/g;
		fs = RLegGRF;
		fsprime = LLegGRF;
	} else {
		cops = copl;
		copsprime = copr;
		weights = LLegGRF(2)/g;
		weightsprime = RLegGRF(2)/g;
		fs = LLegGRF;
		fsprime = RLegGRF;
	}

	if (fsprime(2) < LosingContact )
	{
		weightsprime = 0.00;
		fsprime = Vector3d::Zero();
		copsprime = Vector3d::Zero();
	}
	// Compute the CoP wrt the Support Foot Frame 
	COP_fsr = ( cops * weights + Tssprime_ * copsprime * (weightsprime) ) / (weights + weightsprime);

	COP_fsr = Tis_ * COP_fsr;


	GRF_fsr = fs  + Tssprime_.linear() * fsprime;
	GRF_fsr = Tis_.linear()*GRF_fsr;
}



void humanoid_ekf::publishCOP() {
	COP_msg.point.x = COP_fsr(0);
	COP_msg.point.y = COP_fsr(1);
	COP_msg.point.z = COP_fsr(2);
	COP_msg.header.stamp = ros::Time::now();
	COP_msg.header.frame_id = "odom";
	COP_pub.publish(COP_msg);
	if(visualize_with_rviz)
	{
		cop_path_msg.header = COP_msg.header;
		temp_pose_msg.pose.position.x = COP_msg.point.x;
		temp_pose_msg.pose.position.y = COP_msg.point.y;
		temp_pose_msg.pose.position.z = COP_msg.point.z;
		cop_path_msg.poses.push_back(temp_pose_msg);
		cop_path_pub.publish(cop_path_msg);
	}

}


void humanoid_ekf::publishCoMEstimates() {
	CoM_pos_msg.point.x = nipmEKF->comX;
	CoM_pos_msg.point.y = nipmEKF->comY;
	CoM_pos_msg.point.z = nipmEKF->comZ;
	CoM_pos_msg.header.stamp = ros::Time::now();
	CoM_pos_msg.header.frame_id = "odom";
	CoM_pos_pub.publish(CoM_pos_msg);
	CoM_vel_msg.header.frame_id = "odom";
	CoM_vel_msg.header.stamp = ros::Time::now();
	CoM_vel_msg.twist.linear.x = nipmEKF->velX;
	CoM_vel_msg.twist.linear.y = nipmEKF->velY;
	CoM_vel_msg.twist.linear.z = nipmEKF->velZ;
	CoM_vel_pub.publish(CoM_vel_msg);

	CoM_odom_msg.header.stamp=ros::Time::now();
	CoM_odom_msg.header.frame_id = "odom";
	CoM_odom_msg.pose.pose.position.x = CoM_pos_msg.point.x;
	CoM_odom_msg.pose.pose.position.y = CoM_pos_msg.point.y;
	CoM_odom_msg.pose.pose.position.z = CoM_pos_msg.point.z;
	CoM_odom_msg.twist.twist = CoM_vel_msg.twist;
	//for(int i=0;i<36;i++)
    //odom_est_msg.pose.covariance[i] = 0;
	CoM_odom_pub.publish(CoM_odom_msg);







	external_force_filt_msg.header.frame_id = "odom";
	external_force_filt_msg.header.stamp = ros::Time::now();
	external_force_filt_msg.wrench.force.x = nipmEKF->fX;
	external_force_filt_msg.wrench.force.y = nipmEKF->fY;
	external_force_filt_msg.wrench.force.z = nipmEKF->fZ;
	external_force_filt_pub.publish(external_force_filt_msg);
	if(visualize_with_rviz)
	{
		com_path_msg.header = CoM_pos_msg.header;
		temp_pose_msg.pose.position.x = CoM_pos_msg.point.x;
		temp_pose_msg.pose.position.y = CoM_pos_msg.point.y;
		temp_pose_msg.pose.position.z = CoM_pos_msg.point.z;
		com_path_msg.poses.push_back(temp_pose_msg);
		com_path_pub.publish(com_path_msg);
	}

	if(debug_mode)
	{
			temp_pose_msg.pose.position.x = CoM_enc(0);
			temp_pose_msg.pose.position.y = CoM_enc(1);
			temp_pose_msg.pose.position.z = CoM_enc(2);
			temp_pose_msg.header.stamp = ros::Time::now();
			temp_pose_msg.header.frame_id = base_link_frame;
			rel_CoMPose_pub.publish(temp_pose_msg);
	}

}











void humanoid_ekf::publishJointEstimates() {

	joint_filt_msg.header.stamp = ros::Time::now();
	joint_filt_msg.name.resize(number_of_joints);
	joint_filt_msg.position.resize(number_of_joints);
	joint_filt_msg.velocity.resize(number_of_joints);

	for (unsigned int i = 0; i < number_of_joints; i++) {
		joint_filt_msg.position[i] = JointKF[i]->JointPosition;
		joint_filt_msg.velocity[i] = JointKF[i]->JointVelocity;
		joint_filt_msg.name[i] = JointKF[i]->JointName;
	}

	joint_filt_pub.publish(joint_filt_msg);

}







void humanoid_ekf::advertise() {

	bodyPose_est_pub = n.advertise<geometry_msgs::PoseStamped>(
	"/SERoW/body/pose", 10);


	bodyVel_est_pub = n.advertise<geometry_msgs::TwistStamped>(
	"/SERoW/body/vel", 10);

	bodyAcc_est_pub = n.advertise<sensor_msgs::Imu>(
	"/SERoW/body/acc", 10);

	supportPose_est_pub = n.advertise<geometry_msgs::PoseStamped>(
	"/SERoW/support/pose", 10);
	
	support_leg_pub = n.advertise<std_msgs::String>("/SERoW/support/leg",10);		

	odom_est_pub = n.advertise<nav_msgs::Odometry>("/SERoW/odom",10);		
	


	COP_pub = n.advertise<geometry_msgs::PointStamped>("SERoW/COP",10);

	CoM_pos_pub = n.advertise<geometry_msgs::PointStamped>("SERoW/CoM/pos",10);
	CoM_vel_pub = n.advertise<geometry_msgs::TwistStamped>("SERoW/CoM/vel",10);
	CoM_odom_pub = n.advertise<nav_msgs::Odometry>("/SERoW/CoM/odom",10);

	joint_filt_pub =  n.advertise<sensor_msgs::JointState>("/SERoW/joint_states",10);

	external_force_filt_pub = n.advertise<geometry_msgs::WrenchStamped>("/SERoW/CoM/forces",10);

	if(visualize_with_rviz)
	{
		support_path_msg.poses.resize(10);
		odom_path_msg.poses.resize(10);
		leg_odom_path_msg.poses.resize(10);
		com_path_msg.poses.resize(10);
		cop_path_msg.poses.resize(10);
		support_path_pub = n.advertise<nav_msgs::Path>("/SERoW/support/path",2);
		odom_path_pub = n.advertise<nav_msgs::Path>("/SERoW/odom/path",2);
		leg_odom_path_pub = n.advertise<nav_msgs::Path>("/SERoW/leg_odom/path",2);
		com_path_pub = n.advertise<nav_msgs::Path>("/SERoW/CoM/path",2);
		cop_path_pub = n.advertise<nav_msgs::Path>("/SERoW/COP/path",2);

		if(ground_truth)
		{
			ground_truth_odom_path_msg.poses.resize(10);
			ground_truth_com_path_msg.poses.resize(10);
			ground_truth_odom_path_pub = n.advertise<nav_msgs::Path>("/SERoW/ground_truth/odom/path",2);
			ground_truth_com_path_pub = n.advertise<nav_msgs::Path>("/SERoW/ground_truth/CoM/path",2);
		}
		if(comp_with){
			comp_odom_path_msg.poses.resize(10);
			comp_odom_path_pub = n.advertise<nav_msgs::Path>("/SERoW/comp/odom/path",10);
		}
			
	}
	if(ground_truth)
	{
		ground_truth_com_pub = n.advertise<nav_msgs::Odometry>("/SERoW/ground_truth/CoM/odom",2);
		ground_truth_odom_pub = n.advertise<nav_msgs::Odometry>("/SERoW/ground_truth/odom",2);
		ds_pub = n.advertise<std_msgs::Int32>("/SERoW/is_in_ds",10);

	}
	if(comp_with)
	{
		comp_odom_pub = n.advertise<nav_msgs::Odometry>("/SERoW/comp/odom",2);

	}

	if(debug_mode)
	{
		rel_supportPose_pub = n.advertise<geometry_msgs::PoseStamped>("/SERoW/rel_support/pose", 10);
		rel_swingPose_pub = n.advertise<geometry_msgs::PoseStamped>("/SERoW/rel_swing/pose", 10);
		rel_CoMPose_pub = n.advertise<geometry_msgs::PoseStamped>("/SERoW/rel_CoM/pose", 10);
		leg_odom_pub = n.advertise<nav_msgs::Odometry>("/SERoW/leg_odom",10);
		RLeg_est_pub = n.advertise<geometry_msgs::WrenchStamped>("SERoW/RLeg/GRF",10);
		LLeg_est_pub = n.advertise<geometry_msgs::WrenchStamped>("SERoW/LLeg/GRF",10);
	}

	
}


void humanoid_ekf::subscribeToJointState()
{
	joint_state_sub = n.subscribe(joint_state_topic,1000,&humanoid_ekf::joint_stateCb,this);
}

void humanoid_ekf::joint_stateCb(const sensor_msgs::JointState::ConstPtr& msg)
{
	joint_state_msg = *msg;
	joint_inc = true;

	if(firstJoint && useJointKF)
	{
		number_of_joints = joint_state_msg.name.size();
		JointKF = new JointSSKF*[number_of_joints];
		for (unsigned int i=0; i<number_of_joints; i++){
			JointKF[i] = new JointSSKF();									
			JointKF[i]->init(joint_state_msg.name[i]);
			JointKF[i]->setdt(1.0/joint_freq);
		}
		firstJoint = false;
	}

	for (unsigned int i=0; i< joint_state_msg.name.size(); i++){
		joint_map.insert(make_pair(joint_state_msg.name[i], joint_state_msg.position[i]));
		if(useJointKF && !firstJoint)
			JointKF[i]->filter(joint_state_msg.position[i]);
	}

}

void humanoid_ekf::subscribeToOdom()
{
	odom_sub = n.subscribe(odom_topic,1000,&humanoid_ekf::odomCb,this);
}

void humanoid_ekf::odomCb(const nav_msgs::Odometry::ConstPtr& msg)
{
	odom_msg = *msg;
	odom_inc = true;
	if(firstOdom){
		odom_msg_ = odom_msg;
		firstOdom = false;
	}
}


void humanoid_ekf::subscribeToCompOdom()
{
	compodom_sub = n.subscribe(comp_with_odom_topic,1000,&humanoid_ekf::compodomCb,this);
}

void humanoid_ekf::compodomCb(const nav_msgs::Odometry::ConstPtr& msg)
{
	comp_odom_msg = *msg;
}


void humanoid_ekf::subscribeToGroundTruth()
{
	ground_truth_odom_sub = n.subscribe(ground_truth_odom_topic,1000,&humanoid_ekf::ground_truth_odomCb,this);
}
void humanoid_ekf::ground_truth_odomCb(const nav_msgs::Odometry::ConstPtr& msg)
{
	ground_truth_odom_msg = *msg;
}

void humanoid_ekf::subscribeToGroundTruthCoM()
{
	ground_truth_com_sub = n.subscribe(ground_truth_com_topic,1000,&humanoid_ekf::ground_truth_comCb,this);
}
void humanoid_ekf::ground_truth_comCb(const nav_msgs::Odometry::ConstPtr& msg)
{
	ground_truth_com_odom_msg = *msg;
}




void humanoid_ekf::subscribeToSupportIdx()
{
	support_idx_sub = n.subscribe(support_idx_topic,1000,&humanoid_ekf::support_idxCb,this);
}
void humanoid_ekf::support_idxCb(const std_msgs::Int32::ConstPtr& msg)
{
	 support_idx_msg = *msg;
	 if(support_idx_msg.data == 1){
					support_leg = "LLeg";
					swing_leg = "RLeg";
					support_foot_frame = lfoot_frame;
					swing_foot_frame = rfoot_frame;
					legSwitch = true;

	  }
	  else
	  {
					support_leg = "RLeg";
					swing_leg = "LLeg";
					support_foot_frame = rfoot_frame;
					swing_foot_frame = lfoot_frame;
					legSwitch = true;
	  }
}




void humanoid_ekf::subscribeToDS()
{
	ds_sub = n.subscribe(is_in_ds_topic,1000,&humanoid_ekf::is_in_dsCb,this);
}
void humanoid_ekf::is_in_dsCb(const std_msgs::Bool::ConstPtr& msg)
{
	 if(msg->data)
		is_in_ds_msg.data = 1;
	 else
		is_in_ds_msg.data = 0;
}




void humanoid_ekf::subscribeToIMU()
{
	imu_sub = n.subscribe(imu_topic,1000,&humanoid_ekf::imuCb,this);
}
void humanoid_ekf::imuCb(const sensor_msgs::Imu::ConstPtr& msg)
{
	imu_msg = *msg;
	imu_inc = true;
}

void humanoid_ekf::subscribeToFSR()
{
	//Left Foot Wrench
	lfsr_sub = n.subscribe(lfsr_topic,1000,&humanoid_ekf::lfsrCb,this);
	//Right Foot Wrench
	rfsr_sub = n.subscribe(rfsr_topic,1000,&humanoid_ekf::rfsrCb,this);
	//Left COP
	copl_sub = n.subscribe(copr_topic,1000,&humanoid_ekf::coprCb,this);
	//Right COP
	copr_sub = n.subscribe(copl_topic,1000,&humanoid_ekf::coplCb,this);
}

void humanoid_ekf::lfsrCb(const geometry_msgs::WrenchStamped::ConstPtr& msg)
{
	lfsr_msg = *msg;
	fsr_inc = true;
}
void humanoid_ekf::rfsrCb(const geometry_msgs::WrenchStamped::ConstPtr& msg)
{
	rfsr_msg = *msg;
}

void humanoid_ekf::coplCb(const geometry_msgs::PointStamped::ConstPtr& msg)
{
	copl_msg = *msg;
}
void humanoid_ekf::coprCb(const geometry_msgs::PointStamped::ConstPtr& msg)
{
	copr_msg = *msg;
}




void humanoid_ekf::subscribeToPose()
{
	pose_sub = n.subscribe(pose_topic,1000,&humanoid_ekf::poseCb,this);
}

void humanoid_ekf::poseCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
	pose_msg = *msg;
	pose_inc = true;
	if(firstPose){
		pose_msg_ = pose_msg;
		firstPose = false;
	}
}
