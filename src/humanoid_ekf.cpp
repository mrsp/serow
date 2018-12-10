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
#include <serow/humanoid_ekf.h>


void humanoid_ekf::loadparams() {

	ros::NodeHandle n_p("~");
	// Load Server Parameters
   	n_p.param<std::string>("modelname",modelname,"nao.urdf");
	rd = new serow::robotDyn(modelname,false);

	n_p.param<std::string>("base_link",base_link_frame,"base_link");
	n_p.param<std::string>("lfoot",lfoot_frame,"l_ankle");
	n_p.param<std::string>("rfoot",rfoot_frame,"r_ankle");

	n_p.param<double>("imu_topic_freq",freq,100.0);
	n_p.param<double>("fsr_topic_freq",fsr_freq,100.0);
	n_p.param<double>("LLegUpThres", LLegUpThres,20.0);
	n_p.param<double>("LLegLowThres", LLegLowThres,15.0);
	n_p.param<double>("LosingContact", LosingContact,5.0);
	n_p.param<double>("StrikingContact", StrikingContact,5.0);

	n_p.param<bool>("useLegOdom",useLegOdom,false);
	n_p.param<bool>("usePoseUpdate",usePoseUpdate,false);

	n_p.param<bool>("ground_truth",ground_truth,false);
	n_p.param<bool>("debug_mode",debug_mode,false);

	n_p.param<bool>("support_idx_provided",support_idx_provided,false);
	if(support_idx_provided)
		n_p.param<std::string>("support_idx_topic", support_idx_topic,"support_idx");

	if(ground_truth)
	{
		n_p.param<std::string>("ground_truth_odom_topic", ground_truth_odom_topic,"ground_truth");
		n_p.param<std::string>("ground_truth_com_topic", ground_truth_com_topic,"ground_truth_com");
		n_p.param<std::string>("is_in_ds_topic", is_in_ds_topic,"is_in_ds_topic");
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
	std::vector<double> acc_list;
	n_p.getParam("T_B_A",acc_list);
	T_B_A(0,0) = acc_list[0];
	T_B_A(0,1) = acc_list[1];
	T_B_A(0,2) = acc_list[2];
	T_B_A(0,3) = acc_list[3];
	T_B_A(1,0) = acc_list[4];
	T_B_A(1,1) = acc_list[5];
	T_B_A(1,2) = acc_list[6];
	T_B_A(1,3) = acc_list[7];
	T_B_A(2,0) = acc_list[8];
	T_B_A(2,1) = acc_list[9];
	T_B_A(2,2) = acc_list[10];
	T_B_A(2,3) = acc_list[11];
	T_B_A(3,0) = acc_list[12];
	T_B_A(3,1) = acc_list[13];
	T_B_A(3,2) = acc_list[14];
	T_B_A(3,3) = acc_list[15];
		
	q_B_A = Quaterniond(T_B_A.linear());



	std::vector<double> gyro_list;
	n_p.getParam("T_B_G",gyro_list);
	T_B_G(0,0) = gyro_list[0];
	T_B_G(0,1) = gyro_list[1];
	T_B_G(0,2) = gyro_list[2];
	T_B_G(0,3) = gyro_list[3];
	T_B_G(1,0) = gyro_list[4];
	T_B_G(1,1) = gyro_list[5];
	T_B_G(1,2) = gyro_list[6];
	T_B_G(1,3) = gyro_list[7];
	T_B_G(2,0) = gyro_list[8];
	T_B_G(2,1) = gyro_list[9];
	T_B_G(2,2) = gyro_list[10];
	T_B_G(2,3) = gyro_list[11];
	T_B_G(3,0) = gyro_list[12];
	T_B_G(3,1) = gyro_list[13];
	T_B_G(3,2) = gyro_list[14];
	T_B_G(3,3) = gyro_list[15];
		
	q_B_G = Quaterniond(T_B_G.linear());



	n_p.param<std::string>("pose_topic", pose_topic,"pose");
	n_p.param<std::string>("odom_topic", odom_topic,"odom");
	n_p.param<std::string>("imu_topic", imu_topic,"imu");
	n_p.param<std::string>("joint_state_topic", joint_state_topic,"joint_states");
	n_p.param<std::string>("lfoot_force_torque_topic",lfsr_topic,"force_torque/left");
	n_p.param<std::string>("rfoot_force_torque_topic",rfsr_topic,"force_torque/right");

	n_p.param<std::string>("copl_topic",copl_topic,"cop/left");
	n_p.param<std::string>("copr_topic",copr_topic,"cor/right");


	n_p.param<bool>("estimateCoM", useCoMEKF,false);
	n_p.param<int>("medianWindow", medianWindow,15);



	//Madgwick Filter for Attitude Estimation
	n_p.param<double>("Madgwick_gain", beta,0.012f);
	n_p.param<bool>("useCF", useCF,true);
	n_p.param<double>("freqvmax", cf_freqvmax,2.5);
	n_p.param<double>("freqvmin", cf_freqvmin,0.1);


	mw =  new serow::Madgwick(freq,beta);

}

void humanoid_ekf::loadJointKFparams()
{
	ros::NodeHandle n_p("~");
	n_p.param<double>("joint_topic_freq",joint_freq,100.0);	
	n_p.param<double>("joint_cutoff_freq",joint_cutoff_freq,10.0);	
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
	n_p.param<double>("bias_ax",  imuEKF->bias_ax,0.0);
	n_p.param<double>("bias_ay",  imuEKF->bias_ay,0.0);
	n_p.param<double>("bias_az",  imuEKF->bias_az,0.0);
	n_p.param<double>("bias_gx",  imuEKF->bias_gx,0.0);
	n_p.param<double>("bias_gy",  imuEKF->bias_gy,0.0);
	n_p.param<double>("bias_gz",  imuEKF->bias_gz,0.0);
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

	n_p.param<double>("odom_position_noise_density", imuEKF->odom_px,1.0e-03);
	n_p.param<double>("odom_position_noise_density", imuEKF->odom_py,1.0e-03);
	n_p.param<double>("odom_position_noise_density", imuEKF->odom_pz,1.0e-03);
	n_p.param<double>("odom_orientation_noise_density", imuEKF->odom_ax,1.0e-03);
	n_p.param<double>("odom_orientation_noise_density", imuEKF->odom_ay,1.0e-03);
	n_p.param<double>("odom_orientation_noise_density", imuEKF->odom_az,1.0e-03);

	n_p.param<double>("velocity_noise_density", imuEKF->vel_px,1.0e-02);
	n_p.param<double>("velocity_noise_density", imuEKF->vel_py,1.0e-02);
	n_p.param<double>("velocity_noise_density", imuEKF->vel_pz,1.0e-02);


	n_p.param<double>("gravity", imuEKF->ghat,9.81);
	n_p.param<double>("gravity", g,9.81);
    n_p.param<bool>("useEuler", imuEKF->useEuler,true);
	imuEKF->setAccBias(T_B_A.linear()*Vector3d(bias_ax,bias_ay,bias_az));
	imuEKF->setGyroBias(T_B_G.linear()*Vector3d(bias_gx,bias_gy,bias_gz));
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
	n_p.param<bool>("useEuler", nipmEKF->useEuler,true);
}



humanoid_ekf::humanoid_ekf() 
{
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
	ROS_INFO_STREAM("SERoW Initializing...");

	// Initialize ROS nodes
	n = nh;
	// Load ROS Parameters
	loadparams();
	//Initialization
	init();
	loadJointKFparams();
	// Load IMU parameters
	loadIMUEKFparams();


	if(useCoMEKF)
		loadCoMEKFparams();


	//Subscribe/Publish ROS Topics/Services
	subscribe();
	advertise();

	dynamic_recfg_ = boost::make_shared< dynamic_reconfigure::Server<serow::VarianceControlConfig> >(n);
    dynamic_reconfigure::Server<serow::VarianceControlConfig>::CallbackType cb = boost::bind(&humanoid_ekf::reconfigureCB, this, _1, _2);
    dynamic_recfg_->setCallback(cb);
	is_connected_ = true;

	ros::Duration(1.0).sleep();
	ROS_INFO_STREAM("SERoW Initialized");

	return true;
}



void humanoid_ekf::reconfigureCB(serow::VarianceControlConfig& config, uint32_t level)
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

	subscribeToJointState();
	
	if(!useLegOdom){
		if (usePoseUpdate){
			subscribeToPose();
		}
		else{
			subscribeToOdom();
		}
	}

	if(ground_truth){
		subscribeToGroundTruth();
		subscribeToGroundTruthCoM();
		subscribeToDS();
	}

	if(support_idx_provided)
		subscribeToSupportIdx();

}

void humanoid_ekf::init() {


	/** Initialize Variables **/
	//Kinematic TFs
	Tws = Affine3d::Identity();
	Twb = Affine3d::Identity();
	Twb_ = Twb;
	Tbs = Affine3d::Identity();
	LLegGRF = Vector3d::Zero();
	RLegGRF = Vector3d::Zero();
	LLegGRT = Vector3d::Zero();
	RLegGRT = Vector3d::Zero();
	copl = Vector3d::Zero();
	copr = Vector3d::Zero();
	omegawb = Vector3d::Zero();
	vwb = Vector3d::Zero();
	omegabl = Vector3d::Zero();
	omegabr = Vector3d::Zero();
	vbl= Vector3d::Zero();
	vbr= Vector3d::Zero();
	Twl = Affine3d::Identity();
	Twr = Affine3d::Identity();
	Tbl = Affine3d::Identity();
	Tbr = Affine3d::Identity();
	vwl= Vector3d::Zero();
	vwr= Vector3d::Zero();


	no_motion_residual = Vector3d::Zero();
	firstrun = true;
	firstUpdate = true;
	firstGyrodot = true;
	LLegST = false;
	RLegST = false;
	firstContact = true;

	// Initialize the IMU based EKF 
	imuEKF = new IMUEKF;
	imuEKF->init();

	if(useCoMEKF){
		if(useGyroLPF){
			gyroLPF = new butterworthLPF*[3];
			for(unsigned int i=0;i<3;i++)
				gyroLPF[i] = new butterworthLPF();	
		}
		else{
			gyroMAF = new MovingAverageFilter*[3];
			for(unsigned int i=0;i<3;i++)
				gyroMAF[i] = new MovingAverageFilter();			
		}

		nipmEKF = new CoMEKF;
		nipmEKF->init();
	}
	imu_inc = false;
	fsr_inc = false;
	pose_inc = false;
	joint_inc = false;
	odom_inc = false;
	leg_odom_inc = false;
	leg_vel_inc = false;
	support_inc = false;
    no_motion_indicator = false;
	no_motion_it = 0;
	no_motion_threshold = 5e-4;
	no_motion_it_threshold = 500;
    
	lmdf = MediatorNew(medianWindow);
	rmdf = MediatorNew(medianWindow);
	//llmdf = new WindowMedian<double>(medianWindow);
	//rrmdf = new WindowMedian<double>(medianWindow);

	LLegForceFilt = 0;
	RLegForceFilt = 0;
}



void humanoid_ekf::run() {
	
	static ros::Rate rate(1.05*freq);  //ROS Node Loop Rate
	while (ros::ok()){
		if(imu_inc){
		predictWithImu = false;
		predictWithCoM = false;

		mw->MadgwickAHRSupdateIMU(T_B_G.linear() * (Vector3d(imu_msg.angular_velocity.x,imu_msg.angular_velocity.y,imu_msg.angular_velocity.z)-Vector3d(bias_gx,bias_gy,bias_gz)),
			T_B_A.linear()*(Vector3d(imu_msg.linear_acceleration.x,imu_msg.linear_acceleration.y,imu_msg.linear_acceleration.z)-Vector3d(bias_ax,bias_ay,bias_az)));


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
			publishJointEstimates();
			publishBodyEstimates();
			publishLegEstimates();
			publishSupportEstimates();
			publishContact();
			publishGRF();
			
			if(useCoMEKF){
				publishCoMEstimates();
				publishCOP();
			}
		}
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
			imuEKF->firstrun = false;
		}


		//Compute the attitude and posture with the IMU-Kinematics Fusion
		//Predict with the IMU gyro and acceleration
		if(imu_inc && !predictWithImu && !imuEKF->firstrun){
			imuEKF->predict( T_B_G.linear() * Vector3d(imu_msg.angular_velocity.x,imu_msg.angular_velocity.y,imu_msg.angular_velocity.z),
			T_B_A.linear()*Vector3d(imu_msg.linear_acceleration.x,imu_msg.linear_acceleration.y,imu_msg.linear_acceleration.z));
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
						if(leg_odom_inc){
								pos_update += Twb.translation()-Twb_.translation();
								Quaterniond q_now = qwb;
								Quaterniond q_prev = qwb_;
								q_update *=  q_now  * q_prev.inverse();
								leg_odom_inc = false;
								imuEKF->updateWithOdom(pos_update, q_update);
								//STORE POS 
								if(odom_inc){
									 odom_msg_ = odom_msg;
									 odom_inc = false;
								}
								if(pose_inc){
									pose_inc = false;
									pose_msg_ = pose_msg;
								}

						}
				}
				else
				{
					if(!usePoseUpdate)
					{

						if(leg_vel_inc)
						{
							imuEKF->updateWithTwist(vwb);
							leg_vel_inc = false;
						}
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

		/*
		//Update with the Support foot position and orientation
		if(support_inc && predictWithImu){
			imuEKF->updateWithSupport(Tbs.translation(),qbs);
			support_inc=false;
		}
		*/

		//Estimated TFs for Legs and Support foot
		Twl = Twb * Tbl;
		Twr = Twb * Tbr;
		qwl = Quaterniond(Twl.linear());
		qwr = Quaterniond(Twr.linear());
		Tws = imuEKF->Tib * Tbs;
		qws = Quaterniond(Tws.linear());
}


void humanoid_ekf::estimateWithCoMEKF()
{
	if(joint_inc){
			if (nipmEKF->firstrun){
				nipmEKF->setdt(1.0/fsr_freq);
				nipmEKF->setParams(mass,I_xx,I_yy,g);
				nipmEKF->setCoMPos(Twb*CoM_enc);
				nipmEKF->setCoMExternalForce(Vector3d(bias_fx,bias_fy,bias_fz));
				nipmEKF->firstrun = false;
				if(useGyroLPF){
					gyroLPF[0]->init("gyro X LPF", freq, gyro_fx);
					gyroLPF[1]->init("gyro Y LPF", freq, gyro_fy);
					gyroLPF[2]->init("gyro Z LPF", freq, gyro_fz);
				}
				else{
					for(unsigned int i=0;i<3;i++)
						gyroMAF[i]->setParams(maWindow);
				}
				
			}
	}
	
	//Compute the COP in the Inertial Frame
		if(fsr_inc && predictWithImu && !predictWithCoM && !nipmEKF->firstrun){
			computeCOP(Twl,Twr);
			//Numerically compute the Gyro acceleration in the Inertial Frame and use a 3-Point Low-Pass filter
			filterGyrodot();
			DiagonalMatrix<double,3> Inertia(I_xx,I_yy,I_zz);
			nipmEKF->predict(COP_fsr,GRF_fsr,imuEKF->Rib*Inertia*Gyrodot);
			fsr_inc = false;
			predictWithCoM = true;
		}

		if(joint_inc && predictWithCoM){
			nipmEKF->update(
					imuEKF->acc - imuEKF->g,
					imuEKF->Tib*CoM_enc,
					imuEKF->gyro,Gyrodot);
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


	//Update Pinnochio
	rd->updateJointConfig(joint_state_pos_map,joint_state_vel_map);

	//Get the CoM w.r.t Body Frame
	CoM_enc = rd->comPosition();

	mass = m;
	support_inc = true;
	Tbl.translation() = rd->linkPosition(lfoot_frame);
	qbl = rd->linkOrientation(lfoot_frame);
	Tbl.linear() = qbl.toRotationMatrix();

	Tbr.translation() = rd->linkPosition(rfoot_frame);
	qbr = rd->linkOrientation(rfoot_frame);
	Tbr.linear() = qbr.toRotationMatrix();


	Tbs=Tbl;
	qbs=qbl;
	if(support_leg=="RLeg")
	{
		Tbs = Tbr;
		qbs = qbr;
	}


	//TF Initialization
	if (firstrun){
			Twl.translation() << Tbl.translation()(0), Tbl.translation()(1), 0.00;
			Twl.linear() = Tbl.linear();
			Twr.translation() << Tbr.translation()(0), Tbr.translation()(1), 0.00;
			Twr.linear() = Tbr.linear();
			dr = new serow::deadReckoning(Twl.translation(), Twr.translation(), Twl.linear(), Twr.linear(),
                       mass, 0.3, 1.0, freq, g, useCF, cf_freqvmin, cf_freqvmax);
	}

	

		
		//Differential Kinematics with Pinnochio
		omegabl = rd->getAngularVelocity(lfoot_frame);
		omegabr = rd->getAngularVelocity(rfoot_frame);
		vbl =  rd->getLinearVelocity(lfoot_frame);
		vbr =  rd->getLinearVelocity(rfoot_frame);


		dr->computeDeadReckoning(mw->getR(),  Tbl.linear(),  Tbr.linear(), mw->getGyro(), Tbl.translation(),  Tbr.translation(), vbl,  vbr, omegabl,  omegabr, 
		LLegForceFilt, RLegForceFilt, mw->getAcc());
		
		Twb_ = Twb;
		qwb_ = qwb;
		Twb.translation() = dr->getOdom();
		qwb = Quaterniond(mw->getR());
		vwb = dr->getLinearVel();
		omegawb = mw->getGyro();
		if(firstrun)
		{
			Twb_ = Twb;
			qwb_ = qwb;
			firstrun = false;
		}
	
		vwl = dr->getLFootLinearVel();
		vwr = dr->getRFootLinearVel();

		omegawl = dr->getLFootAngularVel();
		omegawr = dr->getRFootAngularVel();


		leg_odom_inc = true;
		leg_vel_inc = true;
		check_no_motion = false;


}







/* Schmidtt Trigger */
void humanoid_ekf::determineLegContact() {
	
	//Choose Initial Support Foot based on Contact Force
	if(firstContact){
		if(LLegGRF(2)>RLegGRF(2)){
				// Initial support leg 
					support_leg = "LLeg";
					support_foot_frame = lfoot_frame;
		}
		else
		{
					support_leg = "RLeg";
					support_foot_frame = rfoot_frame;
		}
		firstContact = false;
	}
	else{
		//Determine if the Support Foot changed  
		if(!support_idx_provided){
			
			if (vwl.norm()<0.05)
			{
				if ( LLegForceFilt > LLegUpThres  && LLegForceFilt<StrikingContact)
				{
					LLegST = true;
				}
			}
			else{
				if (LLegForceFilt < LLegLowThres)
				{
					LLegST = false;
				}
			}
		
			if (vwr.norm()<0.05)
			{
				if ( RLegForceFilt > LLegUpThres  && RLegForceFilt<StrikingContact)
				{
					RLegST = true;
				}
			}
			else{
				if (RLegForceFilt < LLegLowThres)
				{
					RLegST = false;
				}
			}


		if(LLegST && RLegST)
		{
			if(LLegForceFilt>RLegForceFilt)
			{
				support_leg = "LLeg";
				support_foot_frame = lfoot_frame;
			}
			else
			{
				 support_leg = "RLeg";
				 support_foot_frame = rfoot_frame;
			}
		}
		else if(LLegST)
		{
				support_leg = "LLeg";
				support_foot_frame = lfoot_frame;
		}
		else if(RLegST)
		{
			support_leg = "RLeg";
			support_foot_frame = rfoot_frame;
		}

		}
	}
}
  







void humanoid_ekf::deAllocate()
{
	for (unsigned int i = 0; i < number_of_joints; i++)
		delete[] JointVF[i];
	delete[] JointVF;
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
	delete rd;
	delete mw;
	delete dr;
}





void humanoid_ekf::filterGyrodot() {
	if (!firstGyrodot) {
		//Compute numerical derivative
		Gyrodot = (imuEKF->gyro - Gyro_)*freq;
		if(useGyroLPF){
			Gyrodot(0) = gyroLPF[0]->filter(Gyrodot(0));
			Gyrodot(1) = gyroLPF[1]->filter(Gyrodot(1));
			Gyrodot(2) = gyroLPF[2]->filter(Gyrodot(2));
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
	Gyro_ = imuEKF->gyro;
}


void humanoid_ekf::publishBodyEstimates() {


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
	odom_est_msg.pose.pose.position.x = imuEKF->rX;
	odom_est_msg.pose.pose.position.y = imuEKF->rY;
	odom_est_msg.pose.pose.position.z = imuEKF->rZ;
	odom_est_msg.pose.pose.orientation.x = imuEKF->qib.x();
	odom_est_msg.pose.pose.orientation.y = imuEKF->qib.y();
	odom_est_msg.pose.pose.orientation.z = imuEKF->qib.z();
	odom_est_msg.pose.pose.orientation.w = imuEKF->qib.w();

	odom_est_msg.twist.twist.linear.x = imuEKF->velX;
	odom_est_msg.twist.twist.linear.y = imuEKF->velY;
	odom_est_msg.twist.twist.linear.z = imuEKF->velZ;
	odom_est_msg.twist.twist.angular.x  = imuEKF->gyroX;
	odom_est_msg.twist.twist.angular.y  = imuEKF->gyroY;
	odom_est_msg.twist.twist.angular.z  = imuEKF->gyroZ;

	//for(int i=0;i<36;i++)
    //odom_est_msg.pose.covariance[i] = 0;
	odom_est_pub.publish(odom_est_msg);


		leg_odom_msg.header.stamp=ros::Time::now();
		leg_odom_msg.header.frame_id = "odom";
		leg_odom_msg.pose.pose.position.x = Twb.translation()(0);
		leg_odom_msg.pose.pose.position.y = Twb.translation()(1);
		leg_odom_msg.pose.pose.position.z = Twb.translation()(2);
		leg_odom_msg.pose.pose.orientation.x = qwb.x();
		leg_odom_msg.pose.pose.orientation.y = qwb.y();
		leg_odom_msg.pose.pose.orientation.z = qwb.z();
		leg_odom_msg.pose.pose.orientation.w = qwb.w();
		leg_odom_msg.twist.twist.linear.x = vwb(0);
		leg_odom_msg.twist.twist.linear.y = vwb(1);
		leg_odom_msg.twist.twist.linear.z = vwb(2);
		leg_odom_msg.twist.twist.angular.x = omegawb(0);
		leg_odom_msg.twist.twist.angular.y = omegawb(1);
		leg_odom_msg.twist.twist.angular.z = omegawb(2);
		leg_odom_pub.publish(leg_odom_msg);
	

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
}

void humanoid_ekf::publishSupportEstimates() {
	supportPose_est_msg.header.stamp = ros::Time::now();
	supportPose_est_msg.header.frame_id = "odom";
	supportPose_est_msg.pose.position.x = Tws.translation()(0);
	supportPose_est_msg.pose.position.y = Tws.translation()(1);
	supportPose_est_msg.pose.position.z = Tws.translation()(2);
	supportPose_est_msg.pose.orientation.x = qws.x();
	supportPose_est_msg.pose.orientation.y = qws.y();
	supportPose_est_msg.pose.orientation.z = qws.z();
	supportPose_est_msg.pose.orientation.w = qws.w();
	supportPose_est_pub.publish(supportPose_est_msg);
}


void humanoid_ekf::publishLegEstimates() {
		leftleg_odom_msg.header.stamp=ros::Time::now();
		leftleg_odom_msg.header.frame_id = "odom";
		leftleg_odom_msg.pose.pose.position.x = Twl.translation()(0);
		leftleg_odom_msg.pose.pose.position.y = Twl.translation()(1);
		leftleg_odom_msg.pose.pose.position.z = Twl.translation()(2);
		leftleg_odom_msg.pose.pose.orientation.x = qwl.x();
		leftleg_odom_msg.pose.pose.orientation.y = qwl.y();
		leftleg_odom_msg.pose.pose.orientation.z = qwl.z();
		leftleg_odom_msg.pose.pose.orientation.w = qwl.w();
		leftleg_odom_msg.twist.twist.linear.x = vwl(0);
		leftleg_odom_msg.twist.twist.linear.y = vwl(1);
		leftleg_odom_msg.twist.twist.linear.z = vwl(2);
		leftleg_odom_msg.twist.twist.angular.x = omegawl(0);
		leftleg_odom_msg.twist.twist.angular.y = omegawl(1);
		leftleg_odom_msg.twist.twist.angular.z = omegawl(2);
		leftleg_odom_pub.publish(leftleg_odom_msg);

		rightleg_odom_msg.header.stamp=ros::Time::now();
		rightleg_odom_msg.header.frame_id = "odom";
		rightleg_odom_msg.pose.pose.position.x = Twr.translation()(0);
		rightleg_odom_msg.pose.pose.position.y = Twr.translation()(1);
		rightleg_odom_msg.pose.pose.position.z = Twr.translation()(2);
		rightleg_odom_msg.pose.pose.orientation.x = qwr.x();
		rightleg_odom_msg.pose.pose.orientation.y = qwr.y();
		rightleg_odom_msg.pose.pose.orientation.z = qwr.z();
		rightleg_odom_msg.pose.pose.orientation.w = qwr.w();
		rightleg_odom_msg.twist.twist.linear.x = vwr(0);
		rightleg_odom_msg.twist.twist.linear.y = vwr(1);
		rightleg_odom_msg.twist.twist.linear.z = vwr(2);
		rightleg_odom_msg.twist.twist.angular.x = omegawr(0);
		rightleg_odom_msg.twist.twist.angular.y = omegawr(1);
		rightleg_odom_msg.twist.twist.angular.z = omegawr(2);
		rightleg_odom_pub.publish(rightleg_odom_msg);

		if(debug_mode)
		{
			temp_pose_msg.pose.position.x = Tbl.translation()(0);
			temp_pose_msg.pose.position.y = Tbl.translation()(1);
			temp_pose_msg.pose.position.z = Tbl.translation()(2);
			temp_pose_msg.pose.orientation.x = qbl.x();
			temp_pose_msg.pose.orientation.y = qbl.y();
			temp_pose_msg.pose.orientation.z = qbl.z();
			temp_pose_msg.pose.orientation.w = qbl.w();
			temp_pose_msg.header.stamp = ros::Time::now();
			temp_pose_msg.header.frame_id = base_link_frame;
			rel_leftlegPose_pub.publish(temp_pose_msg);

			temp_pose_msg.pose.position.x = Tbr.translation()(0);
			temp_pose_msg.pose.position.y = Tbr.translation()(1);
			temp_pose_msg.pose.position.z = Tbr.translation()(2);
			temp_pose_msg.pose.orientation.x = qbr.x();
			temp_pose_msg.pose.orientation.y = qbr.y();
			temp_pose_msg.pose.orientation.z = qbr.z();
			temp_pose_msg.pose.orientation.w = qbr.w();

			temp_pose_msg.header.stamp = ros::Time::now();
			temp_pose_msg.header.frame_id = base_link_frame;
			rel_rightlegPose_pub.publish(temp_pose_msg);	
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


void humanoid_ekf::computeCOP(Affine3d Twl_, Affine3d Twr_) {


	Vector3d coplw, coprw;
	double weightl, weightr;



	// Computation of the CoP in the Local Coordinate Frame of the Foot
	copl = Vector3d(copl_msg.point.x, copl_msg.point.y,0);
	copr = Vector3d(copr_msg.point.x, copr_msg.point.y,0);
	weightl = LLegGRF(2)/g;
	weightr = RLegGRF(2)/g;

	if(LLegGRF(2) < LosingContact)
	{
		copl  = Vector3d::Zero();
		weightl = 0.0;
	}
	if(RLegGRF(2) < LosingContact)
	{
		copr  = Vector3d::Zero();
		weightr = 0.0;
	}

	// Compute the CoP wrt the Support Foot Frame 
	coplw = Twl_ * copl;
	coprw = Twr_ * copr;

	COP_fsr = (weightl * coplw + weightr * coprw)/(weightl+weightr);

	GRF_fsr = Twl_.linear()*LLegGRF;
	GRF_fsr += Twr_.linear()*RLegGRF;


}



void humanoid_ekf::publishCOP() {
	COP_msg.point.x = COP_fsr(0);
	COP_msg.point.y = COP_fsr(1);
	COP_msg.point.z = COP_fsr(2);
	COP_msg.header.stamp = ros::Time::now();
	COP_msg.header.frame_id = "odom";
	COP_pub.publish(COP_msg);
}


void humanoid_ekf::publishCoMEstimates() {
	CoM_odom_msg.header.stamp=ros::Time::now();
	CoM_odom_msg.header.frame_id = "odom";
	CoM_odom_msg.pose.pose.position.x = nipmEKF->comX;
	CoM_odom_msg.pose.pose.position.y = nipmEKF->comY;
	CoM_odom_msg.pose.pose.position.z = nipmEKF->comZ;
	CoM_odom_msg.twist.twist.linear.x = nipmEKF->velX;
	CoM_odom_msg.twist.twist.linear.y = nipmEKF->velY;
	CoM_odom_msg.twist.twist.linear.z = nipmEKF->velZ;
	//for(int i=0;i<36;i++)
    //odom_est_msg.pose.covariance[i] = 0;
	CoM_odom_pub.publish(CoM_odom_msg);
	external_force_filt_msg.header.frame_id = "odom";
	external_force_filt_msg.header.stamp = ros::Time::now();
	external_force_filt_msg.wrench.force.x = nipmEKF->fX;
	external_force_filt_msg.wrench.force.y = nipmEKF->fY;
	external_force_filt_msg.wrench.force.z = nipmEKF->fZ;
	external_force_filt_pub.publish(external_force_filt_msg);


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
		joint_filt_msg.position[i] = JointVF[i]->JointPosition;
		joint_filt_msg.velocity[i] = JointVF[i]->JointVelocity;
		joint_filt_msg.name[i] = JointVF[i]->JointName;
	}

	joint_filt_pub.publish(joint_filt_msg);

}







void humanoid_ekf::advertise() {



	supportPose_est_pub =  n.advertise<geometry_msgs::PoseStamped>(
	"/SERoW/support/pose", 1000);

	bodyAcc_est_pub = n.advertise<sensor_msgs::Imu>(
	"/SERoW/body/acc", 1000);

	leftleg_odom_pub = n.advertise<nav_msgs::Odometry>(
	"/SERoW/LLeg/odom", 1000);

	rightleg_odom_pub = n.advertise<nav_msgs::Odometry>(
	"/SERoW/RLeg/odom", 1000);

	support_leg_pub = n.advertise<std_msgs::String>("/SERoW/support/leg",1000);		

	odom_est_pub = n.advertise<nav_msgs::Odometry>("/SERoW/odom",1000);		
	

	COP_pub = n.advertise<geometry_msgs::PointStamped>("SERoW/COP",1000);

	CoM_odom_pub = n.advertise<nav_msgs::Odometry>("/SERoW/CoM/odom",1000);

	joint_filt_pub =  n.advertise<sensor_msgs::JointState>("/SERoW/joint_states",1000);

	external_force_filt_pub = n.advertise<geometry_msgs::WrenchStamped>("/SERoW/CoM/forces",1000);
	leg_odom_pub = n.advertise<nav_msgs::Odometry>("/SERoW/leg_odom",1000);

	if(ground_truth)
	{
		ground_truth_com_pub = n.advertise<nav_msgs::Odometry>("/SERoW/ground_truth/CoM/odom",1000);
		ground_truth_odom_pub = n.advertise<nav_msgs::Odometry>("/SERoW/ground_truth/odom",1000);
		ds_pub = n.advertise<std_msgs::Int32>("/SERoW/is_in_ds",1000);
	}

	if(debug_mode)
	{
		rel_leftlegPose_pub = n.advertise<geometry_msgs::PoseStamped>("/SERoW/rel_LLeg/pose", 1000);
		rel_rightlegPose_pub = n.advertise<geometry_msgs::PoseStamped>("/SERoW/rel_RLeg/pose", 1000);
		rel_CoMPose_pub = n.advertise<geometry_msgs::PoseStamped>("/SERoW/rel_CoM/pose", 1000);
		RLeg_est_pub = n.advertise<geometry_msgs::WrenchStamped>("SERoW/RLeg/GRF",1000);
		LLeg_est_pub = n.advertise<geometry_msgs::WrenchStamped>("SERoW/LLeg/GRF",1000);
	}

	
}


void humanoid_ekf::subscribeToJointState()
{
	joint_state_sub = n.subscribe(joint_state_topic,1,&humanoid_ekf::joint_stateCb,this,ros::TransportHints().tcpNoDelay());
	firstJointStates = true;

}

void humanoid_ekf::joint_stateCb(const sensor_msgs::JointState::ConstPtr& msg)
{
	joint_state_msg = *msg;
	joint_inc = true;


	if(firstJointStates)
	{
		number_of_joints = joint_state_msg.name.size();
		joint_state_vel.resize(number_of_joints);
		joint_state_pos.resize(number_of_joints);
		JointVF = new JointDF*[number_of_joints];
		for (unsigned int i=0; i<number_of_joints; i++){
			JointVF[i] = new JointDF();									
			JointVF[i]->init(joint_state_msg.name[i],joint_freq,joint_cutoff_freq);
		}
		firstJointStates = false;
	}


	for (unsigned int i=0; i< joint_state_msg.name.size(); i++){
				joint_state_pos[i]=joint_state_msg.position[i];
			    joint_state_vel[i]=JointVF[i]->filter(joint_state_msg.position[i]);
				joint_state_pos_map[joint_state_msg.name[i]]=joint_state_pos[i];
				joint_state_vel_map[joint_state_msg.name[i]]=joint_state_vel[i];
	}

}

void humanoid_ekf::subscribeToOdom()
{
	odom_sub = n.subscribe(odom_topic,1,&humanoid_ekf::odomCb,this,ros::TransportHints().tcpNoDelay());
	firstOdom = true;

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



void humanoid_ekf::subscribeToGroundTruth()
{
	ground_truth_odom_sub = n.subscribe(ground_truth_odom_topic,1,&humanoid_ekf::ground_truth_odomCb,this,ros::TransportHints().tcpNoDelay());
	firstGT = true;
}
void humanoid_ekf::ground_truth_odomCb(const nav_msgs::Odometry::ConstPtr& msg)
{
	if(!firstrun){
		ground_truth_odom_msg = *msg;

		if(firstGT){
			offsetGT = Vector3d::Zero();
			offsetGT(0) = ground_truth_odom_msg.pose.pose.position.x - Twb.translation()(0);
			offsetGT(1) = ground_truth_odom_msg.pose.pose.position.y - Twb.translation()(1);
			offsetGT(2) = ground_truth_odom_msg.pose.pose.position.z - Twb.translation()(2);
			firstGT=false;
		}
		
		ground_truth_odom_msg.pose.pose.position.x = ground_truth_odom_msg.pose.pose.position.x - offsetGT(0);
		ground_truth_odom_msg.pose.pose.position.y = ground_truth_odom_msg.pose.pose.position.y - offsetGT(1);
		ground_truth_odom_msg.pose.pose.position.z = ground_truth_odom_msg.pose.pose.position.z - offsetGT(2);

	}

}

void humanoid_ekf::subscribeToGroundTruthCoM()
{
	ground_truth_com_sub = n.subscribe(ground_truth_com_topic,1000,&humanoid_ekf::ground_truth_comCb,this,ros::TransportHints().tcpNoDelay());
	firstGTCoM=true;
}
void humanoid_ekf::ground_truth_comCb(const nav_msgs::Odometry::ConstPtr& msg)
{
	if(!firstrun){
		ground_truth_com_odom_msg = *msg;

		if(firstGTCoM){
			offsetGTCoM = Vector3d::Zero();
			Vector3d tempCoMOffset = Twb * CoM_enc;
			offsetGTCoM(0) = ground_truth_com_odom_msg.pose.pose.position.x - tempCoMOffset(0);
			offsetGTCoM(1) = ground_truth_com_odom_msg.pose.pose.position.y - tempCoMOffset(1);
			offsetGTCoM(2) = ground_truth_com_odom_msg.pose.pose.position.z - tempCoMOffset(2);
			firstGTCoM=false;
		}
		ground_truth_com_odom_msg.pose.pose.position.x = ground_truth_com_odom_msg.pose.pose.position.x - offsetGTCoM(0);
		ground_truth_com_odom_msg.pose.pose.position.y = ground_truth_com_odom_msg.pose.pose.position.y - offsetGTCoM(1);
		ground_truth_com_odom_msg.pose.pose.position.z = ground_truth_com_odom_msg.pose.pose.position.z - offsetGTCoM(2);
	}		
}




void humanoid_ekf::subscribeToSupportIdx()
{
	support_idx_sub = n.subscribe(support_idx_topic,1,&humanoid_ekf::support_idxCb,this,ros::TransportHints().tcpNoDelay());
}
void humanoid_ekf::support_idxCb(const std_msgs::Int32::ConstPtr& msg)
{
	 support_idx_msg = *msg;
	 if(support_idx_msg.data == 1){
					support_leg = "LLeg";
					support_foot_frame = lfoot_frame;
	  }
	  else
	  {
					support_leg = "RLeg";
					support_foot_frame = rfoot_frame;
	  }
}




void humanoid_ekf::subscribeToDS()
{
	ds_sub = n.subscribe(is_in_ds_topic,1,&humanoid_ekf::is_in_dsCb,this,ros::TransportHints().tcpNoDelay());
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
	imu_sub = n.subscribe(imu_topic,1,&humanoid_ekf::imuCb,this,ros::TransportHints().tcpNoDelay());
}
void humanoid_ekf::imuCb(const sensor_msgs::Imu::ConstPtr& msg)
{
	imu_msg = *msg;
	imu_inc = true;
}

void humanoid_ekf::subscribeToFSR()
{
	//Left Foot Wrench
	lfsr_sub = n.subscribe(lfsr_topic,1,&humanoid_ekf::lfsrCb,this,ros::TransportHints().tcpNoDelay());
	//Right Foot Wrench
	rfsr_sub = n.subscribe(rfsr_topic,1,&humanoid_ekf::rfsrCb,this,ros::TransportHints().tcpNoDelay());
	//Left COP
	copl_sub = n.subscribe(copr_topic,1,&humanoid_ekf::coprCb,this,ros::TransportHints().tcpNoDelay());
	//Right COP
	copr_sub = n.subscribe(copl_topic,1,&humanoid_ekf::coplCb,this,ros::TransportHints().tcpNoDelay());
}

void humanoid_ekf::lfsrCb(const geometry_msgs::WrenchStamped::ConstPtr& msg)
{
	lfsr_msg = *msg;

	MediatorInsert(lmdf,lfsr_msg.wrench.force.z);
	LLegForceFilt = MediatorMedian(lmdf);
	
	//llmdf->insert((double)lfsr_msg.wrench.force.z);
	//LLegForceFilt = llmdf->median();
	fsr_inc = true;
	
}
void humanoid_ekf::rfsrCb(const geometry_msgs::WrenchStamped::ConstPtr& msg)
{
	rfsr_msg = *msg;

	MediatorInsert(rmdf,rfsr_msg.wrench.force.z);
	RLegForceFilt = MediatorMedian(rmdf);
	//rrmdf->insert((double)rfsr_msg.wrench.force.z);
	//RLegForceFilt = rrmdf->median();
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
	pose_sub = n.subscribe(pose_topic,1,&humanoid_ekf::poseCb,this,ros::TransportHints().tcpNoDelay());
	firstPose = true;

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
