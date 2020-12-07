/*
 * SERoW - a complete state estimation scheme for humanoid robots
 *
 * Copyright 2017-2020 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
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

void humanoid_ekf::loadparams()
{
    ros::NodeHandle n_p("~");
    // Load Server Parameters
    n_p.param<std::string>("modelname", modelname, "nao.urdf");
    rd = new serow::robotDyn(modelname, false);
    n_p.param<std::string>("base_link", base_link_frame, "base_link");
    n_p.param<std::string>("lfoot", lfoot_frame, "l_ankle");
    n_p.param<std::string>("rfoot", rfoot_frame, "r_ankle");
    n_p.param<double>("imu_topic_freq", freq, 100.0);
    n_p.param<double>("ft_topic_freq", ft_freq, freq);
    n_p.param<double>("joint_topic_freq", joint_freq, 100.0);
    
    freq = min(min(freq,ft_freq),joint_freq);
    cout<<"Estimation Rate is "<<freq<<endl;

    n_p.param<bool>("useInIMUEKF", useInIMUEKF, false);
    n_p.param<double>("VelocityThres", VelocityThres, 0.5);
    n_p.param<double>("LosingContact", LosingContact, 5.0);
    n_p.param<bool>("useGEM", useGEM, false);
    n_p.param<bool>("calibrateIMUbiases", calibrateIMU, true);
    n_p.param<int>("maxImuCalibrationCycles", maxImuCalibrationCycles, 500);
    n_p.param<bool>("computeJointVelocity", computeJointVelocity, true);

    if (useGEM)
    {
        n_p.param<double>("foot_polygon_xmin", foot_polygon_xmin, -0.103);
        n_p.param<double>("foot_polygon_xmax", foot_polygon_xmax, 0.107);
        n_p.param<double>("foot_polygon_ymin", foot_polygon_ymin, -0.055);
        n_p.param<double>("foot_polygon_ymax", foot_polygon_ymax, 0.055);
        n_p.param<double>("lforce_sigma", lforce_sigma, 2.2734);
        n_p.param<double>("rforce_sigma", rforce_sigma, 5.6421);
        n_p.param<double>("lcop_sigma", lcop_sigma, 0.005);
        n_p.param<double>("rcop_sigma", rcop_sigma, 0.005);
        n_p.param<double>("lvnorm_sigma", lvnorm_sigma, 0.1);
        n_p.param<double>("rvnorm_sigma", rvnorm_sigma, 0.1);
        n_p.param<double>("probabilisticContactThreshold", probabilisticContactThreshold, 0.95);
        n_p.param<bool>("ContactDetectionWithCOP", ContactDetectionWithCOP, false);
        n_p.param<bool>("ContactDetectionWithKinematics", ContactDetectionWithKinematics, false);
    }
    else
    {
        n_p.param<double>("LegUpThres", LegHighThres, 20.0);
        n_p.param<double>("LegLowThres", LegLowThres, 15.0);
        n_p.param<double>("StrikingContact", StrikingContact, 5.0);
    }

    n_p.param<bool>("useLegOdom", useLegOdom, false);
    n_p.param<bool>("ground_truth", ground_truth, false);
    n_p.param<bool>("debug_mode", debug_mode, false);

  

    std::vector<double> affine_list;
    if (ground_truth)
    {
        n_p.param<std::string>("ground_truth_odom_topic", ground_truth_odom_topic, "ground_truth");
        n_p.param<std::string>("ground_truth_com_topic", ground_truth_com_topic, "ground_truth_com");
        n_p.getParam("T_B_GT", affine_list);
        T_B_GT.setIdentity();
        if (affine_list.size() == 16)
        {
            T_B_GT(0, 0) = affine_list[0];
            T_B_GT(0, 1) = affine_list[1];
            T_B_GT(0, 2) = affine_list[2];
            T_B_GT(0, 3) = affine_list[3];
            T_B_GT(1, 0) = affine_list[4];
            T_B_GT(1, 1) = affine_list[5];
            T_B_GT(1, 2) = affine_list[6];
            T_B_GT(1, 3) = affine_list[7];
            T_B_GT(2, 0) = affine_list[8];
            T_B_GT(2, 1) = affine_list[9];
            T_B_GT(2, 2) = affine_list[10];
            T_B_GT(2, 3) = affine_list[11];
            T_B_GT(3, 0) = affine_list[12];
            T_B_GT(3, 1) = affine_list[13];
            T_B_GT(3, 2) = affine_list[14];
            T_B_GT(3, 3) = affine_list[15];
        }
        q_B_GT = Quaterniond(T_B_GT.linear());

    }
    
    T_B_P.setIdentity();
    if (!useLegOdom)
    {
        n_p.getParam("T_B_P", affine_list);
        if (affine_list.size() == 16)
        {
            T_B_P(0, 0) = affine_list[0];
            T_B_P(0, 1) = affine_list[1];
            T_B_P(0, 2) = affine_list[2];
            T_B_P(0, 3) = affine_list[3];
            T_B_P(1, 0) = affine_list[4];
            T_B_P(1, 1) = affine_list[5];
            T_B_P(1, 2) = affine_list[6];
            T_B_P(1, 3) = affine_list[7];
            T_B_P(2, 0) = affine_list[8];
            T_B_P(2, 1) = affine_list[9];
            T_B_P(2, 2) = affine_list[10];
            T_B_P(2, 3) = affine_list[11];
            T_B_P(3, 0) = affine_list[12];
            T_B_P(3, 1) = affine_list[13];
            T_B_P(3, 2) = affine_list[14];
            T_B_P(3, 3) = affine_list[15];
        }
        q_B_P = Quaterniond(T_B_P.linear());
    }


    T_B_A.setIdentity();
    n_p.getParam("T_B_A", affine_list);
    if (affine_list.size() == 16)
    {
        T_B_A(0, 0) = affine_list[0];
        T_B_A(0, 1) = affine_list[1];
        T_B_A(0, 2) = affine_list[2];
        T_B_A(0, 3) = affine_list[3];
        T_B_A(1, 0) = affine_list[4];
        T_B_A(1, 1) = affine_list[5];
        T_B_A(1, 2) = affine_list[6];
        T_B_A(1, 3) = affine_list[7];
        T_B_A(2, 0) = affine_list[8];
        T_B_A(2, 1) = affine_list[9];
        T_B_A(2, 2) = affine_list[10];
        T_B_A(2, 3) = affine_list[11];
        T_B_A(3, 0) = affine_list[12];
        T_B_A(3, 1) = affine_list[13];
        T_B_A(3, 2) = affine_list[14];
        T_B_A(3, 3) = affine_list[15];
    }

    T_B_G.setIdentity();
    n_p.getParam("T_B_G", affine_list);
    if (affine_list.size() == 16)
    {
        T_B_G(0, 0) = affine_list[0];
        T_B_G(0, 1) = affine_list[1];
        T_B_G(0, 2) = affine_list[2];
        T_B_G(0, 3) = affine_list[3];
        T_B_G(1, 0) = affine_list[4];
        T_B_G(1, 1) = affine_list[5];
        T_B_G(1, 2) = affine_list[6];
        T_B_G(1, 3) = affine_list[7];
        T_B_G(2, 0) = affine_list[8];
        T_B_G(2, 1) = affine_list[9];
        T_B_G(2, 2) = affine_list[10];
        T_B_G(2, 3) = affine_list[11];
        T_B_G(3, 0) = affine_list[12];
        T_B_G(3, 1) = affine_list[13];
        T_B_G(3, 2) = affine_list[14];
        T_B_G(3, 3) = affine_list[15];
    }
    n_p.param<std::string>("odom_topic", odom_topic, "odom");
    n_p.param<std::string>("imu_topic", imu_topic, "imu");
    n_p.param<std::string>("joint_state_topic", joint_state_topic, "joint_states");
    n_p.param<double>("joint_noise_density", joint_noise_density, 0.03);
    n_p.param<std::string>("lfoot_force_torque_topic", lfsr_topic, "force_torque/left");
    n_p.param<std::string>("rfoot_force_torque_topic", rfsr_topic, "force_torque/right");

    T_FT_LL.setIdentity();
    n_p.getParam("T_FT_LL", affine_list);
    if (affine_list.size() == 16)
    {
        T_FT_LL(0, 0) = affine_list[0];
        T_FT_LL(0, 1) = affine_list[1];
        T_FT_LL(0, 2) = affine_list[2];
        T_FT_LL(0, 3) = affine_list[3];
        T_FT_LL(1, 0) = affine_list[4];
        T_FT_LL(1, 1) = affine_list[5];
        T_FT_LL(1, 2) = affine_list[6];
        T_FT_LL(1, 3) = affine_list[7];
        T_FT_LL(2, 0) = affine_list[8];
        T_FT_LL(2, 1) = affine_list[9];
        T_FT_LL(2, 2) = affine_list[10];
        T_FT_LL(2, 3) = affine_list[11];
        T_FT_LL(3, 0) = affine_list[12];
        T_FT_LL(3, 1) = affine_list[13];
        T_FT_LL(3, 2) = affine_list[14];
        T_FT_LL(3, 3) = affine_list[15];
    }
    p_FT_LL = Vector3d(T_FT_LL(0, 3), T_FT_LL(1, 3), T_FT_LL(2, 3));

    T_FT_RL.setIdentity();
    n_p.getParam("T_FT_RL", affine_list);
    if (affine_list.size() == 16)
    {
        T_FT_RL(0, 0) = affine_list[0];
        T_FT_RL(0, 1) = affine_list[1];
        T_FT_RL(0, 2) = affine_list[2];
        T_FT_RL(0, 3) = affine_list[3];
        T_FT_RL(1, 0) = affine_list[4];
        T_FT_RL(1, 1) = affine_list[5];
        T_FT_RL(1, 2) = affine_list[6];
        T_FT_RL(1, 3) = affine_list[7];
        T_FT_RL(2, 0) = affine_list[8];
        T_FT_RL(2, 1) = affine_list[9];
        T_FT_RL(2, 2) = affine_list[10];
        T_FT_RL(2, 3) = affine_list[11];
        T_FT_RL(3, 0) = affine_list[12];
        T_FT_RL(3, 1) = affine_list[13];
        T_FT_RL(3, 2) = affine_list[14];
        T_FT_RL(3, 3) = affine_list[15];
    }
    p_FT_RL = Vector3d(T_FT_RL(0, 3), T_FT_RL(1, 3), T_FT_RL(2, 3));

    n_p.param<bool>("comp_with", comp_with, false);
    comp_odom0_inc = false;
    if (comp_with)
        n_p.param<std::string>("comp_with_odom0_topic", comp_with_odom0_topic, "compare_with_odom0");

    n_p.param<bool>("estimateCoM", useCoMEKF, false);
    n_p.param<int>("medianWindow", medianWindow, 10);

    //Attitude Estimation for Leg Odometry
    n_p.param<bool>("useMahony", useMahony, true);
    if (useMahony)
    {
        //Mahony Filter for Attitude Estimation
        n_p.param<double>("Mahony_Kp", Kp, 0.25);
        n_p.param<double>("Mahony_Ki", Ki, 0.0);
        mh = new serow::Mahony(freq, Kp, Ki);
    }
    else
    {
        //Madgwick Filter for Attitude Estimation
        n_p.param<double>("Madgwick_gain", beta, 0.012f);
        mw = new serow::Madgwick(freq, beta);
    }
    n_p.param<double>("Tau0", Tau0, 0.5);
    n_p.param<double>("Tau1", Tau1, 0.01);
    n_p.param<double>("mass", mass, 5.14);
    n_p.param<double>("gravity", g, 9.81);
}

void humanoid_ekf::loadJointKFparams()
{
    ros::NodeHandle n_p("~");
    n_p.param<double>("joint_cutoff_freq", joint_cutoff_freq, 10.0);
}

void humanoid_ekf::loadIMUEKFparams()
{
    ros::NodeHandle n_p("~");
    n_p.param<double>("bias_ax", bias_ax, 0.0);
    n_p.param<double>("bias_ay", bias_ay, 0.0);
    n_p.param<double>("bias_az", bias_az, 0.0);
    n_p.param<double>("bias_gx", bias_gx, 0.0);
    n_p.param<double>("bias_gy", bias_gy, 0.0);
    n_p.param<double>("bias_gz", bias_gz, 0.0);

    if (!useInIMUEKF)
    {
        n_p.param<double>("accelerometer_noise_density", imuEKF->acc_qx, 0.001);
        n_p.param<double>("accelerometer_noise_density", imuEKF->acc_qy, 0.001);
        n_p.param<double>("accelerometer_noise_density", imuEKF->acc_qz, 0.001);

        n_p.param<double>("gyroscope_noise_density", imuEKF->gyr_qx, 0.0001);
        n_p.param<double>("gyroscope_noise_density", imuEKF->gyr_qy, 0.0001);
        n_p.param<double>("gyroscope_noise_density", imuEKF->gyr_qz, 0.0001);

        n_p.param<double>("accelerometer_bias_random_walk", imuEKF->accb_qx, 1.0e-04);
        n_p.param<double>("accelerometer_bias_random_walk", imuEKF->accb_qy, 1.0e-04);
        n_p.param<double>("accelerometer_bias_random_walk", imuEKF->accb_qz, 1.0e-04);
        n_p.param<double>("gyroscope_bias_random_walk", imuEKF->gyrb_qx, 1.0e-05);
        n_p.param<double>("gyroscope_bias_random_walk", imuEKF->gyrb_qy, 1.0e-05);
        n_p.param<double>("gyroscope_bias_random_walk", imuEKF->gyrb_qz, 1.0e-05);

        n_p.param<double>("odom_position_noise_density_x", imuEKF->odom_px, 1.0e-01);
        n_p.param<double>("odom_position_noise_density_y", imuEKF->odom_py, 1.0e-01);
        n_p.param<double>("odom_position_noise_density_z", imuEKF->odom_pz, 1.0e-01);
        n_p.param<double>("odom_orientation_noise_density", imuEKF->odom_ax, 1.0e-01);
        n_p.param<double>("odom_orientation_noise_density", imuEKF->odom_ay, 1.0e-01);
        n_p.param<double>("odom_orientation_noise_density", imuEKF->odom_az, 1.0e-01);

        n_p.param<double>("leg_odom_position_noise_density", imuEKF->leg_odom_px, 1.0e-01);
        n_p.param<double>("leg_odom_position_noise_density", imuEKF->leg_odom_py, 1.0e-01);
        n_p.param<double>("leg_odom_position_noise_density", imuEKF->leg_odom_pz, 1.0e-01);
        n_p.param<double>("leg_odom_orientation_noise_density", imuEKF->leg_odom_ax, 1.0e-01);
        n_p.param<double>("leg_odom_orientation_noise_density", imuEKF->leg_odom_ay, 1.0e-01);
        n_p.param<double>("leg_odom_orientation_noise_density", imuEKF->leg_odom_az, 1.0e-01);

        n_p.param<double>("velocity_noise_density_x", imuEKF->vel_px, 1.0e-01);
        n_p.param<double>("velocity_noise_density_y", imuEKF->vel_py, 1.0e-01);
        n_p.param<double>("velocity_noise_density_z", imuEKF->vel_pz, 1.0e-01);
        n_p.param<double>("gravity", imuEKF->ghat, 9.81);
        n_p.param<bool>("useEuler", imuEKF->useEuler, true);
        n_p.param<bool>("useOutlierDetection", useOutlierDetection, false);
        n_p.param<double>("mahalanobis_TH", imuEKF->mahalanobis_TH, -1.0);
    }
    else
    {
        n_p.param<double>("accelerometer_noise_density", imuInEKF->acc_qx, 0.001);
        n_p.param<double>("accelerometer_noise_density", imuInEKF->acc_qy, 0.001);
        n_p.param<double>("accelerometer_noise_density", imuInEKF->acc_qz, 0.001);

        n_p.param<double>("gyroscope_noise_density", imuInEKF->gyr_qx, 0.0001);
        n_p.param<double>("gyroscope_noise_density", imuInEKF->gyr_qy, 0.0001);
        n_p.param<double>("gyroscope_noise_density", imuInEKF->gyr_qz, 0.0001);

        n_p.param<double>("accelerometer_bias_random_walk", imuInEKF->accb_qx, 1.0e-04);
        n_p.param<double>("accelerometer_bias_random_walk", imuInEKF->accb_qy, 1.0e-04);
        n_p.param<double>("accelerometer_bias_random_walk", imuInEKF->accb_qz, 1.0e-04);
        n_p.param<double>("gyroscope_bias_random_walk", imuInEKF->gyrb_qx, 1.0e-05);
        n_p.param<double>("gyroscope_bias_random_walk", imuInEKF->gyrb_qy, 1.0e-05);
        n_p.param<double>("gyroscope_bias_random_walk", imuInEKF->gyrb_qz, 1.0e-05);

        n_p.param<double>("contact_random_walk", imuInEKF->foot_contactx, 1.0e-01);
        n_p.param<double>("contact_random_walk", imuInEKF->foot_contacty, 1.0e-01);
        n_p.param<double>("contact_random_walk", imuInEKF->foot_contactz, 1.0e-01);

        n_p.param<double>("leg_odom_position_noise_density", imuInEKF->leg_odom_px, 5.0e-02);
        n_p.param<double>("leg_odom_position_noise_density", imuInEKF->leg_odom_py, 5.0e-02);
        n_p.param<double>("leg_odom_position_noise_density", imuInEKF->leg_odom_pz, 5.0e-02);
        n_p.param<double>("leg_odom_orientation_noise_density", imuInEKF->leg_odom_ax, 1.0e-01);
        n_p.param<double>("leg_odom_orientation_noise_density", imuInEKF->leg_odom_ay, 1.0e-01);
        n_p.param<double>("leg_odom_orientation_noise_density", imuInEKF->leg_odom_az, 1.0e-01);

        n_p.param<double>("velocity_noise_density_x", imuInEKF->vel_px, 1.0e-01);
        n_p.param<double>("velocity_noise_density_y", imuInEKF->vel_py, 1.0e-01);
        n_p.param<double>("velocity_noise_density_z", imuInEKF->vel_pz, 1.0e-01);

        n_p.param<double>("odom_position_noise_density_x", imuInEKF->odom_px, 1.0e-01);
        n_p.param<double>("odom_position_noise_density_y", imuInEKF->odom_py, 1.0e-01);
        n_p.param<double>("odom_position_noise_density_z", imuInEKF->odom_pz, 1.0e-01);
        n_p.param<double>("odom_orientation_noise_density", imuInEKF->odom_ax, 1.0e-01);
        n_p.param<double>("odom_orientation_noise_density", imuInEKF->odom_ay, 1.0e-01);
        n_p.param<double>("odom_orientation_noise_density", imuInEKF->odom_az, 1.0e-01);
    }
}

void humanoid_ekf::loadCoMEKFparams()
{
    ros::NodeHandle n_p("~");
    n_p.param<double>("com_position_random_walk", nipmEKF->com_q, 1.0e-04);
    n_p.param<double>("com_velocity_random_walk", nipmEKF->comd_q, 1.0e-03);
    n_p.param<double>("external_force_random_walk", nipmEKF->fd_q, 1.0);
    n_p.param<double>("com_position_noise_density", nipmEKF->com_r, 1.0e-04);
    n_p.param<double>("com_acceleration_noise_density", nipmEKF->comdd_r, 5.0e-02);
    n_p.param<double>("Ixx", I_xx, 0.00000);
    n_p.param<double>("Iyy", I_yy, 0.00000);
    n_p.param<double>("Izz", I_zz, 0.00000);
    n_p.param<double>("bias_fx", bias_fx, 0.0);
    n_p.param<double>("bias_fy", bias_fy, 0.0);
    n_p.param<double>("bias_fz", bias_fz, 0.0);
    n_p.param<bool>("useGyroLPF", useGyroLPF, false);
    n_p.param<double>("gyro_cut_off_freq", gyro_fx, 7.0);
    n_p.param<double>("gyro_cut_off_freq", gyro_fy, 7.0);
    n_p.param<double>("gyro_cut_off_freq", gyro_fz, 7.0);
    n_p.param<int>("maWindow", maWindow, 10);
    n_p.param<bool>("useEuler", nipmEKF->useEuler, true);
}

humanoid_ekf::humanoid_ekf()
{
    useCoMEKF = true;
    useLegOdom = false;
    firstUpdate = false;
    firstOdom = false;
    odom_divergence = false;
}

humanoid_ekf::~humanoid_ekf()
{
    if (is_connected_)
        disconnect();
}

void humanoid_ekf::disconnect()
{
    if (!is_connected_)
        return;
    is_connected_ = false;
}

bool humanoid_ekf::connect(const ros::NodeHandle nh)
{
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
    if (useCoMEKF)
        loadCoMEKFparams();

    //Subscribe/Publish ROS Topics/Services
    subscribe();
    advertise();
    //ros::NodeHandle np("~")
    //dynamic_recfg_ = boost::make_shared< dynamic_reconfigure::Server<serow::VarianceControlConfig> >(np);
    //dynamic_reconfigure::Server<serow::VarianceControlConfig>::CallbackType cb = boost::bind(&humanoid_ekf::reconfigureCB, this, _1, _2);
    // dynamic_recfg_->setCallback(cb);
    is_connected_ = true;
    ros::Duration(1.0).sleep();
    ROS_INFO_STREAM("SERoW Initialized");
    return true;
}

bool humanoid_ekf::connected()
{
    return is_connected_;
}

void humanoid_ekf::subscribe()
{
    subscribeToIMU();
    subscribeToFSR();
    subscribeToJointState();

    if (!useLegOdom)
        subscribeToOdom();

    if (ground_truth)
    {
        subscribeToGroundTruth();
        subscribeToGroundTruthCoM();
    }


    if (comp_with)
        subscribeToCompOdom();
}

void humanoid_ekf::init()
{
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
    wbb = Vector3d::Zero();
    abb = Vector3d::Zero();
    omegabl = Vector3d::Zero();
    omegabr = Vector3d::Zero();
    vbl = Vector3d::Zero();
    vbr = Vector3d::Zero();
    Twl = Affine3d::Identity();
    Twr = Affine3d::Identity();
    Tbl = Affine3d::Identity();
    Tbr = Affine3d::Identity();
    vwl = Vector3d::Zero();
    vwr = Vector3d::Zero();
    vbln = Vector3d::Zero();
    vbrn = Vector3d::Zero();
    coplw = Vector3d::Zero();
    coprw = Vector3d::Zero();
    weightl = 0.000;
    weightr = 0.000;
    no_motion_residual = Vector3d::Zero();
    kinematicsInitialized = false;
    firstUpdate = true;
    firstGyrodot = true;
    firstContact = true;
    data_inc = false;
    //Initialize the IMU based EKF
    if (!useInIMUEKF)
    {
        imuEKF = new IMUEKF;
        imuEKF->init();
    }
    else
    {
        imuInEKF = new IMUinEKF;
        imuInEKF->init();
    }

    if (useCoMEKF)
    {
        if (useGyroLPF)
        {
            gyroLPF = new butterworthLPF *[3];
            for (unsigned int i = 0; i < 3; i++)
                gyroLPF[i] = new butterworthLPF();
        }
        else
        {
            gyroMAF = new MovingAverageFilter *[3];
            for (unsigned int i = 0; i < 3; i++)
                gyroMAF[i] = new MovingAverageFilter();
        }
        nipmEKF = new CoMEKF;
        nipmEKF->init();
    }


    no_motion_indicator = false;
    no_motion_it = 0;
    no_motion_threshold = 5e-4;
    no_motion_it_threshold = 500;
    outlier_count = 0;
    lmdf = MediatorNew(medianWindow);
    rmdf = MediatorNew(medianWindow);
    LLegForceFilt = Vector3d::Zero();
    RLegForceFilt = Vector3d::Zero();
    imuCalibrationCycles = 0;
}

/** Main Loop **/
void humanoid_ekf::filteringThread()
{
    static ros::Rate rate(freq); //ROS Node Loop Rate
    while (ros::ok())
    {
        if (joint_data.size() > 0 && base_imu_data.size() > 0 && LLeg_FT_data.size() > 0 && RLeg_FT_data.size() > 0)
        {
            joints(joint_data.pop());
            baseIMU(base_imu_data.pop());
            LLeg_FT(LLeg_FT_data.pop());
            RLeg_FT(RLeg_FT_data.pop());
            computeKinTFs();
            if (!calibrateIMU)
            {
                if (!useInIMUEKF)
                    estimateWithIMUEKF();
                else
                    estimateWithInIMUEKF();

                if (useCoMEKF)
                    estimateWithCoMEKF();

                data_inc = true;
            }
        }
        rate.sleep();
    }
    //De-allocation of Heap
    deAllocate();
}

void humanoid_ekf::estimateWithInIMUEKF()
{
    //Initialize the IMU EKF state
    if (imuInEKF->firstrun)
    {
        imuInEKF->setdt(1.0 / freq);
        imuInEKF->setBodyPos(Twb.translation());
        imuInEKF->setBodyOrientation(Twb.linear());
        imuInEKF->setLeftContact(Vector3d(dr->getLFootIMVPPosition()(0), dr->getLFootIMVPPosition()(1), 0.00));
        imuInEKF->setRightContact(Vector3d(dr->getRFootIMVPPosition()(0), dr->getRFootIMVPPosition()(1), 0.00));
        imuInEKF->setAccBias(Vector3d(bias_ax, bias_ay, bias_az));
        imuInEKF->setGyroBias(Vector3d(bias_gx, bias_gy, bias_gz));
        imuInEKF->firstrun = false;
    }

    //Compute the attitude and posture with the IMU-Kinematics Fusion
    //Predict with the IMU gyro and acceleration
    imuInEKF->predict(wbb, abb, dr->getRFootIMVPPosition(), dr->getLFootIMVPPosition(),
                      dr->getRFootIMVPOrientation(), dr->getLFootIMVPOrientation(),
                      cd->isRLegContact(), cd->isLLegContact());

    imuInEKF->updateWithContacts(dr->getRFootIMVPPosition(), dr->getLFootIMVPPosition(),
                                 JRQnJRt, JLQnJLt,
                                 cd->isRLegContact(), cd->isLLegContact(), cd->getRLegContactProb(), cd->getLLegContactProb());
    //imuInEKF->updateWithOrient(qwb);
    //imuInEKF->updateWithTwist(vwb, dr->getVelocityCovariance() +  cd->getDiffForce()/(m*g)*Matrix3d::Identity());
    //imuInEKF->updateWithTwistOrient(vwb,qwb);
    //imuInEKF->updateWithOdom(Twb.translation(),qwb);

    //Estimated TFs for Legs and Support foot
    Twl = imuInEKF->Tib * Tbl;
    Twr = imuInEKF->Tib * Tbr;
    qwl = Quaterniond(Twl.linear());
    qwr = Quaterniond(Twr.linear());
    Tws = imuInEKF->Tib * Tbs;
    qws = Quaterniond(Tws.linear());
}

void humanoid_ekf::estimateWithIMUEKF()
{
    //Initialize the IMU EKF state
    if (imuEKF->firstrun)
    {
        imuEKF->setdt(1.0 / freq);
        imuEKF->setBodyPos(Twb.translation());
        imuEKF->setBodyOrientation(Twb.linear());
        imuEKF->setAccBias(Vector3d(bias_ax, bias_ay, bias_az));
        imuEKF->setGyroBias(Vector3d(bias_gx, bias_gy, bias_gz));
        imuEKF->firstrun = false;
    }

    //Compute the attitude and posture with the IMU-Kinematics Fusion
    //Predict with the IMU gyro and acceleration
    imuEKF->predict(wbb, abb);
    
    //Check for no motion
    if (check_no_motion)
    {
        no_motion_residual = Twb.translation() - Twb_.translation();
        if (no_motion_residual.norm() < no_motion_threshold)
            no_motion_it++;
        else
        {
            no_motion_indicator = false;
            no_motion_it = 0;
        }
        if (no_motion_it > no_motion_it_threshold)
        {
            no_motion_indicator = true;
            no_motion_it = 0;
        }
        check_no_motion = false;
    }

    //Update EKF
    if (firstUpdate)
    {
        pos_update = Twb.translation();
        q_update = qwb;
        //First Update
        firstUpdate = false;
        imuEKF->updateWithLegOdom(pos_update, q_update);
    }
    else
    {
        //Update with the odometry
        if (no_motion_indicator || useLegOdom )
        {
            //Diff leg odom update
            pos_leg_update = Twb.translation() - Twb_.translation();
            q_leg_update = qwb * qwb_.inverse();
            pos_update += pos_leg_update;
            q_update *= q_leg_update;
            //imuEKF->updateWithTwistRotation(vwb, q_update);
            imuEKF->updateWithLegOdom(pos_update, q_update);
            //imuEKF->updateWithTwist(vwb);
        }
        else
        {

            if (odom_inc && !odom_divergence)
            {
                if (outlier_count < 3)
                {
                    pos_update_ = pos_update;
                    pos_update += T_B_P.linear() * Vector3d(odom_msg.pose.pose.position.x - odom_msg_.pose.pose.position.x,
                                                            odom_msg.pose.pose.position.y - odom_msg_.pose.pose.position.y, odom_msg.pose.pose.position.z - odom_msg_.pose.pose.position.z);

                    q_now = q_B_P * Quaterniond(odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x,
                                                odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z);

                    q_prev = q_B_P * Quaterniond(odom_msg_.pose.pose.orientation.w, odom_msg_.pose.pose.orientation.x,
                                                 odom_msg_.pose.pose.orientation.y, odom_msg_.pose.pose.orientation.z);

                    q_update_ = q_update;

                    q_update *= (q_now * q_prev.inverse());

                    odom_inc = false;
                    odom_msg_ = odom_msg;
                    
                    outlier = imuEKF->updateWithOdom(pos_update, q_update, useOutlierDetection);
                    if (outlier)
                    {
                        outlier_count++;
                        pos_update = pos_update_;
                        q_update = q_update_;
                    }
                    else
                    {
                        outlier_count = 0;
                    }
                }
                else
                {
                    odom_divergence = true;
                }
            }

            if (odom_divergence)
            {
                //std::cout<<"Odom divergence, updating only with leg odometry"<<std::endl;
                pos_update += pos_leg_update;
                q_update *= q_leg_update;
                imuEKF->updateWithTwistRotation(vwb, q_update);
                //imuEKF->updateWithTwist(vwb);

            }
        }
    }

    //Estimated TFs for Legs and Support foot
    Twl = imuEKF->Tib * Tbl;
    Twr = imuEKF->Tib * Tbr;
    qwl = Quaterniond(Twl.linear());
    qwr = Quaterniond(Twr.linear());
    Tws = imuEKF->Tib * Tbs;
    qws = Quaterniond(Tws.linear());
}

void humanoid_ekf::estimateWithCoMEKF()
{
    if (nipmEKF->firstrun)
    {
        nipmEKF->setdt(1.0 / freq);
        nipmEKF->setParams(mass, I_xx, I_yy, g);
        nipmEKF->setCoMPos(CoM_leg_odom);
        nipmEKF->setCoMExternalForce(Vector3d(bias_fx, bias_fy, bias_fz));
        nipmEKF->firstrun = false;
        if (useGyroLPF)
        {
            gyroLPF[0]->init("gyro X LPF", freq, gyro_fx);
            gyroLPF[1]->init("gyro Y LPF", freq, gyro_fy);
            gyroLPF[2]->init("gyro Z LPF", freq, gyro_fz);
        }
        else
        {
            for (unsigned int i = 0; i < 3; i++)
                gyroMAF[i]->setParams(maWindow);
        }
    }

    //Compute the COP in the Inertial Frame
    computeGlobalCOP(Twl, Twr);
    //Numerically compute the Gyro acceleration in the Inertial Frame and use a 3-Point Low-Pass filter
    filterGyrodot();
    DiagonalMatrix<double, 3> Inertia(I_xx, I_yy, I_zz);
    if (!useInIMUEKF)
    {
        nipmEKF->predict(COP_fsr, GRF_fsr, imuEKF->Rib * Inertia * Gyrodot);
        nipmEKF->update(
            imuEKF->acc + imuEKF->g,
            imuEKF->Tib * CoM_enc,
            imuEKF->gyro, Gyrodot);
    }
    else
    {
        nipmEKF->predict(COP_fsr, GRF_fsr, imuInEKF->Rib * Inertia * Gyrodot);
        nipmEKF->update(
            imuInEKF->acc + imuInEKF->g,
            imuInEKF->Tib * CoM_enc,
            imuInEKF->gyro, Gyrodot);
    }
}

void humanoid_ekf::computeKinTFs()
{

    //Update the Kinematic Structure
    rd->updateJointConfig(joint_state_pos_map, joint_state_vel_map, joint_noise_density);

    //Get the CoM w.r.t Body Frame
    CoM_enc = rd->comPosition();

    Tbl.translation() = rd->linkPosition(lfoot_frame);
    qbl = rd->linkOrientation(lfoot_frame);
    Tbl.linear() = qbl.toRotationMatrix();

    Tbr.translation() = rd->linkPosition(rfoot_frame);
    qbr = rd->linkOrientation(rfoot_frame);
    Tbr.linear() = qbr.toRotationMatrix();

    //TF Initialization
    if (!kinematicsInitialized)
    {
        Twl.translation() << Tbl.translation()(0), Tbl.translation()(1), 0.00;
        Twl.linear() = Tbl.linear();
        Twr.translation() << Tbr.translation()(0), Tbr.translation()(1), 0.00;
        Twr.linear() = Tbr.linear();
        dr = new serow::deadReckoning(Twl.translation(), Twr.translation(), Twl.linear(), Twr.linear(),
                                      mass, Tau0, Tau1, freq, g, p_FT_LL, p_FT_RL);
    }

    //Differential Kinematics with Pinnochio
    omegabl = rd->getAngularVelocity(lfoot_frame);
    omegabr = rd->getAngularVelocity(rfoot_frame);
    vbl = rd->getLinearVelocity(lfoot_frame);
    vbr = rd->getLinearVelocity(rfoot_frame);

    //Noises for update
    vbln = rd->getLinearVelocityNoise(lfoot_frame);
    vbrn = rd->getLinearVelocityNoise(rfoot_frame);
    JLQnJLt = vbln * vbln.transpose();
    JRQnJRt = vbrn * vbrn.transpose();

    if (useMahony)
    {
        qwb_ = qwb;
        qwb = Quaterniond(mh->getR());
        omegawb = mh->getGyro();
    }
    else
    {
        qwb_ = qwb;
        qwb = Quaterniond(mw->getR());
        omegawb = mw->getGyro();
    }

    Twb.linear() = qwb.toRotationMatrix();

    RLegForceFilt = Twb.linear() * Tbr.linear() * RLegForceFilt;
    LLegForceFilt = Twb.linear() * Tbl.linear() * LLegForceFilt;

    RLegGRF = Twb.linear() * Tbr.linear() * RLegGRF;
    LLegGRF = Twb.linear() * Tbl.linear() * LLegGRF;

    RLegGRT = Twb.linear() * Tbr.linear() * RLegGRT;
    LLegGRT = Twb.linear() * Tbl.linear() * LLegGRT;

    //Compute the GRF wrt world Frame, Forces are alread in the world frame
    GRF_fsr = RLegGRF;
    GRF_fsr += LLegGRF;

    if (firstContact)
    {
        cd = new serow::ContactDetection();
        if (useGEM)
        {
            cd->init(lfoot_frame, rfoot_frame, LosingContact, LosingContact, foot_polygon_xmin, foot_polygon_xmax,
                     foot_polygon_ymin, foot_polygon_ymax, lforce_sigma, rforce_sigma, lcop_sigma, rcop_sigma, VelocityThres,
                     lvnorm_sigma, rvnorm_sigma, ContactDetectionWithCOP, ContactDetectionWithKinematics, probabilisticContactThreshold, medianWindow);
        }
        else
        {
            cd->init(lfoot_frame, rfoot_frame, LegHighThres, LegLowThres, StrikingContact, VelocityThres, medianWindow);
        }

        firstContact = false;
    }

    if (useGEM)
    {
        cd->computeSupportFoot(LLegForceFilt(2), RLegForceFilt(2),
                               copl(0), copl(1), copr(0), copr(1),
                               vwl.norm(), vwr.norm());
    }
    else
    {
        cd->computeForceWeights(LLegForceFilt(2), RLegForceFilt(2));
        cd->SchmittTrigger(LLegForceFilt(2), RLegForceFilt(2));
    }

    Tbs = Tbl;
    qbs = qbl;
    support_leg = cd->getSupportLeg();
    if (support_leg.compare("RLeg") == 0)
    {
        Tbs = Tbr;
        qbs = qbr;
    }

    dr->computeDeadReckoning(Twb.linear(), Tbl.linear(), Tbr.linear(), omegawb, wbb,
                             Tbl.translation(), Tbr.translation(),
                             vbl, vbr, omegabl, omegabr,
                             LLegForceFilt(2), RLegForceFilt(2), LLegGRF, RLegGRF, LLegGRT, RLegGRT);

    //dr->computeDeadReckoningGEM(Twb.linear(),  Tbl.linear(),  Tbr.linear(),omegawb, Tbl.translation(),  Tbr.translation(), vbl,  vbr, omegabl,  omegabr,
    //                        cd->getLLegContactProb(),  cd->getRLegContactProb(), LLegGRF, RLegGRF, LLegGRT, RLegGRT);

    Twb_ = Twb;
    Twb.translation() = dr->getOdom();
    vwb = dr->getLinearVel();
    vwl = dr->getLFootLinearVel();
    vwr = dr->getRFootLinearVel();
    omegawl = dr->getLFootAngularVel();
    omegawr = dr->getRFootAngularVel();

    CoM_leg_odom = Twb * CoM_enc;
    check_no_motion = false;
    if (!kinematicsInitialized)
        kinematicsInitialized = true;
}

void humanoid_ekf::deAllocate()
{
    for (unsigned int i = 0; i < number_of_joints; i++)
        delete[] JointVF[i];
    delete[] JointVF;

    if (useCoMEKF)
    {
        delete nipmEKF;
        if (useGyroLPF)
        {
            for (unsigned int i = 0; i < 3; i++)
                delete[] gyroLPF[i];
            delete[] gyroLPF;
        }
        else
        {
            for (unsigned int i = 0; i < 3; i++)
                delete[] gyroMAF[i];
            delete[] gyroMAF;
        }
    }
    if (!useInIMUEKF)
        delete imuEKF;
    else
        delete imuInEKF;

    delete rd;
    delete mw;
    delete mh;
    delete dr;
    delete cd;
}

void humanoid_ekf::filterGyrodot()
{
    if (!firstGyrodot)
    {
        //Compute numerical derivative
        if (!useInIMUEKF)
        {
            Gyrodot = (imuEKF->gyro - Gyro_) * freq;
        }
        else
        {
            Gyrodot = (imuInEKF->gyro - Gyro_) * freq;
        }

        if (useGyroLPF)
        {
            Gyrodot(0) = gyroLPF[0]->filter(Gyrodot(0));
            Gyrodot(1) = gyroLPF[1]->filter(Gyrodot(1));
            Gyrodot(2) = gyroLPF[2]->filter(Gyrodot(2));
        }
        else
        {
            gyroMAF[0]->filter(Gyrodot(0));
            gyroMAF[1]->filter(Gyrodot(1));
            gyroMAF[2]->filter(Gyrodot(2));

            Gyrodot(0) = gyroMAF[0]->x;
            Gyrodot(1) = gyroMAF[1]->x;
            Gyrodot(2) = gyroMAF[2]->x;
        }
    }
    else
    {
        Gyrodot = Vector3d::Zero();
        firstGyrodot = false;
    }
    if (!useInIMUEKF)
        Gyro_ = imuEKF->gyro;
    else
        Gyro_ = imuInEKF->gyro;
}


void humanoid_ekf::computeGlobalCOP(Affine3d Twl_, Affine3d Twr_)
{

    //Compute the CoP wrt the Support Foot Frame
    coplw = Twl_ * copl;
    coprw = Twr_ * copr;

    if (weightl + weightr > 0.0)
    {
        COP_fsr = (weightl * coplw + weightr * coprw) / (weightl + weightr);
    }
    else
    {
        COP_fsr = Vector3d::Zero();
    }
}


void humanoid_ekf::advertise()
{

    SupportPose_pub = n.advertise<geometry_msgs::PoseStamped>("serow/support/pose", 1000);

    baseIMU_pub = n.advertise<sensor_msgs::Imu>("serow/base/acc", 1000);

    LLegOdom_pub = n.advertise<nav_msgs::Odometry>("serow/LLeg/odom", 1000);

    RLegOdom_pub = n.advertise<nav_msgs::Odometry>("serow/RLeg/odom", 1000);

    SupportLegId_pub = n.advertise<std_msgs::String>("serow/support/leg", 1000);

    baseOdom_pub = n.advertise<nav_msgs::Odometry>("serow/base/odom", 1000);



    CoMLegOdom_pub = n.advertise<nav_msgs::Odometry>("serow/CoM/leg_odom", 1000);

    if(computeJointVelocity)
        joint_pub = n.advertise<sensor_msgs::JointState>("serow/joint_states", 1000);

    if(useCoMEKF)
    {
        CoMOdom_pub = n.advertise<nav_msgs::Odometry>("serow/CoM/odom", 1000);
        ExternalWrench_pub = n.advertise<geometry_msgs::WrenchStamped>("serow/CoM/wrench", 1000);
        COP_pub = n.advertise<geometry_msgs::PointStamped>("serow/COP", 1000);
    }
    RLegWrench_pub = n.advertise<geometry_msgs::WrenchStamped>("serow/RLeg/wrench", 1000);

    LLegWrench_pub = n.advertise<geometry_msgs::WrenchStamped>("serow/LLeg/wrench", 1000);

    legOdom_pub = n.advertise<nav_msgs::Odometry>("serow/base/leg_odom", 1000);

    if(ground_truth)
    {
        ground_truth_com_pub = n.advertise<nav_msgs::Odometry>("serow/ground_truth/CoM/odom", 1000);

        ground_truth_odom_pub = n.advertise<nav_msgs::Odometry>("serow/ground_truth/base/odom", 1000);
    }

    if(debug_mode)
    {
        rel_LLegPose_pub = n.advertise<geometry_msgs::PoseStamped>("serow/rel_LLeg/pose", 1000);

        rel_RLegPose_pub = n.advertise<geometry_msgs::PoseStamped>("serow/rel_RLeg/pose", 1000);

        rel_CoMPose_pub = n.advertise<geometry_msgs::PoseStamped>("serow/rel_CoM/pose", 1000);   
    }

    if (comp_with)
        comp_odom0_pub = n.advertise<nav_msgs::Odometry>("serow/comp/base/odom0", 1000);
}

void humanoid_ekf::subscribeToJointState()
{

    joint_state_sub = n.subscribe(joint_state_topic, 1000, &humanoid_ekf::joint_stateCb, this);
    firstJointStates = true;
}

void humanoid_ekf::joint_stateCb(const sensor_msgs::JointState::ConstPtr &msg)
{
    joint_data.push(*msg);
    if (joint_data.size() > (int) freq/20)
        joint_data.pop();
}

void humanoid_ekf::joints(const sensor_msgs::JointState &msg)
{

    if (firstJointStates)
    {
        number_of_joints = msg.name.size();
        joint_state_vel.resize(number_of_joints);
        joint_state_pos.resize(number_of_joints);
        if (computeJointVelocity)
        {
            JointVF = new JointDF *[number_of_joints];
            for (unsigned int i = 0; i < number_of_joints; i++)
            {
                JointVF[i] = new JointDF();
                JointVF[i]->init(msg.name[i], freq, joint_cutoff_freq);
            }
        }
        firstJointStates = false;
    }

    if (computeJointVelocity)
    {
        for (unsigned int i = 0; i < msg.name.size(); i++)
        {
            joint_state_pos[i] = msg.position[i];
            joint_state_vel[i] = JointVF[i]->filter(msg.position[i]);
            joint_state_pos_map[msg.name[i]] = joint_state_pos[i];
            joint_state_vel_map[msg.name[i]] = joint_state_vel[i];
        }
    }
    else
    {
        for (unsigned int i = 0; i < msg.name.size(); i++)
        {
            joint_state_pos[i] = msg.position[i];
            joint_state_vel[i] = msg.velocity[i];
            joint_state_pos_map[msg.name[i]] = joint_state_pos[i];
            joint_state_vel_map[msg.name[i]] = joint_state_vel[i];
        }
    }
}

void humanoid_ekf::subscribeToOdom()
{

    odom_sub = n.subscribe(odom_topic, 1000, &humanoid_ekf::odomCb, this);
    firstOdom = true;
}

void humanoid_ekf::odomCb(const nav_msgs::Odometry::ConstPtr &msg)
{
    odom_msg = *msg;
    odom_inc = true;
    if (firstOdom)
    {
        odom_msg_ = odom_msg;
        firstOdom = false;
    }
}

void humanoid_ekf::subscribeToGroundTruth()
{
    ground_truth_odom_sub = n.subscribe(ground_truth_odom_topic, 1000, &humanoid_ekf::ground_truth_odomCb, this);
    firstGT = true;
}
void humanoid_ekf::ground_truth_odomCb(const nav_msgs::Odometry::ConstPtr &msg)
{
    ground_truth_odom_msg = *msg;
    if (kinematicsInitialized)
    {

        if (firstGT)
        {
            gt_odomq = qwb;
            gt_odom = Twb.translation();
            firstGT = false;
        }
        else
        {
            gt_odom += T_B_GT.linear() * Vector3d(ground_truth_odom_msg.pose.pose.position.x - ground_truth_odom_msg_.pose.pose.position.x,
                                                  ground_truth_odom_msg.pose.pose.position.y - ground_truth_odom_msg_.pose.pose.position.y,
                                                  ground_truth_odom_msg.pose.pose.position.z - ground_truth_odom_msg_.pose.pose.position.z);

            tempq = q_B_GT * Quaterniond(ground_truth_odom_msg.pose.pose.orientation.w, ground_truth_odom_msg.pose.pose.orientation.x, ground_truth_odom_msg.pose.pose.orientation.y, ground_truth_odom_msg.pose.pose.orientation.z);
            tempq_ = q_B_GT * Quaterniond(ground_truth_odom_msg_.pose.pose.orientation.w, ground_truth_odom_msg_.pose.pose.orientation.x, ground_truth_odom_msg_.pose.pose.orientation.y, ground_truth_odom_msg_.pose.pose.orientation.z);
            gt_odomq *= (tempq * tempq_.inverse());
        }

        ground_truth_odom_pub_msg.pose.pose.position.x = gt_odom(0);
        ground_truth_odom_pub_msg.pose.pose.position.y = gt_odom(1);
        ground_truth_odom_pub_msg.pose.pose.position.z = gt_odom(2);
        ground_truth_odom_pub_msg.pose.pose.orientation.w = gt_odomq.w();
        ground_truth_odom_pub_msg.pose.pose.orientation.x = gt_odomq.x();
        ground_truth_odom_pub_msg.pose.pose.orientation.y = gt_odomq.y();
        ground_truth_odom_pub_msg.pose.pose.orientation.z = gt_odomq.z();
    }
    ground_truth_odom_msg_ = ground_truth_odom_msg;
}

void humanoid_ekf::subscribeToGroundTruthCoM()
{
    ground_truth_com_sub = n.subscribe(ground_truth_com_topic, 1000, &humanoid_ekf::ground_truth_comCb, this);
    firstGTCoM = true;
}
void humanoid_ekf::ground_truth_comCb(const nav_msgs::Odometry::ConstPtr &msg)
{
    if (kinematicsInitialized)
    {
        ground_truth_com_odom_msg = *msg;
        temp = T_B_GT.linear() * Vector3d(ground_truth_com_odom_msg.pose.pose.position.x, ground_truth_com_odom_msg.pose.pose.position.y, ground_truth_com_odom_msg.pose.pose.position.z);
        tempq = q_B_GT * Quaterniond(ground_truth_com_odom_msg.pose.pose.orientation.w, ground_truth_com_odom_msg.pose.pose.orientation.x, ground_truth_com_odom_msg.pose.pose.orientation.y, ground_truth_com_odom_msg.pose.pose.orientation.z);
        if (firstGTCoM)
        {
            Vector3d tempCoMOffset = Twb * CoM_enc;
            offsetGTCoM = tempCoMOffset - temp;
            qoffsetGTCoM = qwb * tempq.inverse();
            firstGTCoM = false;
        }
        tempq = qoffsetGTCoM * tempq;
        temp = offsetGTCoM + temp;
        ground_truth_com_odom_msg.pose.pose.position.x = temp(0);
        ground_truth_com_odom_msg.pose.pose.position.y = temp(1);
        ground_truth_com_odom_msg.pose.pose.position.z = temp(2);

        ground_truth_com_odom_msg.pose.pose.orientation.w = tempq.w();
        ground_truth_com_odom_msg.pose.pose.orientation.x = tempq.x();
        ground_truth_com_odom_msg.pose.pose.orientation.y = tempq.y();
        ground_truth_com_odom_msg.pose.pose.orientation.z = tempq.z();
    }
}

void humanoid_ekf::subscribeToCompOdom()
{

    compodom0_sub = n.subscribe(comp_with_odom0_topic, 1000, &humanoid_ekf::compodom0Cb, this);
    firstCO = true;
}

void humanoid_ekf::compodom0Cb(const nav_msgs::Odometry::ConstPtr &msg)
{
    if (kinematicsInitialized)
    {
        comp_odom0_msg = *msg;
        temp = T_B_P.linear() * Vector3d(comp_odom0_msg.pose.pose.position.x, comp_odom0_msg.pose.pose.position.y, comp_odom0_msg.pose.pose.position.z);
        tempq = q_B_P * Quaterniond(comp_odom0_msg.pose.pose.orientation.w, comp_odom0_msg.pose.pose.orientation.x, comp_odom0_msg.pose.pose.orientation.y, comp_odom0_msg.pose.pose.orientation.z);
        if (firstCO)
        {
            qoffsetCO = qwb * tempq.inverse();
            offsetCO = Twb.translation() - temp;
            firstCO = false;
        }
        tempq = (qoffsetCO * tempq);
        temp = offsetCO + temp;

        comp_odom0_msg.pose.pose.position.x = temp(0);
        comp_odom0_msg.pose.pose.position.y = temp(1);
        comp_odom0_msg.pose.pose.position.z = temp(2);
        comp_odom0_msg.pose.pose.orientation.w = tempq.w();
        comp_odom0_msg.pose.pose.orientation.x = tempq.x();
        comp_odom0_msg.pose.pose.orientation.y = tempq.y();
        comp_odom0_msg.pose.pose.orientation.z = tempq.z();

        comp_odom0_inc = true;
    }
}


void humanoid_ekf::subscribeToIMU()
{
    imu_sub = n.subscribe(imu_topic, 1000, &humanoid_ekf::imuCb, this);
}
void humanoid_ekf::imuCb(const sensor_msgs::Imu::ConstPtr &msg)
{
    base_imu_data.push(*msg);
    if (base_imu_data.size() > (int) freq/20)
        base_imu_data.pop();
}
void humanoid_ekf::baseIMU(const sensor_msgs::Imu &msg)
{
    wbb = T_B_G.linear() * Vector3d(msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z);
    abb = T_B_A.linear() * Vector3d(msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z);
    
    if(useMahony)
    {
        mh->updateIMU(wbb, abb);
        Rwb = mh->getR();
    }
    else
    {
        mw->updateIMU(wbb, abb);
        Rwb = mw->getR();
    }

    if (imuCalibrationCycles < maxImuCalibrationCycles && calibrateIMU)
    {
        bias_g += wbb;
        bias_a += abb -  Rwb.transpose() * Vector3d(0,0,g); 
        imuCalibrationCycles++;
        return;
    }
    else if (calibrateIMU)
    {
        bias_ax = bias_a(0) / imuCalibrationCycles;
        bias_ay = bias_a(1) / imuCalibrationCycles;
        bias_az = bias_a(2) / imuCalibrationCycles;
        bias_gx = bias_g(0) / imuCalibrationCycles;
        bias_gy = bias_g(1) / imuCalibrationCycles;
        bias_gz = bias_g(2) / imuCalibrationCycles;
        calibrateIMU = false;
        std::cout << "Calibration finished at " << imuCalibrationCycles << std::endl;
        std::cout << "Gyro biases " << bias_gx << " " << bias_gy << " " << bias_gz << std::endl;
        std::cout << "Acc biases " << bias_ax << " " << bias_ay << " " << bias_az << std::endl;
    }
}

void humanoid_ekf::subscribeToFSR()
{
    //Left Foot Wrench
    lfsr_sub = n.subscribe(lfsr_topic, 1000, &humanoid_ekf::lfsrCb, this);
    //Right Foot Wrench
    rfsr_sub = n.subscribe(rfsr_topic, 1000, &humanoid_ekf::rfsrCb, this);
}

void humanoid_ekf::lfsrCb(const geometry_msgs::WrenchStamped::ConstPtr &msg)
{
    LLeg_FT_data.push(*msg);
    if (LLeg_FT_data.size() > (int) freq/20)
        LLeg_FT_data.pop();
}

void humanoid_ekf::LLeg_FT(const geometry_msgs::WrenchStamped &msg)
{
    LLegGRF(0) = msg.wrench.force.x;
    LLegGRF(1) = msg.wrench.force.y;
    LLegGRF(2) = msg.wrench.force.z;
    LLegGRT(0) = msg.wrench.torque.x;
    LLegGRT(1) = msg.wrench.torque.y;
    LLegGRT(2) = msg.wrench.torque.z;
    LLegGRF = T_FT_LL.linear() * LLegGRF;
    LLegGRT = T_FT_LL.linear() * LLegGRT;
    LLegForceFilt = LLegGRF;
    MediatorInsert(lmdf, LLegGRF(2));
    LLegForceFilt(2) = MediatorMedian(lmdf);

    weightl = 0;
    copl = Vector3d::Zero();
    if (LLegGRF(2) >= LosingContact)
    {
        copl(0) = -LLegGRT(1) / LLegGRF(2);
        copl(1) = LLegGRT(0) / LLegGRF(2);
        weightl = LLegGRF(2) / g;
    }
    else
    {
        copl = Vector3d::Zero();
        LLegGRF = Vector3d::Zero();
        LLegGRT = Vector3d::Zero();
        weightl = 0.0;
    }
}

void humanoid_ekf::rfsrCb(const geometry_msgs::WrenchStamped::ConstPtr &msg)
{
    RLeg_FT_data.push(*msg);
    if (RLeg_FT_data.size() > (int) freq/20)
        RLeg_FT_data.pop();

    //rfsr_msg = *msg;
}
void humanoid_ekf::RLeg_FT(const geometry_msgs::WrenchStamped &msg)
{
    RLegGRF(0) = msg.wrench.force.x;
    RLegGRF(1) = msg.wrench.force.y;
    RLegGRF(2) = msg.wrench.force.z;
    RLegGRT(0) = msg.wrench.torque.x;
    RLegGRT(1) = msg.wrench.torque.y;
    RLegGRT(2) = msg.wrench.torque.z;
    RLegGRF = T_FT_RL.linear() * RLegGRF;
    RLegGRT = T_FT_RL.linear() * RLegGRT;
    RLegForceFilt = RLegGRF;

    MediatorInsert(rmdf, RLegGRF(2));
    RLegForceFilt(2) = MediatorMedian(rmdf);
    copr = Vector3d::Zero();
    weightr = 0.0;
    if (RLegGRF(2) >= LosingContact)
    {
        copr(0) = -RLegGRT(1) / RLegGRF(2);
        copr(1) = RLegGRT(0) / RLegGRF(2);
        weightr = RLegGRF(2) / g;
    }
    else
    {
        copr = Vector3d::Zero();
        RLegGRF = Vector3d::Zero();
        RLegGRT = Vector3d::Zero();
        weightr = 0.0;
    }
}
void humanoid_ekf::publishBodyEstimates()
{

    if (!useInIMUEKF)
    {
        sensor_msgs::Imu tmp_imu_msg;
        tmp_imu_msg.header.stamp = ros::Time::now();
        tmp_imu_msg.header.frame_id = "odom";
        tmp_imu_msg.linear_acceleration.x = imuEKF->accX;
        tmp_imu_msg.linear_acceleration.y = imuEKF->accY;
        tmp_imu_msg.linear_acceleration.z = imuEKF->accZ;
        tmp_imu_msg.angular_velocity.x = imuEKF->gyroX;
        tmp_imu_msg.angular_velocity.y = imuEKF->gyroY;
        tmp_imu_msg.angular_velocity.z = imuEKF->gyroZ;
        baseIMU_pub.publish(tmp_imu_msg);


	    nav_msgs::Odometry tmp_odom_msg;
        tmp_odom_msg.child_frame_id = base_link_frame;
        tmp_odom_msg.header.stamp = ros::Time::now();
        tmp_odom_msg.header.frame_id = "odom";
        tmp_odom_msg.pose.pose.position.x = imuEKF->rX;
        tmp_odom_msg.pose.pose.position.y = imuEKF->rY;
        tmp_odom_msg.pose.pose.position.z = imuEKF->rZ;
        tmp_odom_msg.pose.pose.orientation.x = imuEKF->qib.x();
        tmp_odom_msg.pose.pose.orientation.y = imuEKF->qib.y();
        tmp_odom_msg.pose.pose.orientation.z = imuEKF->qib.z();
        tmp_odom_msg.pose.pose.orientation.w = imuEKF->qib.w();

        tmp_odom_msg.twist.twist.linear.x = imuEKF->velX;
        tmp_odom_msg.twist.twist.linear.y = imuEKF->velY;
        tmp_odom_msg.twist.twist.linear.z = imuEKF->velZ;
        tmp_odom_msg.twist.twist.angular.x = imuEKF->gyroX;
        tmp_odom_msg.twist.twist.angular.y = imuEKF->gyroY;
        tmp_odom_msg.twist.twist.angular.z = imuEKF->gyroZ;

        //for(int i=0;i<36;i++)
        //odom_est_msg.pose.covariance[i] = 0;
        baseOdom_pub.publish(tmp_odom_msg);
    }
    else
    {
        sensor_msgs::Imu tmp_imu_msg;
        tmp_imu_msg.header.stamp = ros::Time::now();
        tmp_imu_msg.header.frame_id = "odom";
        tmp_imu_msg.linear_acceleration.x = imuInEKF->accX;
        tmp_imu_msg.linear_acceleration.y = imuInEKF->accY;
        tmp_imu_msg.linear_acceleration.z = imuInEKF->accZ;

        tmp_imu_msg.angular_velocity.x = imuInEKF->gyroX;
        tmp_imu_msg.angular_velocity.y = imuInEKF->gyroY;
        tmp_imu_msg.angular_velocity.z = imuInEKF->gyroZ;
        baseIMU_pub.publish(tmp_imu_msg);

	    nav_msgs::Odometry tmp_odom_msg;
        tmp_odom_msg.child_frame_id = base_link_frame;
        tmp_odom_msg.header.stamp = ros::Time::now();
        tmp_odom_msg.header.frame_id = "odom";
        tmp_odom_msg.pose.pose.position.x = imuInEKF->rX;
        tmp_odom_msg.pose.pose.position.y = imuInEKF->rY;
        tmp_odom_msg.pose.pose.position.z = imuInEKF->rZ;
        tmp_odom_msg.pose.pose.orientation.x = imuInEKF->qib.x();
        tmp_odom_msg.pose.pose.orientation.y = imuInEKF->qib.y();
        tmp_odom_msg.pose.pose.orientation.z = imuInEKF->qib.z();
        tmp_odom_msg.pose.pose.orientation.w = imuInEKF->qib.w();

        tmp_odom_msg.twist.twist.linear.x = imuInEKF->velX;
        tmp_odom_msg.twist.twist.linear.y = imuInEKF->velY;
        tmp_odom_msg.twist.twist.linear.z = imuInEKF->velZ;
        tmp_odom_msg.twist.twist.angular.x = imuInEKF->gyroX;
        tmp_odom_msg.twist.twist.angular.y = imuInEKF->gyroY;
        tmp_odom_msg.twist.twist.angular.z = imuInEKF->gyroZ;
        baseOdom_pub.publish(tmp_odom_msg);
    }

	nav_msgs::Odometry tmp_odom_msg;
    tmp_odom_msg.child_frame_id = base_link_frame;
    tmp_odom_msg.header.stamp = ros::Time::now();
    tmp_odom_msg.header.frame_id = "odom";
    tmp_odom_msg.pose.pose.position.x = Twb.translation()(0);
    tmp_odom_msg.pose.pose.position.y = Twb.translation()(1);
    tmp_odom_msg.pose.pose.position.z = Twb.translation()(2);
    tmp_odom_msg.pose.pose.orientation.x = qwb.x();
    tmp_odom_msg.pose.pose.orientation.y = qwb.y();
    tmp_odom_msg.pose.pose.orientation.z = qwb.z();
    tmp_odom_msg.pose.pose.orientation.w = qwb.w();
    tmp_odom_msg.twist.twist.linear.x = vwb(0);
    tmp_odom_msg.twist.twist.linear.y = vwb(1);
    tmp_odom_msg.twist.twist.linear.z = vwb(2);
    tmp_odom_msg.twist.twist.angular.x = omegawb(0);
    tmp_odom_msg.twist.twist.angular.y = omegawb(1);
    tmp_odom_msg.twist.twist.angular.z = omegawb(2);
    legOdom_pub.publish(tmp_odom_msg);

    if (ground_truth)
    {
        ground_truth_com_odom_msg.child_frame_id = "CoM_frame";
        ground_truth_com_odom_msg.header.stamp = ros::Time::now();
        ground_truth_com_odom_msg.header.frame_id = "odom";
        ground_truth_com_pub.publish(ground_truth_com_odom_msg);

        ground_truth_odom_pub_msg.child_frame_id = base_link_frame;
        ground_truth_odom_pub_msg.header.stamp = ros::Time::now();
        ground_truth_odom_pub_msg.header.frame_id = "odom";
        ground_truth_odom_pub.publish(ground_truth_odom_pub_msg);
    }
    if (comp_odom0_inc)
    {
        comp_odom0_msg.child_frame_id = base_link_frame;
        comp_odom0_msg.header.stamp =  ros::Time::now();
        comp_odom0_msg.header.frame_id = "odom";
        comp_odom0_pub.publish(comp_odom0_msg);
        comp_odom0_inc = false;
    }
}

void humanoid_ekf::publishSupportEstimates()
{
	geometry_msgs::PoseStamped tmp_pose_msg;
    tmp_pose_msg.header.stamp = ros::Time::now();
    tmp_pose_msg.header.frame_id = "odom";
    tmp_pose_msg.pose.position.x = Tws.translation()(0);
    tmp_pose_msg.pose.position.y = Tws.translation()(1);
    tmp_pose_msg.pose.position.z = Tws.translation()(2);
    tmp_pose_msg.pose.orientation.x = qws.x();
    tmp_pose_msg.pose.orientation.y = qws.y();
    tmp_pose_msg.pose.orientation.z = qws.z();
    tmp_pose_msg.pose.orientation.w = qws.w();
    SupportPose_pub.publish(tmp_pose_msg);
}

void humanoid_ekf::publishLegEstimates()
{
	nav_msgs::Odometry tmp_LLeg_odom_msg;
    tmp_LLeg_odom_msg.child_frame_id = lfoot_frame;
    tmp_LLeg_odom_msg.header.stamp = ros::Time::now();
    tmp_LLeg_odom_msg.header.frame_id = "odom";
    tmp_LLeg_odom_msg.pose.pose.position.x = Twl.translation()(0);
    tmp_LLeg_odom_msg.pose.pose.position.y = Twl.translation()(1);
    tmp_LLeg_odom_msg.pose.pose.position.z = Twl.translation()(2);
    tmp_LLeg_odom_msg.pose.pose.orientation.x = qwl.x();
    tmp_LLeg_odom_msg.pose.pose.orientation.y = qwl.y();
    tmp_LLeg_odom_msg.pose.pose.orientation.z = qwl.z();
    tmp_LLeg_odom_msg.pose.pose.orientation.w = qwl.w();
    tmp_LLeg_odom_msg.twist.twist.linear.x = vwl(0);
    tmp_LLeg_odom_msg.twist.twist.linear.y = vwl(1);
    tmp_LLeg_odom_msg.twist.twist.linear.z = vwl(2);
    tmp_LLeg_odom_msg.twist.twist.angular.x = omegawl(0);
    tmp_LLeg_odom_msg.twist.twist.angular.y = omegawl(1);
    tmp_LLeg_odom_msg.twist.twist.angular.z = omegawl(2);
    LLegOdom_pub.publish(tmp_LLeg_odom_msg);

	nav_msgs::Odometry tmp_RLeg_odom_msg;
    tmp_RLeg_odom_msg.child_frame_id = rfoot_frame;
    tmp_RLeg_odom_msg.header.stamp = ros::Time::now();
    tmp_RLeg_odom_msg.header.frame_id = "odom";
    tmp_RLeg_odom_msg.pose.pose.position.x = Twr.translation()(0);
    tmp_RLeg_odom_msg.pose.pose.position.y = Twr.translation()(1);
    tmp_RLeg_odom_msg.pose.pose.position.z = Twr.translation()(2);
    tmp_RLeg_odom_msg.pose.pose.orientation.x = qwr.x();
    tmp_RLeg_odom_msg.pose.pose.orientation.y = qwr.y();
    tmp_RLeg_odom_msg.pose.pose.orientation.z = qwr.z();
    tmp_RLeg_odom_msg.pose.pose.orientation.w = qwr.w();
    tmp_RLeg_odom_msg.twist.twist.linear.x = vwr(0);
    tmp_RLeg_odom_msg.twist.twist.linear.y = vwr(1);
    tmp_RLeg_odom_msg.twist.twist.linear.z = vwr(2);
    tmp_RLeg_odom_msg.twist.twist.angular.x = omegawr(0);
    tmp_RLeg_odom_msg.twist.twist.angular.y = omegawr(1);
    tmp_RLeg_odom_msg.twist.twist.angular.z = omegawr(2);
    RLegOdom_pub.publish(tmp_RLeg_odom_msg);

    if (debug_mode)
    {
        geometry_msgs::PoseStamped tmp_LLeg_pose_msg;
        tmp_LLeg_pose_msg.pose.position.x = Tbl.translation()(0);
        tmp_LLeg_pose_msg.pose.position.y = Tbl.translation()(1);
        tmp_LLeg_pose_msg.pose.position.z = Tbl.translation()(2);
        tmp_LLeg_pose_msg.pose.orientation.x = qbl.x();
        tmp_LLeg_pose_msg.pose.orientation.y = qbl.y();
        tmp_LLeg_pose_msg.pose.orientation.z = qbl.z();
        tmp_LLeg_pose_msg.pose.orientation.w = qbl.w();
        tmp_LLeg_pose_msg.header.stamp = ros::Time::now();
        tmp_LLeg_pose_msg.header.frame_id = base_link_frame;
        rel_LLegPose_pub.publish(tmp_LLeg_pose_msg);

        geometry_msgs::PoseStamped tmp_RLeg_pose_msg;
        tmp_RLeg_pose_msg.pose.position.x = Tbr.translation()(0);
        tmp_RLeg_pose_msg.pose.position.y = Tbr.translation()(1);
        tmp_RLeg_pose_msg.pose.position.z = Tbr.translation()(2);
        tmp_RLeg_pose_msg.pose.orientation.x = qbr.x();
        tmp_RLeg_pose_msg.pose.orientation.y = qbr.y();
        tmp_RLeg_pose_msg.pose.orientation.z = qbr.z();
        tmp_RLeg_pose_msg.pose.orientation.w = qbr.w();
        tmp_RLeg_pose_msg.header.stamp = ros::Time::now();
        tmp_RLeg_pose_msg.header.frame_id = base_link_frame;
        rel_RLegPose_pub.publish(tmp_RLeg_pose_msg);
    }
}



void humanoid_ekf::publishJointEstimates()
{

	sensor_msgs::JointState  tmp_joint_msg;
    tmp_joint_msg.header.stamp = ros::Time::now();
    tmp_joint_msg.name.resize(number_of_joints);
    tmp_joint_msg.position.resize(number_of_joints);
    tmp_joint_msg.velocity.resize(number_of_joints);

    for (unsigned int i = 0; i < number_of_joints; i++)
    {
        tmp_joint_msg.position[i] = JointVF[i]->JointPosition;
        tmp_joint_msg.velocity[i] = JointVF[i]->JointVelocity;
        tmp_joint_msg.name[i] = JointVF[i]->JointName;
    }

    joint_pub.publish(tmp_joint_msg);
}



void humanoid_ekf::publishContact()
{
    std_msgs::String tmp_string_msg;
    tmp_string_msg.data = support_leg;
    SupportLegId_pub.publish(tmp_string_msg);
}

void humanoid_ekf::publishGRF()
{
    geometry_msgs::WrenchStamped tmp_LLeg_wrench_msg;
    tmp_LLeg_wrench_msg.wrench.force.x = LLegGRF(0);
    tmp_LLeg_wrench_msg.wrench.force.y = LLegGRF(1);
    tmp_LLeg_wrench_msg.wrench.force.z = LLegGRF(2);
    tmp_LLeg_wrench_msg.wrench.torque.x = LLegGRT(0);
    tmp_LLeg_wrench_msg.wrench.torque.y = LLegGRT(1);
    tmp_LLeg_wrench_msg.wrench.torque.z = LLegGRT(2);
    tmp_LLeg_wrench_msg.header.frame_id = lfoot_frame;
    tmp_LLeg_wrench_msg.header.stamp = ros::Time::now();
    LLegWrench_pub.publish(tmp_LLeg_wrench_msg);

    geometry_msgs::WrenchStamped tmp_RLeg_wrench_msg;
    tmp_RLeg_wrench_msg.wrench.force.x = RLegGRF(0);
    tmp_RLeg_wrench_msg.wrench.force.y = RLegGRF(1);
    tmp_RLeg_wrench_msg.wrench.force.z = RLegGRF(2);
    tmp_RLeg_wrench_msg.wrench.torque.x = RLegGRT(0);
    tmp_RLeg_wrench_msg.wrench.torque.y = RLegGRT(1);
    tmp_RLeg_wrench_msg.wrench.torque.z = RLegGRT(2);
    tmp_RLeg_wrench_msg.header.frame_id = rfoot_frame;
    tmp_RLeg_wrench_msg.header.stamp = ros::Time::now();
    RLegWrench_pub.publish(tmp_RLeg_wrench_msg);
}

void humanoid_ekf::publishCOP()
{
	geometry_msgs::PointStamped tmp_point_msg;
    tmp_point_msg.point.x = COP_fsr(0);
    tmp_point_msg.point.y = COP_fsr(1);
    tmp_point_msg.point.z = COP_fsr(2);
    tmp_point_msg.header.stamp = ros::Time::now();
    tmp_point_msg.header.frame_id = "odom";
    COP_pub.publish(tmp_point_msg);
}

void humanoid_ekf::publishCoMEstimates()
{

	nav_msgs::Odometry tmp_CoM_odom_msg;
    tmp_CoM_odom_msg.child_frame_id = "CoM_frame";
    tmp_CoM_odom_msg.header.stamp = ros::Time::now();
    tmp_CoM_odom_msg.header.frame_id = "odom";
    tmp_CoM_odom_msg.pose.pose.position.x = nipmEKF->comX;
    tmp_CoM_odom_msg.pose.pose.position.y = nipmEKF->comY;
    tmp_CoM_odom_msg.pose.pose.position.z = nipmEKF->comZ;
    tmp_CoM_odom_msg.twist.twist.linear.x = nipmEKF->velX;
    tmp_CoM_odom_msg.twist.twist.linear.y = nipmEKF->velY;
    tmp_CoM_odom_msg.twist.twist.linear.z = nipmEKF->velZ;
    //for(int i=0;i<36;i++)
    //odom_est_msg.pose.covariance[i] = 0;
    CoMOdom_pub.publish(tmp_CoM_odom_msg);

	nav_msgs::Odometry tmp_Leg_CoM_odom_msg;
    tmp_Leg_CoM_odom_msg.child_frame_id = "CoM_frame";
    tmp_Leg_CoM_odom_msg.header.stamp = ros::Time::now();
    tmp_Leg_CoM_odom_msg.header.frame_id = "odom";
    tmp_Leg_CoM_odom_msg.pose.pose.position.x = CoM_leg_odom(0);
    tmp_Leg_CoM_odom_msg.pose.pose.position.y = CoM_leg_odom(1);
    tmp_Leg_CoM_odom_msg.pose.pose.position.z = CoM_leg_odom(2);
    tmp_Leg_CoM_odom_msg.twist.twist.linear.x = 0;
    tmp_Leg_CoM_odom_msg.twist.twist.linear.y = 0;
    tmp_Leg_CoM_odom_msg.twist.twist.linear.z = 0;
    //for(int i=0;i<36;i++)
    //odom_est_msg.pose.covariance[i] = 0;
    CoMLegOdom_pub.publish(tmp_Leg_CoM_odom_msg);

    geometry_msgs::WrenchStamped tmp_wrench_msg;
    tmp_wrench_msg.header.frame_id = "odom";
    tmp_wrench_msg.header.stamp = ros::Time::now();
    tmp_wrench_msg.wrench.force.x = nipmEKF->fX;
    tmp_wrench_msg.wrench.force.y = nipmEKF->fY;
    tmp_wrench_msg.wrench.force.z = nipmEKF->fZ;
    ExternalWrench_pub.publish(tmp_wrench_msg);

    if (debug_mode)
    {
        geometry_msgs::PoseStamped tmp_pose_msg;
        tmp_pose_msg.pose.position.x = CoM_enc(0);
        tmp_pose_msg.pose.position.y = CoM_enc(1);
        tmp_pose_msg.pose.position.z = CoM_enc(2);
        tmp_pose_msg.header.stamp = ros::Time::now();
        tmp_pose_msg.header.frame_id = base_link_frame;
        rel_CoMPose_pub.publish(tmp_pose_msg);
    }
}








void humanoid_ekf::run()
{
    filtering_thread = std::thread([this] { this->filteringThread(); });
    output_thread = std::thread([this] { this->outputPublishThread(); });
    ros::spin();
}

void humanoid_ekf::outputPublishThread()
{

    ros::Rate rate(2.0*freq);
    while (ros::ok())
    {

        if (!data_inc)
            continue;
        output_lock.lock();
        //Publish Data
        if (computeJointVelocity)
            publishJointEstimates();
        
        publishBodyEstimates();
        publishLegEstimates();
        publishSupportEstimates();
        publishContact();
        publishGRF();

        if (useCoMEKF)
        {
            publishCoMEstimates();
            publishCOP();
        }
        data_inc = false;
        output_lock.unlock();

        rate.sleep();
    }
}