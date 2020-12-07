/*
 * SERoW - a complete state estimation scheme for Legged robots
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
#include <serow/quadruped_ekf.h>

void quadruped_ekf::loadparams()
{

    ros::NodeHandle n_p("~");
    // Load Server Parameters
    n_p.param<std::string>("modelname", modelname, "centauro.urdf");
    rd = new serow::robotDyn(modelname, false);

    n_p.param<std::string>("base_link", base_link_frame, "base_link");
    n_p.param<std::string>("LFfoot", LFfoot_frame, "lf_ankle");
    n_p.param<std::string>("LHfoot", LHfoot_frame, "lh_ankle");
    n_p.param<std::string>("RFfoot", RFfoot_frame, "rf_ankle");
    n_p.param<std::string>("RHfoot", RHfoot_frame, "rh_ankle");

    n_p.param<std::string>("LFfoot_force_torque_topic", LFfsr_topic, "force_torque/leftFront");
    n_p.param<std::string>("LHfoot_force_torque_topic", LHfsr_topic, "force_torque/leftHind");
    n_p.param<std::string>("RFfoot_force_torque_topic", RFfsr_topic, "force_torque/rightFront");
    n_p.param<std::string>("RHfoot_force_torque_topic", RHfsr_topic, "force_torque/rightHind");


    n_p.param<double>("imu_topic_freq", freq, 100.0);
    n_p.param<double>("ft_topic_freq", ft_freq, freq);
    n_p.param<double>("joint_topic_freq", joint_freq, 100.0);
    freq = min(min(freq,ft_freq),joint_freq);
    cout<<"Freq "<<freq<<endl;
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
        n_p.param<double>("LFforce_sigma", LFforce_sigma, 2.2734);
        n_p.param<double>("LHforce_sigma", LHforce_sigma, 5.6421);
        n_p.param<double>("RFforce_sigma", LFforce_sigma, 2.2734);
        n_p.param<double>("RHforce_sigma", LHforce_sigma, 5.6421);

        n_p.param<double>("LFcop_sigma", LFcop_sigma, 0.005);
        n_p.param<double>("LHcop_sigma", LHcop_sigma, 0.005);
        n_p.param<double>("RFcop_sigma", LFcop_sigma, 0.005);
        n_p.param<double>("RHcop_sigma", LHcop_sigma, 0.005);

        n_p.param<double>("LFvnorm_sigma", LFvnorm_sigma, 0.1);
        n_p.param<double>("LHvnorm_sigma", LHvnorm_sigma, 0.1);
        n_p.param<double>("RFvnorm_sigma", RFvnorm_sigma, 0.1);
        n_p.param<double>("RHvnorm_sigma", RHvnorm_sigma, 0.1);

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



    //GET TF from Left Front F/T sensor to Left Front Leg
    T_FT_LF.setIdentity();
    n_p.getParam("T_FT_LF", affine_list);
    if (affine_list.size() == 16)
    {
        T_FT_LF(0, 0) = affine_list[0];
        T_FT_LF(0, 1) = affine_list[1];
        T_FT_LF(0, 2) = affine_list[2];
        T_FT_LF(0, 3) = affine_list[3];
        T_FT_LF(1, 0) = affine_list[4];
        T_FT_LF(1, 1) = affine_list[5];
        T_FT_LF(1, 2) = affine_list[6];
        T_FT_LF(1, 3) = affine_list[7];
        T_FT_LF(2, 0) = affine_list[8];
        T_FT_LF(2, 1) = affine_list[9];
        T_FT_LF(2, 2) = affine_list[10];
        T_FT_LF(2, 3) = affine_list[11];
        T_FT_LF(3, 0) = affine_list[12];
        T_FT_LF(3, 1) = affine_list[13];
        T_FT_LF(3, 2) = affine_list[14];
        T_FT_LF(3, 3) = affine_list[15];
    }
    p_FT_LF = Vector3d(T_FT_LF(0, 3), T_FT_LF(1, 3), T_FT_LF(2, 3));

    //GET TF from Left Hind F/T sensor to Left Hind Leg
    T_FT_LH.setIdentity();

    n_p.getParam("T_FT_LH", affine_list);
    if (affine_list.size() == 16)
    {
        T_FT_LH(0, 0) = affine_list[0];
        T_FT_LH(0, 1) = affine_list[1];
        T_FT_LH(0, 2) = affine_list[2];
        T_FT_LH(0, 3) = affine_list[3];
        T_FT_LH(1, 0) = affine_list[4];
        T_FT_LH(1, 1) = affine_list[5];
        T_FT_LH(1, 2) = affine_list[6];
        T_FT_LH(1, 3) = affine_list[7];
        T_FT_LH(2, 0) = affine_list[8];
        T_FT_LH(2, 1) = affine_list[9];
        T_FT_LH(2, 2) = affine_list[10];
        T_FT_LH(2, 3) = affine_list[11];
        T_FT_LH(3, 0) = affine_list[12];
        T_FT_LH(3, 1) = affine_list[13];
        T_FT_LH(3, 2) = affine_list[14];
        T_FT_LH(3, 3) = affine_list[15];
    }
    p_FT_LH = Vector3d(T_FT_LH(0, 3), T_FT_LH(1, 3), T_FT_LH(2, 3));


    //GET TF from Right Front F/T sensor to Right Front Leg
    n_p.getParam("T_FT_RF", affine_list);
    if (affine_list.size() == 16)
    {
        T_FT_RF(0, 0) = affine_list[0];
        T_FT_RF(0, 1) = affine_list[1];
        T_FT_RF(0, 2) = affine_list[2];
        T_FT_RF(0, 3) = affine_list[3];
        T_FT_RF(1, 0) = affine_list[4];
        T_FT_RF(1, 1) = affine_list[5];
        T_FT_RF(1, 2) = affine_list[6];
        T_FT_RF(1, 3) = affine_list[7];
        T_FT_RF(2, 0) = affine_list[8];
        T_FT_RF(2, 1) = affine_list[9];
        T_FT_RF(2, 2) = affine_list[10];
        T_FT_RF(2, 3) = affine_list[11];
        T_FT_RF(3, 0) = affine_list[12];
        T_FT_RF(3, 1) = affine_list[13];
        T_FT_RF(3, 2) = affine_list[14];
        T_FT_RF(3, 3) = affine_list[15];
    }
    p_FT_RF = Vector3d(T_FT_RF(0, 3), T_FT_RF(1, 3), T_FT_RF(2, 3));

    //GET TF from Right Hind F/T sensor to Right Hind Leg
    n_p.getParam("T_FT_RH", affine_list);
    if (affine_list.size() == 16)
    {
        T_FT_RH(0, 0) = affine_list[0];
        T_FT_RH(0, 1) = affine_list[1];
        T_FT_RH(0, 2) = affine_list[2];
        T_FT_RH(0, 3) = affine_list[3];
        T_FT_RH(1, 0) = affine_list[4];
        T_FT_RH(1, 1) = affine_list[5];
        T_FT_RH(1, 2) = affine_list[6];
        T_FT_RH(1, 3) = affine_list[7];
        T_FT_RH(2, 0) = affine_list[8];
        T_FT_RH(2, 1) = affine_list[9];
        T_FT_RH(2, 2) = affine_list[10];
        T_FT_RH(2, 3) = affine_list[11];
        T_FT_RH(3, 0) = affine_list[12];
        T_FT_RH(3, 1) = affine_list[13];
        T_FT_RH(3, 2) = affine_list[14];
        T_FT_RH(3, 3) = affine_list[15];
    }
    p_FT_RH = Vector3d(T_FT_RH(0, 3), T_FT_RH(1, 3), T_FT_RH(2, 3));

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
    n_p.param<double>("mass", m, 5.14);
    n_p.param<double>("gravity", g, 9.81);

}

void quadruped_ekf::loadJointKFparams()
{
    ros::NodeHandle n_p("~");
    n_p.param<double>("joint_topic_freq", joint_freq, 100.0);
    n_p.param<double>("joint_cutoff_freq", joint_cutoff_freq, 10.0);
}

void quadruped_ekf::loadIMUEKFparams()
{
    ros::NodeHandle n_p("~");
    n_p.param<double>("bias_ax", bias_ax, 0.0);
    n_p.param<double>("bias_ay", bias_ay, 0.0);
    n_p.param<double>("bias_az", bias_az, 0.0);
    n_p.param<double>("bias_gx", bias_gx, 0.0);
    n_p.param<double>("bias_gy", bias_gy, 0.0);
    n_p.param<double>("bias_gz", bias_gz, 0.0);

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

void quadruped_ekf::loadCoMEKFparams()
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

quadruped_ekf::quadruped_ekf()
{
    useCoMEKF = true;
    useLegOdom = false;
    firstUpdate = false;
    firstOdom = false;
}

quadruped_ekf::~quadruped_ekf()
{
    if (is_connected_)
        disconnect();
}

void quadruped_ekf::disconnect()
{
    if (!is_connected_)
        return;

    is_connected_ = false;
}

bool quadruped_ekf::connect(const ros::NodeHandle nh)
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
    //
    //ros::NodeHandle np("~")
    //dynamic_recfg_ = boost::make_shared< dynamic_reconfigure::Server<serow::VarianceControlConfig> >(np);
    //dynamic_reconfigure::Server<serow::VarianceControlConfig>::CallbackType cb = boost::bind(&quadruped_ekf::reconfigureCB, this, _1, _2);
    // dynamic_recfg_->setCallback(cb);
    is_connected_ = true;

    ros::Duration(1.0).sleep();
    ROS_INFO_STREAM("SERoW Initialized");

    return true;
}


bool quadruped_ekf::connected()
{
    return is_connected_;
}

void quadruped_ekf::subscribe()
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

void quadruped_ekf::init()
{

    /** Initialize Variables **/
    //Kinematic TFs
    Tws  = Affine3d::Identity();
    Twb  = Affine3d::Identity();
    Twb_ = Affine3d::Identity();
    Tbs  = Affine3d::Identity();
    TwLF = Affine3d::Identity();
    TwLH = Affine3d::Identity();
    TwRF = Affine3d::Identity();
    TwRH = Affine3d::Identity();
    TbLF = Affine3d::Identity();
    TbLH = Affine3d::Identity();
    TbRF = Affine3d::Identity();
    TbRH = Affine3d::Identity();

    //GRF
    LFLegGRF = Vector3d::Zero();
    LHLegGRF = Vector3d::Zero();
    RFLegGRF = Vector3d::Zero();
    RHLegGRF = Vector3d::Zero();

    LFLegForceFilt =  Vector3d::Zero();
    RFLegForceFilt =  Vector3d::Zero();
    LHLegForceFilt =  Vector3d::Zero();
    RHLegForceFilt =  Vector3d::Zero();
    //GRT
    LFLegGRT = Vector3d::Zero();
    LHLegGRT = Vector3d::Zero();
    RFLegGRT = Vector3d::Zero();
    RHLegGRT = Vector3d::Zero();

    //LOCAL COP
    copLF = Vector3d::Zero();
    copLH = Vector3d::Zero();
    copRF = Vector3d::Zero();
    copRH = Vector3d::Zero();

    weightLF = 0.000;
    weightLH = 0.000;
    weightRF = 0.000;
    weightRH = 0.000;

    //GLOBAL COP
    copwLF = Vector3d::Zero();
    copwLH = Vector3d::Zero();
    copwRF = Vector3d::Zero();
    copwRH = Vector3d::Zero();

    //Global Base/Leg Twist
    omegawb = Vector3d::Zero();
    vwb = Vector3d::Zero();
    vwLF = Vector3d::Zero();
    vwLH = Vector3d::Zero();
    vwRF = Vector3d::Zero();
    vwRH = Vector3d::Zero();
    vbLFn = Vector3d::Zero();
    vbLHn = Vector3d::Zero();
    vbRFn = Vector3d::Zero();
    vbRHn = Vector3d::Zero();

    //Local Leg Twist
    omegabLF= Vector3d::Zero();
    omegabLH= Vector3d::Zero();
    omegabRF = Vector3d::Zero();
    omegabRH = Vector3d::Zero();
    vbLF = Vector3d::Zero();
    vbLH = Vector3d::Zero();
    vbRF = Vector3d::Zero();
    vbRH = Vector3d::Zero();

    kinematicsInitialized = false;
    firstUpdate = true;
    firstGyrodot = true;
    firstContact = true;
    data_inc = false;

    // Initialize the IMU based EKF
    imuInEKF = new IMUinEKFQuad;
    imuInEKF->init();
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

  

    odom_inc = false;
 
    LFmdf = MediatorNew(medianWindow);
    RFmdf = MediatorNew(medianWindow);
    LHmdf = MediatorNew(medianWindow);
    RHmdf = MediatorNew(medianWindow);

    imuCalibrationCycles = 0;
    bias_g = Vector3d::Zero();
    bias_a = Vector3d::Zero();


}

/** Main Loop **/
void quadruped_ekf::filteringThread()
{
    static ros::Rate rate(freq); //ROS Node Loop Rate
    while (ros::ok())
    {
        if (joint_data.size() > 0 && base_imu_data.size() > 0 && LFLeg_FT_data.size() > 0 && RFLeg_FT_data.size() > 0 && LHLeg_FT_data.size() > 0 && RHLeg_FT_data.size() > 0)
        {
            joints(joint_data.pop());
            baseIMU(base_imu_data.pop());
            LFLeg_FT(LFLeg_FT_data.pop());
            RFLeg_FT(RFLeg_FT_data.pop());
            LHLeg_FT(LHLeg_FT_data.pop());
            RHLeg_FT(RHLeg_FT_data.pop());
            computeKinTFs();
            if (!calibrateIMU)
            {
     
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


void quadruped_ekf::joints(const sensor_msgs::JointState &msg)
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


void quadruped_ekf::run()
{
    filtering_thread = std::thread([this] { this->filteringThread(); });
    output_thread = std::thread([this] { this->outputPublishThread(); });
    ros::spin();
}

/*
void quadruped_ekf::run()
{

    static ros::Rate rate(2.0*freq); //ROS Node Loop Rate
    while (ros::ok())
    {
        if (imu_inc)
        {
            predictWithImu = false;
            predictWithCoM = false;
            if (useMahony)
            {
                mh->updateIMU(T_B_G.linear() * (Vector3d(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z)),
                              T_B_A.linear() * (Vector3d(imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z)));
                Rwb = mh->getR();
            }
            else
            {
                mw->updateIMU(T_B_G.linear() * (Vector3d(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z)),
                              T_B_A.linear() * (Vector3d(imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z)));

                Rwb = mw->getR();
            }

            if(imuCalibrationCycles < maxImuCalibrationCycles && imuCalibrated)
            {
                bias_g += T_B_G.linear() * Vector3d(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z);
                bias_a += T_B_A.linear() * Vector3d(imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z) -  Rwb.transpose() * Vector3d(0,0,g); 

                imuCalibrationCycles++;
                continue;
            }
            else if(imuCalibrated)
            {
                bias_ax = bias_a(0)/imuCalibrationCycles;
                bias_ay = bias_a(1)/imuCalibrationCycles;
                bias_az = bias_a(2)/imuCalibrationCycles;
                bias_gx = bias_g(0)/imuCalibrationCycles;
                bias_gy = bias_g(1)/imuCalibrationCycles;
                bias_gz = bias_g(2)/imuCalibrationCycles;
                imuCalibrated = false;
                std::cout<<"Calibration finished at "<<imuCalibrationCycles<<std::endl;
                std::cout<<"Gyro biases "<<bias_gx<<" "<<bias_gy<<" "<<bias_gz<<std::endl;
                std::cout<<"Acc biases "<<bias_ax<<" "<<bias_ay<<" "<<bias_az<<std::endl;
            }
            //Compute the required transformation matrices (tfs) with Kinematics
            if (joint_inc)
                computeKinTFs();
        
            //Main Loop
            if (kinematicsInitialized)
            {
                
                estimateWithInIMUEKF();
                if (useCoMEKF)
                    estimateWithCoMEKF();

                //Publish Data
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
            }
        }
        ros::spinOnce();
        rate.sleep();
    }
    //De-allocation of Heap
    deAllocate();
}
*/
void quadruped_ekf::estimateWithInIMUEKF()
{
    //Initialize the IMU EKF state
    if (imuInEKF->firstrun == true)
    {
        imuInEKF->setdt(1.0 / freq);
        imuInEKF->setBodyPos(Twb.translation());
        imuInEKF->setBodyOrientation(Twb.linear());
        imuInEKF->setAccBias(Vector3d(bias_ax, bias_ay, bias_az));
        imuInEKF->setGyroBias(Vector3d(bias_gx, bias_gy, bias_gz));
        imuInEKF->setLeftFrontContact(Vector3d(dr->getLFFootIMVPPosition()(0), dr->getLFFootIMVPPosition()(1), 0.00));
        imuInEKF->setLeftHindContact(Vector3d(dr->getLHFootIMVPPosition()(0), dr->getLHFootIMVPPosition()(1), 0.00));
        imuInEKF->setRightFrontContact(Vector3d(dr->getRFFootIMVPPosition()(0), dr->getRFFootIMVPPosition()(1), 0.00));
        imuInEKF->setRightHindContact(Vector3d(dr->getRHFootIMVPPosition()(0), dr->getRHFootIMVPPosition()(1), 0.00));
        imuInEKF->firstrun = false;
    }


    // cout<<"Contact Status"<<endl;
    // cout<<"RF "<<cd->isRFLegContact()<<endl;
    // cout<<"RH "<< cd->isRHLegContact()<<endl;
    //  cout<<"LF "<< cd->isLFLegContact()<<endl;
    //   cout<<"LH "<<cd->isLHLegContact()<<endl;
    //Compute the attitude and posture with the IMU-Kinematics Fusion
    //Predict with the IMU gyro and acceleration

    imuInEKF->predict(wbb,
                      abb,
                      dr->getRFFootIMVPPosition(), dr->getRHFootIMVPPosition(), dr->getLFFootIMVPPosition(), dr->getLHFootIMVPPosition(),
                      dr->getRFFootIMVPOrientation(), dr->getRHFootIMVPOrientation(), dr->getLFFootIMVPOrientation(), dr->getLHFootIMVPOrientation(),
                      cd->isRFLegContact(), cd->isRHLegContact(), cd->isLFLegContact(), cd->isLHLegContact());

    imuInEKF->updateWithContacts(dr->getRFFootIMVPPosition(), dr->getRHFootIMVPPosition(), dr->getLFFootIMVPPosition(), dr->getLHFootIMVPPosition(),
                                 JRFQnJRFt + cd->getRFDiffForce() / (m * g) * Matrix3d::Identity(), JRHQnJRHt + cd->getRHDiffForce() / (m * g) * Matrix3d::Identity(),
                                 JLFQnJLFt + cd->getLFDiffForce() / (m * g) * Matrix3d::Identity(), JLHQnJLHt + cd->getLHDiffForce() / (m * g) * Matrix3d::Identity(),
                                 cd->isRFLegContact(), cd->isRHLegContact(), cd->isLFLegContact(), cd->isLHLegContact(),
                                 cd->getRFLegContactProb(), cd->getRHLegContactProb(), cd->getLFLegContactProb(), cd->getLHLegContactProb());

    //imuInEKF->updateWithOrient(qwb);
    //imuInEKF->updateWithTwist(vwb, dr->getVelocityCovariance() +  cd->getDiffForce()/(m*g)*Matrix3d::Identity());
    //imuInEKF->updateWithTwistOrient(vwb,qwb);
    //imuInEKF->updateWithOdom(Twb.translation(),qwb);

    //Estimated TFs for Legs and Support foot

    TwLF = imuInEKF->Tib * TbLF;
    TwLH = imuInEKF->Tib * TbLH;
    TwRF = imuInEKF->Tib * TbRF;
    TwRH = imuInEKF->Tib * TbRH;

    qwLF = Quaterniond(TwLF.linear());
    qwLH = Quaterniond(TwLH.linear());
    qwRF = Quaterniond(TwRF.linear());
    qwRH = Quaterniond(TwRH.linear());

    Tws = imuInEKF->Tib * Tbs;
    qws = Quaterniond(Tws.linear());
}

void quadruped_ekf::estimateWithCoMEKF()
{

   
        if (nipmEKF->firstrun)
        {
            nipmEKF->setdt(1.0 / ft_freq);
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
  
        computeGlobalCOP(TwLF, TwLH, TwRF, TwRH);
        //Numerically compute the Gyro acceleration in the Inertial Frame and use a 3-Point Low-Pass filter
        filterGyrodot();
        DiagonalMatrix<double, 3> Inertia(I_xx, I_yy, I_zz);
        nipmEKF->predict(COP_fsr, GRF_fsr, imuInEKF->Rib * Inertia * Gyrodot);
 
   

 
        nipmEKF->update(
            imuInEKF->acc + imuInEKF->g,
            imuInEKF->Tib * CoM_enc,
            imuInEKF->gyro, Gyrodot);
  
}

void quadruped_ekf::computeKinTFs()
{

    //Update the Kinematic Structure
    rd->updateJointConfig(joint_state_pos_map, joint_state_vel_map, joint_noise_density);

    //Get the CoM w.r.t Body Frame
    CoM_enc = rd->comPosition();

    mass = m;
    TbLF.translation() = rd->linkPosition(LFfoot_frame);
    qbLF = rd->linkOrientation(LFfoot_frame);
    TbLF.linear() = qbLF.toRotationMatrix();

    TbLH.translation() = rd->linkPosition(LHfoot_frame);
    qbLH = rd->linkOrientation(LHfoot_frame);
    TbLH.linear() = qbLH.toRotationMatrix();

    TbRF.translation() = rd->linkPosition(RFfoot_frame);
    qbRF = rd->linkOrientation(RFfoot_frame);
    TbRF.linear() = qbRF.toRotationMatrix();

    TbRH.translation() = rd->linkPosition(RHfoot_frame);
    qbRH = rd->linkOrientation(RHfoot_frame);
    TbRH.linear() = qbRH.toRotationMatrix();


    //TF Initialization
    if (!kinematicsInitialized)
    {
        TwLF.translation() << TbLF.translation()(0), TbLF.translation()(1), 0.00;
        TwLF.linear() = TbLF.linear();
        TwRF.translation() << TbRF.translation()(0), TbRF.translation()(1), 0.00;
        TwRF.linear() = TbRF.linear();

        TwLH.translation() << TbLH.translation()(0), TbLH.translation()(1), 0.00;
        TwLH.linear() = TbLH.linear();
        TwRH.translation() << TbRH.translation()(0), TbRH.translation()(1), 0.00;
        TwRH.linear() = TbRH.linear();


        dr = new serow::deadReckoningQuad(TwLF.translation(), TwLH.translation(), TwRF.translation(), TwRH.translation(), 
                                    TwLF.linear(), TwLH.linear(), TwRF.linear(), TwRH.linear(),
                                    mass, Tau0, Tau1, joint_freq, g, p_FT_LF, p_FT_LH,  p_FT_RF, p_FT_RH);
    }

    //Differential Kinematics with Pinnochio
    omegabLF = rd->getAngularVelocity(LFfoot_frame);
    omegabLH = rd->getAngularVelocity(LHfoot_frame);

    omegabRF = rd->getAngularVelocity(RFfoot_frame);
    omegabRH = rd->getAngularVelocity(RHfoot_frame);

    vbLF = rd->getLinearVelocity(LFfoot_frame);
    vbLH = rd->getLinearVelocity(LHfoot_frame);

    vbRF = rd->getLinearVelocity(RFfoot_frame);
    vbRH = rd->getLinearVelocity(RHfoot_frame);

    //Noises for update
    vbLFn = rd->getLinearVelocityNoise(LFfoot_frame);
    vbLHn = rd->getLinearVelocityNoise(LHfoot_frame);
    vbRFn = rd->getLinearVelocityNoise(RFfoot_frame);
    vbRHn = rd->getLinearVelocityNoise(RHfoot_frame);

    JLFQnJLFt = vbLFn * vbLFn.transpose();
    JLHQnJLHt = vbLHn * vbLHn.transpose();

    JRFQnJRFt = vbRFn * vbRFn.transpose();
    JRHQnJRHt = vbRHn * vbRHn.transpose();

    if(useMahony)
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

    RFLegForceFilt = Twb.linear() * TbRF.linear() * RFLegForceFilt;
    RHLegForceFilt = Twb.linear() * TbRH.linear() * RHLegForceFilt;
    LFLegForceFilt = Twb.linear() * TbLF.linear() * LFLegForceFilt;
    LHLegForceFilt = Twb.linear() * TbLH.linear() * LHLegForceFilt;

    RFLegGRF = Twb.linear() * TbRF.linear() * RFLegGRF;
    RHLegGRF = Twb.linear() * TbRH.linear() * RHLegGRF;
    LFLegGRF = Twb.linear() * TbLF.linear() * LFLegGRF;
    LHLegGRF = Twb.linear() * TbLH.linear() * LHLegGRF;

    RFLegGRT = Twb.linear() * TbRF.linear() * RFLegGRT;
    RHLegGRT = Twb.linear() * TbRH.linear() * RHLegGRT;
    LFLegGRT = Twb.linear() * TbLF.linear() * LFLegGRT;
    LHLegGRT = Twb.linear() * TbLH.linear() * LHLegGRT;

    //Compute the GRF wrt world Frame, Forces are alread in the world frame
    GRF_fsr = LFLegGRF;
    GRF_fsr += RFLegGRF;
    GRF_fsr += LHLegGRF;
    GRF_fsr += RHLegGRF;
    if (firstContact)
    {
        cd = new serow::ContactDetectionQuad();
        if (useGEM)
        {
            cd->init(LFfoot_frame, LHfoot_frame, RFfoot_frame, RHfoot_frame, LosingContact, LosingContact, LosingContact, LosingContact, foot_polygon_xmin, foot_polygon_xmax,
                     foot_polygon_ymin, foot_polygon_ymax, LFforce_sigma, LHforce_sigma, RFforce_sigma, RHforce_sigma, LFcop_sigma, LHcop_sigma,
                     RFcop_sigma, RHcop_sigma, VelocityThres, LFvnorm_sigma, LHvnorm_sigma, RFvnorm_sigma, RHvnorm_sigma, ContactDetectionWithCOP, ContactDetectionWithKinematics, probabilisticContactThreshold, medianWindow);
        }
        else
        {
            cd->init(LFfoot_frame, LHfoot_frame, RFfoot_frame, RHfoot_frame, LegHighThres, LegLowThres, StrikingContact, VelocityThres, mass, g, medianWindow);
        }

        firstContact = false;
    }
    if (useGEM)
    {
        cd->computeSupportFoot(LFLegForceFilt(2), LHLegForceFilt(2), RFLegForceFilt(2), RHLegForceFilt(2),
                               copLF(0), copLF(1), copLH(0), copLH(1), copRF(0), copRF(1), copRH(0), copRH(1),
                               vwLF.norm(), vwLH.norm(), vwRF.norm(), vwRH.norm());
    }
    else
    {   
        // cout<<"FORCES "<<endl;
        // cout<<LFLegGRF<<endl;
        // cout<<LHLegGRF<<endl;
        // cout<<RFLegGRF<<endl;
        // cout<<RHLegGRF<<endl;
        // cout<<"---"<<endl;
        cd->computeForceWeights(LFLegForceFilt(2), LHLegForceFilt(2), RFLegForceFilt(2), RHLegForceFilt(2));
        cd->SchmittTrigger(LFLegForceFilt(2), LHLegForceFilt(2), RFLegForceFilt(2), RHLegForceFilt(2));
    }

    Tbs = TbLF;
    qbs = qbLF;
    support_leg = cd->getSupportLeg();
    if (support_leg.compare("LHLeg") == 0)
    {
        Tbs = TbLH;
        qbs = qbLH;
    }
    else if (support_leg.compare("RFLeg") == 0)
    {
        Tbs = TbRF;
        qbs = qbRF;
    }
    else if (support_leg.compare("RHLeg") == 0)
    {
        Tbs = TbRH;
        qbs = qbRH;
    }

    // dr->computeDeadReckoningGEM(Twb.linear(), TbLF.linear(), TbLH.linear(), TbRF.linear(), TbRH.linear(),  omegawb, T_B_G.linear() * Vector3d(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z),
    //                            TbLF.translation(),  TbLH.translation(), TbRF.translation(),  TbRH.translation(),
    //                            vbLF, vbLH, vbRF, vbRH, omegabLF, omegabLH, omegabRF, omegabRH,
    //                            cd->getLFLegContactProb(), cd->getLHLegContactProb(),  cd->getRFLegContactProb(), cd->getRHLegContactProb(),
    //                            LFLegGRF, LHLegGRF, RFLegGRF, RHLegGRF, LFLegGRT, LHLegGRT, RFLegGRT, RHLegGRT);

    dr->computeDeadReckoning(Twb.linear(), TbLF.linear(), TbLH.linear(), TbRF.linear(), TbRH.linear(), omegawb, T_B_G.linear() * Vector3d(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z),
                             TbLF.translation(), TbLH.translation(), TbRF.translation(), TbRH.translation(),
                             vbLF, vbLH, vbRF, vbRH, omegabLF, omegabLH, omegabRF, omegabRH,
                             LFLegForceFilt(2), LHLegForceFilt(2), RFLegForceFilt(2), RHLegForceFilt(2), LFLegGRF, LHLegGRF, RFLegGRF, RHLegGRF, LFLegGRT, LHLegGRT, RFLegGRT, RHLegGRT);

    Twb_ = Twb;
    Twb.translation() = dr->getOdom();

    vwb = dr->getLinearVel();
    vwLF = dr->getLFFootLinearVel();
    vwLH = dr->getLHFootLinearVel();

    vwRF = dr->getRFFootLinearVel();
    vwRH = dr->getRHFootLinearVel();

    omegawLF = dr->getLFFootAngularVel();
    omegawLH = dr->getLHFootAngularVel();

    omegawRF = dr->getRFFootAngularVel();
    omegawRH = dr->getRHFootAngularVel();

    CoM_leg_odom = Twb * CoM_enc;
    check_no_motion = false;
    if (!kinematicsInitialized)
        kinematicsInitialized = true;
}

void quadruped_ekf::deAllocate()
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
    delete imuInEKF;
    delete rd;
    delete mw;
    delete mh;
    delete dr;
    delete cd;
}

void quadruped_ekf::filterGyrodot()
{
    if (!firstGyrodot)
    {
        //Compute numerical derivative
        Gyrodot = (imuInEKF->gyro - Gyro_) * freq;
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
    Gyro_ = imuInEKF->gyro;
}



void quadruped_ekf::outputPublishThread()
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


void quadruped_ekf::publishGRF()
{

    if (debug_mode)
    {
        LFLeg_est_msg.wrench.force.x = LFLegGRF(0);
        LFLeg_est_msg.wrench.force.y = LFLegGRF(1);
        LFLeg_est_msg.wrench.force.z = LFLegGRF(2);
        LFLeg_est_msg.wrench.torque.x = LFLegGRT(0);
        LFLeg_est_msg.wrench.torque.y = LFLegGRT(1);
        LFLeg_est_msg.wrench.torque.z = LFLegGRT(2);
        LFLeg_est_msg.header.frame_id = LFfoot_frame;
        LFLeg_est_msg.header.stamp = ros::Time::now();
        LFLeg_est_pub.publish(LFLeg_est_msg);

        LHLeg_est_msg.wrench.force.x = LHLegGRF(0);
        LHLeg_est_msg.wrench.force.y = LHLegGRF(1);
        LHLeg_est_msg.wrench.force.z = LHLegGRF(2);
        LHLeg_est_msg.wrench.torque.x = LHLegGRT(0);
        LHLeg_est_msg.wrench.torque.y = LHLegGRT(1);
        LHLeg_est_msg.wrench.torque.z = LHLegGRT(2);
        LHLeg_est_msg.header.frame_id = LHfoot_frame;
        LHLeg_est_msg.header.stamp = ros::Time::now();
        LHLeg_est_pub.publish(LHLeg_est_msg);


        RFLeg_est_msg.wrench.force.x = RFLegGRF(0);
        RFLeg_est_msg.wrench.force.y = RFLegGRF(1);
        RFLeg_est_msg.wrench.force.z = RFLegGRF(2);
        RFLeg_est_msg.wrench.torque.x = RFLegGRT(0);
        RFLeg_est_msg.wrench.torque.y = RFLegGRT(1);
        RFLeg_est_msg.wrench.torque.z = RFLegGRT(2);
        RFLeg_est_msg.header.frame_id = RFfoot_frame;
        RFLeg_est_msg.header.stamp = ros::Time::now();
        RFLeg_est_pub.publish(RFLeg_est_msg);

        RHLeg_est_msg.wrench.force.x = RHLegGRF(0);
        RHLeg_est_msg.wrench.force.y = RHLegGRF(1);
        RHLeg_est_msg.wrench.force.z = RHLegGRF(2);
        RHLeg_est_msg.wrench.torque.x = RHLegGRT(0);
        RHLeg_est_msg.wrench.torque.y = RHLegGRT(1);
        RHLeg_est_msg.wrench.torque.z = RHLegGRT(2);
        RHLeg_est_msg.header.frame_id = RHfoot_frame;
        RHLeg_est_msg.header.stamp = ros::Time::now();
        RHLeg_est_pub.publish(RHLeg_est_msg);

    }
}

void quadruped_ekf::computeGlobalCOP(Affine3d TwLF_, Affine3d TwLH_, Affine3d TwRF_, Affine3d TwRH_)
{

    copwLF = TwLF_ * copLF;
    copwRF = TwRF_ * copRF;
    copwLH = TwLH_ * copLH;
    copwRH = TwRH_ * copRH;

    //Compute the CoP wrt world Frame
    if (weightLF + weightRF + weightLH + weightRH > 0.0)
    {
        COP_fsr  = weightLF * copwLF;
        COP_fsr += weightRF * copwRF;
        COP_fsr += weightLH * copwLH;
        COP_fsr += weightRH * copwRH;
        COP_fsr  = COP_fsr /(weightLF + weightRF + weightLH + weightRH);
    }
    else
    {
        COP_fsr = Vector3d::Zero();
    }
}

void quadruped_ekf::publishCOP()
{
    COP_msg.point.x = COP_fsr(0);
    COP_msg.point.y = COP_fsr(1);
    COP_msg.point.z = COP_fsr(2);
    COP_msg.header.stamp = ros::Time::now();
    COP_msg.header.frame_id = "odom";
    COP_pub.publish(COP_msg);
}

void quadruped_ekf::publishCoMEstimates()
{
    CoM_odom_msg.child_frame_id = "CoM_frame";
    CoM_odom_msg.header.stamp = ros::Time::now();
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
    CoM_odom_msg.child_frame_id = "CoM_frame";
    CoM_odom_msg.header.stamp = ros::Time::now();
    CoM_odom_msg.header.frame_id = "odom";
    CoM_odom_msg.pose.pose.position.x = CoM_leg_odom(0);
    CoM_odom_msg.pose.pose.position.y = CoM_leg_odom(1);
    CoM_odom_msg.pose.pose.position.z = CoM_leg_odom(2);
    CoM_odom_msg.twist.twist.linear.x = 0;
    CoM_odom_msg.twist.twist.linear.y = 0;
    CoM_odom_msg.twist.twist.linear.z = 0;
    //for(int i=0;i<36;i++)
    //odom_est_msg.pose.covariance[i] = 0;
    CoM_leg_odom_pub.publish(CoM_odom_msg);

    external_force_filt_msg.header.frame_id = "odom";
    external_force_filt_msg.header.stamp = ros::Time::now();
    external_force_filt_msg.wrench.force.x = nipmEKF->fX;
    external_force_filt_msg.wrench.force.y = nipmEKF->fY;
    external_force_filt_msg.wrench.force.z = nipmEKF->fZ;
    external_force_filt_pub.publish(external_force_filt_msg);

    if (debug_mode)
    {
        temp_pose_msg.pose.position.x = CoM_enc(0);
        temp_pose_msg.pose.position.y = CoM_enc(1);
        temp_pose_msg.pose.position.z = CoM_enc(2);
        temp_pose_msg.header.stamp = ros::Time::now();
        temp_pose_msg.header.frame_id = base_link_frame;
        rel_CoMPose_pub.publish(temp_pose_msg);
    }
}

void quadruped_ekf::publishJointEstimates()
{

    joint_filt_msg.header.stamp = ros::Time::now();
    joint_filt_msg.name.resize(number_of_joints);
    joint_filt_msg.position.resize(number_of_joints);
    joint_filt_msg.velocity.resize(number_of_joints);

    for (unsigned int i = 0; i < number_of_joints; i++)
    {
        joint_filt_msg.position[i] = JointVF[i]->JointPosition;
        joint_filt_msg.velocity[i] = JointVF[i]->JointVelocity;
        joint_filt_msg.name[i] = JointVF[i]->JointName;
    }

    joint_filt_pub.publish(joint_filt_msg);
}

void quadruped_ekf::advertise()
{

    supportPose_est_pub = n.advertise<geometry_msgs::PoseStamped>(
        "serow/support/pose", 1000);

    bodyAcc_est_pub = n.advertise<sensor_msgs::Imu>(
        "serow/base/acc", 1000);

    LFLeg_odom_pub = n.advertise<nav_msgs::Odometry>(
        "serow/LFLeg/odom", 1000);

    RFLeg_odom_pub = n.advertise<nav_msgs::Odometry>(
        "serow/RFLeg/odom", 1000);


    LHLeg_odom_pub = n.advertise<nav_msgs::Odometry>(
        "serow/LHLeg/odom", 1000);

    RHLeg_odom_pub = n.advertise<nav_msgs::Odometry>(
        "serow/RHLeg/odom", 1000);

    support_leg_pub = n.advertise<std_msgs::String>("serow/support/leg", 1000);

    odom_est_pub = n.advertise<nav_msgs::Odometry>("serow/base/odom", 1000);
    CoM_leg_odom_pub = n.advertise<nav_msgs::Odometry>("serow/CoM/leg_odom", 1000);

    if(useCoMEKF)
    {    
        CoM_odom_pub = n.advertise<nav_msgs::Odometry>("serow/CoM/odom", 1000);
        COP_pub = n.advertise<geometry_msgs::PointStamped>("serow/COP", 1000);
        external_force_filt_pub = n.advertise<geometry_msgs::WrenchStamped>("serow/CoM/wrench", 1000);
    }



    if(computeJointVelocity)
        joint_filt_pub = n.advertise<sensor_msgs::JointState>("serow/joint_states", 1000);

    leg_odom_pub = n.advertise<nav_msgs::Odometry>("serow/base/leg_odom", 1000);

    if (ground_truth)
    {
        ground_truth_com_pub = n.advertise<nav_msgs::Odometry>("serow/ground_truth/CoM/odom", 1000);
        ground_truth_odom_pub = n.advertise<nav_msgs::Odometry>("serow/ground_truth/base/odom", 1000);
        ds_pub = n.advertise<std_msgs::Int32>("serow/is_in_ds", 1000);
    }

    if (debug_mode)
    {
        rel_LFLegPose_pub = n.advertise<geometry_msgs::PoseStamped>("serow/rel_LFLeg/pose", 1000);
        rel_RFLegPose_pub = n.advertise<geometry_msgs::PoseStamped>("serow/rel_RFLeg/pose", 1000);
        rel_LHLegPose_pub = n.advertise<geometry_msgs::PoseStamped>("serow/rel_LHLeg/pose", 1000);
        rel_RHLegPose_pub = n.advertise<geometry_msgs::PoseStamped>("serow/rel_RHLeg/pose", 1000);



        rel_CoMPose_pub = n.advertise<geometry_msgs::PoseStamped>("serow/rel_CoM/pose", 1000);
        RFLeg_est_pub = n.advertise<geometry_msgs::WrenchStamped>("serow/RFLeg/GRF", 1000);
        LFLeg_est_pub = n.advertise<geometry_msgs::WrenchStamped>("serow/LFLeg/GRF", 1000);
        RHLeg_est_pub = n.advertise<geometry_msgs::WrenchStamped>("serow/RHLeg/GRF", 1000);
        LHLeg_est_pub = n.advertise<geometry_msgs::WrenchStamped>("serow/LHLeg/GRF", 1000);        
    }
    if (comp_with)
        comp_odom0_pub = n.advertise<nav_msgs::Odometry>("serow/comp/base/odom0", 1000);
}

void quadruped_ekf::subscribeToJointState()
{

    joint_state_sub = n.subscribe(joint_state_topic, 1000, &quadruped_ekf::joint_stateCb, this, ros::TransportHints().tcpNoDelay());
    firstJointStates = true;
}

void quadruped_ekf::joint_stateCb(const sensor_msgs::JointState::ConstPtr &msg)
{
    joint_data.push(*msg);
    if (joint_data.size() > (int)freq / 20)
        joint_data.pop();
}

void quadruped_ekf::subscribeToOdom()
{

    odom_sub = n.subscribe(odom_topic, 1000, &quadruped_ekf::odomCb, this, ros::TransportHints().tcpNoDelay());
    firstOdom = true;
}

void quadruped_ekf::odomCb(const nav_msgs::Odometry::ConstPtr &msg)
{
    odom_msg = *msg;
    odom_inc = true;
    if (firstOdom)
    {
        odom_msg_ = odom_msg;
        firstOdom = false;
    }
}

void quadruped_ekf::subscribeToGroundTruth()
{
    ground_truth_odom_sub = n.subscribe(ground_truth_odom_topic, 1, &quadruped_ekf::ground_truth_odomCb, this, ros::TransportHints().tcpNoDelay());
    firstGT = true;
}
void quadruped_ekf::ground_truth_odomCb(const nav_msgs::Odometry::ConstPtr &msg)
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
              ground_truth_odom_msg.pose.pose.position.z -  ground_truth_odom_msg_.pose.pose.position.z);

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

void quadruped_ekf::subscribeToGroundTruthCoM()
{
    ground_truth_com_sub = n.subscribe(ground_truth_com_topic, 1000, &quadruped_ekf::ground_truth_comCb, this, ros::TransportHints().tcpNoDelay());
    firstGTCoM = true;
}
void quadruped_ekf::ground_truth_comCb(const nav_msgs::Odometry::ConstPtr &msg)
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

void quadruped_ekf::subscribeToCompOdom()
{

    compodom0_sub = n.subscribe(comp_with_odom0_topic, 1000, &quadruped_ekf::compodom0Cb, this, ros::TransportHints().tcpNoDelay());
    firstCO = true;
}

void quadruped_ekf::compodom0Cb(const nav_msgs::Odometry::ConstPtr &msg)
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


void quadruped_ekf::subscribeToIMU()
{
    imu_sub = n.subscribe(imu_topic, 1000, &quadruped_ekf::imuCb, this, ros::TransportHints().tcpNoDelay());
}
void quadruped_ekf::imuCb(const sensor_msgs::Imu::ConstPtr &msg)
{
    base_imu_data.push(*msg);
    if (base_imu_data.size() > (int) freq/20)
        base_imu_data.pop();
}
void quadruped_ekf::baseIMU(const sensor_msgs::Imu &msg)
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
        bias_g /= imuCalibrationCycles;
        bias_a /= imuCalibrationCycles;
        calibrateIMU = false;
        std::cout << "Calibration finished at " << imuCalibrationCycles << std::endl;
        std::cout << "Gyro biases " << bias_gx << " " << bias_gy << " " << bias_gz << std::endl;
        std::cout << "Acc biases " << bias_ax << " " << bias_ay << " " << bias_az << std::endl;
    }
}



void quadruped_ekf::subscribeToFSR()
{
    //Left Foot Wrench
    LFft_sub = n.subscribe(LFfsr_topic, 1000, &quadruped_ekf::LFfsrCb, this, ros::TransportHints().tcpNoDelay());
    //Right Foot Wrench
    RFft_sub = n.subscribe(RFfsr_topic, 1000, &quadruped_ekf::RFfsrCb, this, ros::TransportHints().tcpNoDelay());

    //Left Foot Wrench
    LHft_sub = n.subscribe(LHfsr_topic, 1000, &quadruped_ekf::LHfsrCb, this, ros::TransportHints().tcpNoDelay());
    //Right Foot Wrench
    RHft_sub = n.subscribe(RHfsr_topic, 1000, &quadruped_ekf::RHfsrCb, this, ros::TransportHints().tcpNoDelay());


}

void quadruped_ekf::LFfsrCb(const geometry_msgs::WrenchStamped::ConstPtr &msg)
{
    LFLeg_FT_data.push(*msg);
    if (LFLeg_FT_data.size() > (int) freq/20)
        LFLeg_FT_data.pop();
}
void quadruped_ekf::LFLeg_FT(const geometry_msgs::WrenchStamped &msg)
{
    LFLegGRF(0) = msg.wrench.force.x;
    LFLegGRF(1) = msg.wrench.force.y;
    LFLegGRF(2) = msg.wrench.force.z;
    LFLegGRT(0) = msg.wrench.torque.x;
    LFLegGRT(1) = msg.wrench.torque.y;
    LFLegGRT(2) = msg.wrench.torque.z;
    LFLegGRF = T_FT_LF.linear() * LFLegGRF;
    LFLegGRT = T_FT_LF.linear() * LFLegGRT;
    LFLegForceFilt = LFLegGRF;
    MediatorInsert(LFmdf, LFLegGRF(2));
    LFLegForceFilt(2) = MediatorMedian(LFmdf);

    copLF = Vector3d::Zero();
    if (LFLegGRF(2) >= LosingContact)
    {
        copLF(0) = -LFLegGRT(1) / LFLegGRF(2);
        copLF(1) = LFLegGRT(0) / LFLegGRF(2);
    }
    weightLF = LFLegGRF(2) / g;
}
void quadruped_ekf::RFfsrCb(const geometry_msgs::WrenchStamped::ConstPtr &msg)
{
   
    RFLeg_FT_data.push(*msg);
    if (RFLeg_FT_data.size() > (int) freq/20)
        RFLeg_FT_data.pop();
}
void quadruped_ekf::RFLeg_FT(const geometry_msgs::WrenchStamped &msg)
{
    RFLegGRF(0) = msg.wrench.force.x;
    RFLegGRF(1) = msg.wrench.force.y;
    RFLegGRF(2) = msg.wrench.force.z;
    RFLegGRT(0) = msg.wrench.torque.x;
    RFLegGRT(1) = msg.wrench.torque.y;
    RFLegGRT(2) = msg.wrench.torque.z;
    RFLegGRF = T_FT_RF.linear() * RFLegGRF;
    RFLegGRT = T_FT_RF.linear() * RFLegGRT;
    RFLegForceFilt = RFLegGRF;

    MediatorInsert(RFmdf, RFLegGRF(2));
    RFLegForceFilt(2) = MediatorMedian(RFmdf);

    copRF = Vector3d::Zero();
    if (RFLegGRF(2) >= LosingContact)
    {
        copRF(0) = -RFLegGRT(1) / RFLegGRF(2);
        copRF(1) =  RFLegGRT(0) / RFLegGRF(2);
    }
    weightRF = RFLegGRF(2) / g;
}

void quadruped_ekf::LHfsrCb(const geometry_msgs::WrenchStamped::ConstPtr &msg)
{
    LHLeg_FT_data.push(*msg);
    if (LHLeg_FT_data.size() > (int) freq/20)
        LHLeg_FT_data.pop();
}
void quadruped_ekf::LHLeg_FT(const geometry_msgs::WrenchStamped &msg)
{
    LHLegGRF(0) = msg.wrench.force.x;
    LHLegGRF(1) = msg.wrench.force.y;
    LHLegGRF(2) = msg.wrench.force.z;
    LHLegGRT(0) = msg.wrench.torque.x;
    LHLegGRT(1) = msg.wrench.torque.y;
    LHLegGRT(2) = msg.wrench.torque.z;
    LHLegGRF = T_FT_LH.linear() * LHLegGRF;
    LHLegGRT = T_FT_LH.linear() * LHLegGRT;
    LHLegForceFilt = LHLegGRF;

    MediatorInsert(LHmdf, LHLegGRF(2));
    LHLegForceFilt(2) = MediatorMedian(LHmdf);

     
    copLH = Vector3d::Zero();
    if (LHLegGRF(2) >= LosingContact)
    {
        copLH(0) = -LHLegGRT(1) / LHLegGRF(2);
        copLH(1) = LHLegGRT(0) / LHLegGRF(2);
    }
    weightLH = LHLegGRF(2) / g;
}

void quadruped_ekf::RHfsrCb(const geometry_msgs::WrenchStamped::ConstPtr &msg)
{
    RHLeg_FT_data.push(*msg);
    if (RHLeg_FT_data.size() > (int) freq/20)
        RHLeg_FT_data.pop();
}
void quadruped_ekf::RHLeg_FT(const geometry_msgs::WrenchStamped &msg)
{
    RHLegGRF(0) = msg.wrench.force.x;
    RHLegGRF(1) = msg.wrench.force.y;
    RHLegGRF(2) = msg.wrench.force.z;
    RHLegGRT(0) = msg.wrench.torque.x;
    RHLegGRT(1) = msg.wrench.torque.y;
    RHLegGRT(2) = msg.wrench.torque.z;
    RHLegGRF = T_FT_RH.linear() * RHLegGRF;
    RHLegGRT = T_FT_RH.linear() * RHLegGRT;
    RHLegForceFilt = RHLegGRF;

    MediatorInsert(RHmdf, RHLegGRF(2));
    RHLegForceFilt(2) = MediatorMedian(RHmdf);

    copRH = Vector3d::Zero();
    if (RHLegGRF(2) >= LosingContact)
    {
        copRH(0) = -RHLegGRT(1) / RHLegGRF(2);
        copRH(1) =  RHLegGRT(0) / RHLegGRF(2);
    }
    weightRH = RHLegGRF(2) / g;
}

void quadruped_ekf::publishBodyEstimates()
{
        bodyAcc_est_msg.header.stamp = ros::Time::now();
        bodyAcc_est_msg.header.frame_id = "odom";
        bodyAcc_est_msg.linear_acceleration.x = imuInEKF->accX;
        bodyAcc_est_msg.linear_acceleration.y = imuInEKF->accY;
        bodyAcc_est_msg.linear_acceleration.z = imuInEKF->accZ;

        bodyAcc_est_msg.angular_velocity.x = imuInEKF->gyroX;
        bodyAcc_est_msg.angular_velocity.y = imuInEKF->gyroY;
        bodyAcc_est_msg.angular_velocity.z = imuInEKF->gyroZ;
        bodyAcc_est_pub.publish(bodyAcc_est_msg);

        odom_est_msg.child_frame_id = base_link_frame;
        odom_est_msg.header.stamp = ros::Time::now();
        odom_est_msg.header.frame_id = "odom";
        odom_est_msg.pose.pose.position.x = imuInEKF->rX;
        odom_est_msg.pose.pose.position.y = imuInEKF->rY;
        odom_est_msg.pose.pose.position.z = imuInEKF->rZ;
        odom_est_msg.pose.pose.orientation.x = imuInEKF->qib.x();
        odom_est_msg.pose.pose.orientation.y = imuInEKF->qib.y();
        odom_est_msg.pose.pose.orientation.z = imuInEKF->qib.z();
        odom_est_msg.pose.pose.orientation.w = imuInEKF->qib.w();

        odom_est_msg.twist.twist.linear.x = imuInEKF->velX;
        odom_est_msg.twist.twist.linear.y = imuInEKF->velY;
        odom_est_msg.twist.twist.linear.z = imuInEKF->velZ;
        odom_est_msg.twist.twist.angular.x = imuInEKF->gyroX;
        odom_est_msg.twist.twist.angular.y = imuInEKF->gyroY;
        odom_est_msg.twist.twist.angular.z = imuInEKF->gyroZ;
        odom_est_pub.publish(odom_est_msg);
    

    leg_odom_msg.child_frame_id = base_link_frame;
    leg_odom_msg.header.stamp = ros::Time::now();
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
        ds_pub.publish(is_in_ds_msg);
    }
    comp_odom0_msg.header = odom_est_msg.header;
    comp_odom0_pub.publish(comp_odom0_msg);
    comp_odom0_inc = false;
}

void quadruped_ekf::publishSupportEstimates()
{
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

void quadruped_ekf::publishLegEstimates()
{

    LFLeg_odom_msg.child_frame_id = LFfoot_frame;
    LFLeg_odom_msg.header.stamp = ros::Time::now();
    LFLeg_odom_msg.header.frame_id = "odom";
    LFLeg_odom_msg.pose.pose.position.x = TwLF.translation()(0);
    LFLeg_odom_msg.pose.pose.position.y = TwLF.translation()(1);
    LFLeg_odom_msg.pose.pose.position.z = TwLF.translation()(2);
    LFLeg_odom_msg.pose.pose.orientation.x = qwLF.x();
    LFLeg_odom_msg.pose.pose.orientation.y = qwLF.y();
    LFLeg_odom_msg.pose.pose.orientation.z = qwLF.z();
    LFLeg_odom_msg.pose.pose.orientation.w = qwLF.w();
    LFLeg_odom_msg.twist.twist.linear.x = vwLF(0);
    LFLeg_odom_msg.twist.twist.linear.y = vwLF(1);
    LFLeg_odom_msg.twist.twist.linear.z = vwLF(2);
    LFLeg_odom_msg.twist.twist.angular.x = omegawLF(0);
    LFLeg_odom_msg.twist.twist.angular.y = omegawLF(1);
    LFLeg_odom_msg.twist.twist.angular.z = omegawLF(2);
    LFLeg_odom_pub.publish(LFLeg_odom_msg);



    LHLeg_odom_msg.child_frame_id = LHfoot_frame;
    LHLeg_odom_msg.header.stamp = ros::Time::now();
    LHLeg_odom_msg.header.frame_id = "odom";
    LHLeg_odom_msg.pose.pose.position.x = TwLH.translation()(0);
    LHLeg_odom_msg.pose.pose.position.y = TwLH.translation()(1);
    LHLeg_odom_msg.pose.pose.position.z = TwLH.translation()(2);
    LHLeg_odom_msg.pose.pose.orientation.x = qwLH.x();
    LHLeg_odom_msg.pose.pose.orientation.y = qwLH.y();
    LHLeg_odom_msg.pose.pose.orientation.z = qwLH.z();
    LHLeg_odom_msg.pose.pose.orientation.w = qwLH.w();
    LHLeg_odom_msg.twist.twist.linear.x = vwLH(0);
    LHLeg_odom_msg.twist.twist.linear.y = vwLH(1);
    LHLeg_odom_msg.twist.twist.linear.z = vwLH(2);
    LHLeg_odom_msg.twist.twist.angular.x = omegawLH(0);
    LHLeg_odom_msg.twist.twist.angular.y = omegawLH(1);
    LHLeg_odom_msg.twist.twist.angular.z = omegawLH(2);
    LHLeg_odom_pub.publish(LHLeg_odom_msg);

    RFLeg_odom_msg.child_frame_id = RFfoot_frame;
    RFLeg_odom_msg.header.stamp = ros::Time::now();
    RFLeg_odom_msg.header.frame_id = "odom";
    RFLeg_odom_msg.pose.pose.position.x = TwRF.translation()(0);
    RFLeg_odom_msg.pose.pose.position.y = TwRF.translation()(1);
    RFLeg_odom_msg.pose.pose.position.z = TwRF.translation()(2);
    RFLeg_odom_msg.pose.pose.orientation.x = qwRF.x();
    RFLeg_odom_msg.pose.pose.orientation.y = qwRF.y();
    RFLeg_odom_msg.pose.pose.orientation.z = qwRF.z();
    RFLeg_odom_msg.pose.pose.orientation.w = qwRF.w();
    RFLeg_odom_msg.twist.twist.linear.x = vwRF(0);
    RFLeg_odom_msg.twist.twist.linear.y = vwRF(1);
    RFLeg_odom_msg.twist.twist.linear.z = vwRF(2);
    RFLeg_odom_msg.twist.twist.angular.x = omegawRF(0);
    RFLeg_odom_msg.twist.twist.angular.y = omegawRF(1);
    RFLeg_odom_msg.twist.twist.angular.z = omegawRF(2);
    RFLeg_odom_pub.publish(RFLeg_odom_msg);



    RHLeg_odom_msg.child_frame_id = RHfoot_frame;
    RHLeg_odom_msg.header.stamp = ros::Time::now();
    RHLeg_odom_msg.header.frame_id = "odom";
    RHLeg_odom_msg.pose.pose.position.x = TwRH.translation()(0);
    RHLeg_odom_msg.pose.pose.position.y = TwRH.translation()(1);
    RHLeg_odom_msg.pose.pose.position.z = TwRH.translation()(2);
    RHLeg_odom_msg.pose.pose.orientation.x = qwRH.x();
    RHLeg_odom_msg.pose.pose.orientation.y = qwRH.y();
    RHLeg_odom_msg.pose.pose.orientation.z = qwRH.z();
    RHLeg_odom_msg.pose.pose.orientation.w = qwRH.w();
    RHLeg_odom_msg.twist.twist.linear.x = vwRH(0);
    RHLeg_odom_msg.twist.twist.linear.y = vwRH(1);
    RHLeg_odom_msg.twist.twist.linear.z = vwRH(2);
    RHLeg_odom_msg.twist.twist.angular.x = omegawRH(0);
    RHLeg_odom_msg.twist.twist.angular.y = omegawRH(1);
    RHLeg_odom_msg.twist.twist.angular.z = omegawRH(2);
    RHLeg_odom_pub.publish(RHLeg_odom_msg);


    if (debug_mode)
    {
        temp_pose_msg.pose.position.x = TbLF.translation()(0);
        temp_pose_msg.pose.position.y = TbLF.translation()(1);
        temp_pose_msg.pose.position.z = TbLF.translation()(2);
        temp_pose_msg.pose.orientation.x = qbLF.x();
        temp_pose_msg.pose.orientation.y = qbLF.y();
        temp_pose_msg.pose.orientation.z = qbLF.z();
        temp_pose_msg.pose.orientation.w = qbLF.w();
        temp_pose_msg.header.stamp = ros::Time::now();
        temp_pose_msg.header.frame_id = base_link_frame;
        rel_LFLegPose_pub.publish(temp_pose_msg);


        temp_pose_msg.pose.position.x = TbLH.translation()(0);
        temp_pose_msg.pose.position.y = TbLH.translation()(1);
        temp_pose_msg.pose.position.z = TbLH.translation()(2);
        temp_pose_msg.pose.orientation.x = qbLH.x();
        temp_pose_msg.pose.orientation.y = qbLH.y();
        temp_pose_msg.pose.orientation.z = qbLH.z();
        temp_pose_msg.pose.orientation.w = qbLH.w();
        temp_pose_msg.header.stamp = ros::Time::now();
        temp_pose_msg.header.frame_id = base_link_frame;
        rel_LHLegPose_pub.publish(temp_pose_msg);


        temp_pose_msg.pose.position.x = TbRF.translation()(0);
        temp_pose_msg.pose.position.y = TbRF.translation()(1);
        temp_pose_msg.pose.position.z = TbRF.translation()(2);
        temp_pose_msg.pose.orientation.x = qbRF.x();
        temp_pose_msg.pose.orientation.y = qbRF.y();
        temp_pose_msg.pose.orientation.z = qbRF.z();
        temp_pose_msg.pose.orientation.w = qbRF.w();
        temp_pose_msg.header.stamp = ros::Time::now();
        temp_pose_msg.header.frame_id = base_link_frame;
        rel_RFLegPose_pub.publish(temp_pose_msg);

        temp_pose_msg.pose.position.x = TbRH.translation()(0);
        temp_pose_msg.pose.position.y = TbRH.translation()(1);
        temp_pose_msg.pose.position.z = TbRH.translation()(2);
        temp_pose_msg.pose.orientation.x = qbRH.x();
        temp_pose_msg.pose.orientation.y = qbRH.y();
        temp_pose_msg.pose.orientation.z = qbRH.z();
        temp_pose_msg.pose.orientation.w = qbRH.w();
        temp_pose_msg.header.stamp = ros::Time::now();
        temp_pose_msg.header.frame_id = base_link_frame;
        rel_RHLegPose_pub.publish(temp_pose_msg);

    }
}

void quadruped_ekf::publishContact()
{
    support_leg_msg.data = support_leg;
    support_leg_pub.publish(support_leg_msg);
}
