/* 
 * Copyright 2017-2021 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
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
 *		 nor the names of its contributors may be used to endorse or promote products derived from
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
 /**
 * @brief Base Estimator combining Inertial Measurement Unit (IMU) and Odometry Measuruements either 
 * from leg odometry or external odometry e.g Visual Odometry (VO) or Lidar Odometry (LO)
 * @author Stylianos Piperakis
 * @details State is  position in World frame 
 * velocity in  Base frame 
 * orientation of Body frame wrt the World frame
 * accelerometer bias in Base frame 
 * gyro bias in Base frame 
 * Measurements are: Base Position/Orinetation in World frame by Leg Odometry or Visual Odometry (VO) or Lidar Odometry (LO),
 * when VO/LO is considered the kinematically computed base velocity (Twist) is also employed for update.
 */
#ifndef  __ContactEKF_H__
#define  __ContactEKF_H__

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>       /* isnan, sqrt */
 

 
using namespace Eigen;

class ContactEKF {

private:
	/// Linearized state-input model
	Matrix<double, 27, 24> Lcf;
	/// Error Covariance, Linearized state transition model, Identity matrix, state uncertainty matrix
	Matrix<double, 27, 27> P, Af, Acf, If, Qff;
	/// Linearized Measurement model 
	Matrix<double, 6, 27> Hf, Hvf;
	Matrix<double, 12, 27> Hcf;
	Matrix<double, 3, 27> Hv;
	/// State-Input Uncertainty matrix
	Matrix<double, 24, 24> Qf;
	/// Kalman Gain
	Matrix<double, 27, 6> Kf;
	Matrix<double, 27, 3> Kv;
	Matrix<double, 27, 12> Kcf;
	/// Correction state vector
	Matrix<double, 27, 1> dxf;
	/// Update error covariance and Measurement noise
	Matrix<double, 6, 6> s, R;
	Matrix<double, 12, 12> sc, Rc;

	Matrix<double, 3, 3> sv, Rv;
	/// position, velocity , acc bias, gyro bias, bias corrected acc, bias corrected gyr vectors
	Vector3d r, v, omega, f, fhat, omegahat;
	/// Innovation vectors
	Vector3d zv;
	Matrix<double, 6, 1> z;
    Matrix<double, 12, 1> zf;

    

	/**
	 *  @brief computes the state transition matrix for linearized error state dynamics
     *  
	*/
	Matrix<double,27,27> computeTrans(Matrix<double,27,1> x_, Matrix<double,3,3> Rib_, Vector3d omega_, Vector3d f_);
	/**
	 *  @brief performs euler (first-order) discretization to the nonlinear state-space dynamics
     *  
	*/
	void euler(Vector3d omega_, Vector3d f_,Vector3d pbl_,Vector3d  pbr_,int  contactL_,int  contactR_);
	
	/**
	 *  @brief computes the discrete-time nonlinear state-space dynamics
     *  
	*/
    Matrix<double,27,1> computeDiscreteDyn(Matrix<double,27,1> x_, Matrix<double,3,3> Rib_, Vector3d omega_, Vector3d f_, Vector3d pbl_, Vector3d pbr_, int contactL_, int contactR_);
	/**
	 *  @brief computes the continuous-time nonlinear state-space dynamics
     *  
	*/
	Matrix<double,27,1> computeContinuousDyn(Matrix<double,27,1> x_, Matrix<double,3,3> Rib_, Vector3d omega_, Vector3d f_);

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
             
	/// Filter's State vector - with biases included
	Matrix<double, 27, 1> x;

	bool firstrun;
	bool useEuler;

	/// Gravity vector, gyro bias/acceleration bias vector,  estimated acceleration/velocity/position/RPY vector
    Vector3d g, bgyr, bacc, gyro, acc, vel, pos, angle;

	//Noise Stds
	double  acc_qx,acc_qy,acc_qz,gyr_qx,gyr_qy,gyr_qz,gyrb_qx,gyrb_qy,gyrb_qz,
	accb_qx,accb_qy,accb_qz, odom_px, odom_py, odom_pz, odom_ax, odom_ay,odom_az,
	vel_px, vel_py, vel_pz, leg_odom_px, leg_odom_py, leg_odom_pz, leg_odom_ax,
	leg_odom_ay, leg_odom_az, lp_qx, lp_qy, lp_qz, lo_qx, lo_qy, lo_qz, rp_qx, 
	rp_qy, rp_qz, ro_qx, ro_qy, ro_qz, lp_px, lp_py, lp_pz, lo_px, lo_py, lo_pz,
	rp_px, rp_py, rp_pz, ro_px, ro_py, ro_pz;

	double gyroX, gyroY, gyroZ, angleX, angleY, angleZ, bias_gx, bias_gy, bias_gz,
			bias_ax, bias_ay, bias_az, ghat;

	double accX, accY, accZ, velX, velY, velZ, rX, rY, rZ;

	/// Rotation of base/left leg/right left w.r.t the world frame reference
	Matrix3d Rib, Ril, Rir;
	/// Homogeneous transformation of base/left leg/right left w.r.t the world frame reference
	Affine3d  Tib, Til, Tir;
	/// Quaternion of base/left leg/right left w.r.t the world frame reference
	Quaterniond qib, qil, qir;

	/// Sampling time = 1.0/Sampling Frequency
	double dt;

	/**
	 *  @brief Constructor of the Base Estimator
     *  @details
     *   Initializes the gravity vector 
	*/
	ContactEKF();
	/** @fn void setdt(double dtt)
	 *  @brief sets the discretization of the Error State Kalman Filter (ESKF)
	 *  @param dtt sampling time in seconds
	 */
	void updateVars();
	/** @fn void setdt(double dtt)
	 *  @brief sets the discretization of the Error State Kalman Filter (ESKF)
	 *  @param dtt sampling time in seconds
	 */
	void setdt(double dtt) {
		dt = dtt;
	}
	/** @fn void setGyroBias(Vector3d bgyr_)
	 *  @brief initializes the angular velocity bias state of the Error State Kalman Filter (ESKF)
	 *  @param bgyr_ angular velocity bias in the base coordinates
	 */
	void setGyroBias(Vector3d bgyr_)
	{
		bgyr = bgyr_;
		x(9) = bgyr_(0);
		x(10) = bgyr_(1);
		x(11) = bgyr_(2);
		bias_gx = bgyr_(0);
		bias_gy = bgyr_(1);
		bias_gz = bgyr_(2);
	}
	/** @fn void setAccBias(Vector3d bacc_)
	 *  @brief initializes the acceleration bias state of the Error State Kalman Filter (ESKF)
	 *  @param bacc_ acceleration bias in the base coordinates
	 */
	void setAccBias(Vector3d bacc_)
	{
		bacc = bacc_;
		x(12) = bacc_(0);
		x(13) = bacc_(1);
		x(14) = bacc_(2);
		bias_ax = bacc_(0);
		bias_ay = bacc_(1);
		bias_az = bacc_(2);
	}
	/** @fn void setLeftContact bp)
	 *  @brief initializes the left leg contact state of the Error State Kalman Filter (ESKF)
	 *  @param bp Contact position of the left leg in the world frame
	 */
	void setLeftContact(Vector3d bp){

		x.segment<3>(15) = bp;
	}
	
	/** @fn void setLeftFootOrientation(Matrix3d Rot_)
	 *  @brief initializes the left leg orientation state of the Error State Kalman Filter (ESKF)
	 *  @param Rot_ Rotation matrix of the left leg in the world frame
	 */

	void setLeftFootOrientation(Matrix3d Rot_)
	{
		Ril = Rot_;
		qil = Quaterniond(Ril);
	}

	/** @fn void setRightContactVector3d bp)
	 *  @brief initializes the right leg contact state of the Error State Kalman Filter (ESKF)
	 *  @param bp Contact position of the right leg in the world frame
	 */
	void setRightContact(Vector3d bp){

		x.segment<3>(21) = bp;
	}
	

	/** @fn void setRightFootOrientation(Matrix3d Rot_)
	 *  @brief initializes the right leg orientation state of the Error State Kalman Filter (ESKF)
	 *  @param Rot_ Rotation matrix of the right leg in the world frame
	 */
	void setRightFootOrientation(Matrix3d Rot_)
	{
		Rir = Rot_;
		qir = Quaterniond(Rir);
	}

	/** @fn void setBodyPos(Vector3d bp)
	 *  @brief initializes the base position state of the Error State Kalman Filter (ESKF)
	 *  @param bp Position of the base in the world frame
	 */
	void setBodyPos(Vector3d bp) {
		x(6) = bp(0);
		x(7) = bp(1);
		x(8) = bp(2);
	}
	/** @fn void setBodyOrientation(Matrix3d Rot_)
	 *  @brief initializes the base rotation state of the Error State Kalman Filter (ESKF)
	 *  @param Rot_ Rotation of the base in the world frame
	 */
	void setBodyOrientation(Matrix3d Rot_){
		Rib = Rot_;
	}
	/** @fn void setBodyVel(Vector3d bv)
	 *  @brief initializes the base velocity state of the Error State Kalman Filter (ESKF)
	 *  @param bv linear velocity of the base in the base frame
	 */
    void setBodyVel(Vector3d bv)
    {
        x.segment<3>(0).noalias() =  bv;
    }
	/** @fn void predict(Vector3d omega_, Vector3d f_);
	 *  @brief realises the predict step of the Error State Kalman Filter (ESKF)
	 *  @param omega_ angular velocity of the base in the base frame
	 *  @param f_ linear acceleration of the base in the base frame
	 *  @param pbl_ 3D relative left leg position measurement in the base frame
	 *  @param pbr_ 3D relative right leg position measurement in the base frame
	 *  @param Rbl_ Rotation matrix of the left leg w.r.t the base frame 
	 *  @param Rbr_ Rotation matrix of the right leg w.r.t the base frame 
	 *  @param contactL_ Left Leg contact status (true/false)
	 *  @param contactR_ Right Leg contact status (true/false)
	 */
	void predict(Vector3d omega_, Vector3d f_ , Vector3d pbl_, Vector3d pbr_, Matrix3d Rbl_, Matrix3d Rbr_, int contactL_, int contactR_);
	/** @fn void updateWithOdom(Vector3d y, Quaterniond qy, bool useOutlierDetection);
	 *  @brief realises the pose update step of the Error State Kalman Filter (ESKF)
	 *  @param y 3D base position measurement in the world frame
	 *  @param qy orientation of the base w.r.t the world frame in quaternion
	 */


	/** @fn void updateWithLegContacts(Vector3d yl, Quaterniond qyl, Vector3d yr, Quaterniond qyr, Matrix3d JLQeJL, Matrix3d JRQeJR,  double probL_, double probR_, int contactL_, int contactR_);
	 *  @brief realises the pose update step of the Error State Kalman Filter (ESKF) with two (Right/Left) leg relative kinematic measurements
	 *  @param yl 3D relative left leg position measurement in the base frame
	 *  @param qyl  orientation of the left leg w.r.t the base frame in quaternion
	 *  @param yr 3D relative right leg position measurement in the base frame
	 *  @param qyr  orientation of the right leg w.r.t the base frame in quaternion
	 *  @param JLQeJL Relative left leg Jacobian
	 *  @param JRQeJR Relative right leg Jacobian
	 *  @param probL_ Left leg contact probability
	 *	@param probR_ Right leg contact probability
	 *  @param contactL_ Left Leg contact status (true/false)
	 *  @param contactR_ Right Leg contact status (true/false)
	 *  @note  Visual/Lidar odometry can be subject to outlier measurements that negatively affect the filter state 
	 */
	void updateWithLegContacts(Vector3d yl, Quaterniond qyl, Vector3d yr, Quaterniond qyr, Matrix3d JLQeJL, Matrix3d JRQeJR,  double probL_, double probR_, int contactL_, int contactR_);
	
	
	/** @fn void updateWithOdom(Vector3d y, Quaterniond qy);
	 *  @brief realises the pose update step of the Error State Kalman Filter (ESKF) with an Odometry measurement
	 *  @param y 3D base position measurement in the world frame
	 *  @param qy orientation of the base w.r.t the world frame in quaternion
	 *  @note  Visual/Lidar odometry can be subject to outlier measurements that negatively affect the filter state 
	 */
	void updateWithOdom(Vector3d y, Quaterniond qy);
	/** @fn void updateWithLegOdom(Vector3d y, Quaterniond qy);
	 *  @brief realises the pose update step of the Error State Kalman Filter (ESKF) with Leg Odometry
	 *  @param y 3D base position measurement in the world frame
	 *  @param qy orientation of the base w.r.t the world frame in quaternion
	 *  @note Leg odometry is accurate when accurate contact states are detected
	 */
	void updateWithLegOdom(Vector3d y, Quaterniond qy);
	/** @fn void updateWithTwist(Vector3d y);
	 *  @brief realises the  update step of the Error State Kalman Filter (ESKF) with a base linear velocity measurement
	 *  @param y 3D base velociy measurement in the world frame
	 */
	void updateWithTwist(Vector3d y);
	/** @fn void updateWithTwistRotation(Vector3d y,Quaterniond qy);
	 *  @brief realises the  update step of the Error State Kalman Filter (ESKF) with a base linear velocity measurement and orientation measurement
	 *  @param y 3D base velociy measurement in the world frame
	 * 	@param qy orientation of the base w.r.t the world frame in quaternion
	 */
	void updateWithTwistRotation(Vector3d y,Quaterniond qy);
	/**
	 *  @fn void init()
	 *  @brief Initializes the Base Estimator
     *  @details
     *   Initializes:  State-Error Covariance  P, State x, Linearization Matrices for process and measurement models Acf, Lcf, Hf and rest class variables
	*/
	void init();
	/** @fn Matrix3d wedge(Vector3d v)
	 * 	@brief Computes the skew symmetric matrix of a 3-D vector
	 *  @param v  3D Twist vector 
	 *  @return   3x3 skew symmetric representation
	 */
	Matrix3d wedge(
			Vector3d v) {

		Matrix3d skew;

		skew = Matrix3d::Zero();
		skew(0, 1) = -v(2);
		skew(0, 2) = v(1);
		skew(1, 2) = -v(0);
		skew(1, 0) = v(2);
		skew(2, 0) = -v(1);
		skew(2, 1) = v(0);

		return skew;

	}
	/** @fn Vector3d vec(Matrix3d M)
	 *  @brief Computes the vector represation of a skew symmetric matrix
	 *  @param M  3x3 skew symmetric matrix
	 *  @return   3D Twist vector
	 */
	Vector3d vec(Matrix3d M)
	{
		Vector3d v = Vector3d::Zero();
		v(0) = M(2,1);
		v(1) = M(0,2);
		v(2) = M(1,0);
		return v;
	}

	/** @fn  Matrix<double, 3, 3> expMap(Vector3d omega)
	 *  @brief Computes the exponential map according to the Rodriquez Formula for component in so(3)
	 *  @param omega 3D twist in so(3) algebra
	 *  @return   3x3 Rotation in  SO(3) group
	 */
	inline Matrix<double, 3, 3> expMap(
			Vector3d omega) {

		Matrix<double, 3, 3> res;
		double omeganorm;

		omeganorm = omega.norm();
		res =  Matrix<double, 3, 3>::Identity();

		if(omeganorm > std::numeric_limits<double>::epsilon())
		{
			Matrix<double, 3, 3>  omega_skew;
			omega_skew = Matrix<double, 3, 3>::Zero();

			omega_skew = wedge(omega);
			res += omega_skew * (sin(omeganorm) / omeganorm);
			res += (omega_skew * omega_skew) * (
					(1.000 - cos(omeganorm)) / (omeganorm * omeganorm));
		}

		return res;
	}
	/** @fn Vector3d logMap(Matrix<double, 3, 3> Rt)
	 *  @brief Computes the logarithmic map for a component in SO(3) group
	 *  @param Rt 3x3 Rotation in SO(3) group
	 *  @return   3D twist in so(3) algebra
	 */
	inline Vector3d logMap(
			Matrix<double, 3, 3> Rt) {

		Vector3d res = Vector3d::Zero();
		double costheta = (Rt.trace()-1.0)/2.0;
		double theta = acos(costheta);

		if (fabs(theta) > std::numeric_limits<double>::epsilon()) {
			Matrix<double, 3, 3> lnR = Matrix<double, 3, 3>::Zero();
			lnR.noalias() =  Rt - Rt.transpose();
			lnR *= theta /(2.0*sin(theta));
			res = vec(lnR); 			
		}

		return res;
	}
	/** @fn Vector3d logMap(Quaterniond q)
	 *  @brief Computes the logarithmic map for a component in SO(3) group
	 *  @param q Quaternion in SO(3) group
	 *  @return   3D twist in so(3) algebra
	 */
	inline Vector3d logMap(
			Quaterniond q) {

		Vector3d omega;
		omega = Vector3d::Zero();

		double temp = q.norm();

		Vector3d tempV = Vector3d(q.x(), q.y(), q.z());

		double temp_ = tempV.norm();
		if (temp_ > std::numeric_limits<double>::epsilon()) 
		{
			tempV *= (1.000 / temp_);
			omega = tempV * (2.0 * acos(q.w() / temp));
		}


		return omega;
	}
	/** @fn Vector3d getEulerAngles(Matrix3d Rt)
	 *  @brief Computes Euler Angles from a Rotation Matrix
	 *  @param Rt 3x3 Rotation in SO(3) group
	 *  @return   3D Vector with Roll-Pitch-Yaw
	 */
	inline Vector3d getEulerAngles(
			Matrix3d Rt) {
		Vector3d res;
		res = Vector3d::Zero();

		res(0) = atan2(Rt(2, 1), Rt(2, 2));
		res(1) = atan2(-Rt(2, 0), sqrt(pow(Rt(2, 1), 2) + pow(Rt(2, 2), 2)));
		res(2) = atan2(Rt(1, 0), Rt(0, 0));
		return res;
	}
	/** @fn  Matrix3d getRotationMatrix(Vector3d angles_)
	 *  @brief Computes Rotation Matrix from Euler Angles according to YPR convention
	 *  @param angles_ 3D Vector with Roll-Pitch-Yaw
	 *  @return  3x3 Rotation in SO(3) group
	 */
	inline Matrix3d getRotationMatrix(
			Vector3d angles_) {
		Matrix3d res, Rz, Ry, Rx;
		Rz = Matrix3d::Zero();

		Rz(0, 0) = cos(angles_(2));
		Rz(0, 1) = -sin(angles_(2));
		Rz(1, 0) = sin(angles_(2));
		Rz(1, 1) = cos(angles_(2));
		Rz(2, 2) = 1.000;

		Ry = Matrix3d::Zero();
		Ry(0, 0) = cos(angles_(1));
		Ry(0, 2) = sin(angles_(1));
		Ry(1, 1) = 1.000;
		Ry(2, 0) = -sin(angles_(1));
		Ry(2, 2) = cos(angles_(1));

		Rx = Matrix3d::Zero();
		Rx(0, 0) = 1.00;
		Rx(1, 1) = cos(angles_(0));
		Rx(1, 2) = -sin(angles_(0));
		Rx(2, 1) = sin(angles_(0));
		Rx(2, 2) = cos(angles_(0));

		// YAW PITCH ROLL Convention for right handed counterclowise coordinate systems
		res = Rz * Ry * Rx;

		return res;
	}
};
#endif