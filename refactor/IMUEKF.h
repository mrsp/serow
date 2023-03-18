/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */

/**
 * @brief Base Estimator combining Inertial Measurement Unit (IMU) and
 * Odometry Measuruements either from leg odometry or external odometry e.g
 * Visual Odometry (VO) or Lidar Odometry (LO)
 * @author Stylianos Piperakis
 * @details State is  position in World frame
 * velocity in  Base frame
 * orientation of Body frame wrt the World frame
 * accelerometer bias in Base frame
 * gyro bias in Base frame
 * Measurements are: Base Position/Orinetation in World frame by Leg Odometry
 * or Visual Odometry (VO) or Lidar Odometry (LO), when VO/LO is considered the
 * kinematically computed base velocity (Twist) is also employed for update.
 */
#pragma once
#include <cmath> // isnan, sqrt
#include <eigen3/Eigen/Dense>
#include "State.hpp"

using namespace Eigen;

class IMUEKF {
   private:
	State state_;
    /// Linearized state-input model input vector is: gyro noise, acc noise, gyro-bias noise, 
	/// acc-bias noise
    Eigen::Matrix<double, 15, 12> Lc_;
    
	Eigen::Matrix<double, 12, 12> Q_;
	
	/// Error Covariance state is velocity in base frame, orientation in the world frame, position 
	/// in the world frame, Gyro and accelerometer biases
    Eigen::Matrix<double, 15, 15> P_, I_;

    /// Robust Gaussian ESKF
    /// Beta distribution parameters
    /// more info in Outlier-Robust State Estimation for Humanoid Robots https://www.researchgate.net/publication/334745931_Outlier-Robust_State_Estimation_for_Humanoid_Robots
    double tau, zeta, f0, e0, e_t, f_t;
    double efpsi, lnp, ln1_p, pzeta_1, pzeta_0, norm_factor;
    /// Updated Measurement noise matrix
    Eigen::Matrix<double, 6, 6> R_z;
    /// Updated Error Covariance matrix
    Eigen::Matrix<double, 15, 15> P_i;
    /// Corrected state
    Eigen::Matrix<double, 15, 1> x_i, x_i_;
    /// Corrected Rotation matrix from base to world frame
    Eigen::Matrix3d Rib_i;
    bool outlier;


    /**
     *  @brief computes the state transition matrix for linearized error state dynamics
     *
     */
    Eigen::Matrix<double, 15, 15> computeTransitionMatrix(const Eigen::Matrix<double, 15, 1>& x,
                                                          const Eigen::Matrix<double, 3, 3>& Rib,
                                                          Eigen::Vector3d angular_velocity,
                                                          Eigen::Vector3d linear_acceleration);
    /**
     *  @brief performs euler (first-order) discretization to the nonlinear state-space dynamics
     *
     */
    void euler(Eigen::Vector3d angular_velocity, Eigen::Vector3d linear_acceleration, double dt);
    /**
     *  @brief updates the parameters for the outlier detection on odometry measurements
     *
     */
    void updateOutlierDetectionParams(Eigen::Matrix<double, 3, 3> B);
    /**
     *  @brief computes the digamma Function Approximation
     *
     */
    double computePsi(double xx);
    /**
     *  @brief computes the discrete-time nonlinear state-space dynamics
     *
     */
    Eigen::Matrix<double, 15, 1> computeDynamics(const Eigen::Matrix<double, 15, 1> x,
                                                 const Eigen::Matrix3d& Rib,
                                                 Eigen::Vector3d angular_velocity,
                                                 Eigen::Vector3d linear_acceleration,
                                                 double dt);

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // State vector - with biases included
    Matrix<double, 15, 1> x;

    bool firstrun;
    bool useEuler;

    // Gravity vector
    Vector3d g, bgyr, bacc, gyro, acc, vel, pos, angle;

    // Noise Stds
    double acc_qx, acc_qy, acc_qz, gyr_qx, gyr_qy, gyr_qz, gyrb_qx, gyrb_qy, gyrb_qz,
        accb_qx, accb_qy, accb_qz, odom_px, odom_py, odom_pz, odom_ax, odom_ay, odom_az,
        vel_px, vel_py, vel_pz, leg_odom_px, leg_odom_py, leg_odom_pz, leg_odom_ax,
        leg_odom_ay, leg_odom_az;

    double gyroX, gyroY, gyroZ, angleX, angleY, angleZ, bias_gx, bias_gy, bias_gz,
        bias_ax, bias_ay, bias_az, ghat;

    double accX, accY, accZ, velX, velY, velZ, rX, rY, rZ;
    double mahalanobis_TH;
    Matrix3d Rib;

    Affine3d Tib;

    Quaterniond qib;

    /**
     *  @brief Constructor of the Base Estimator
     *  @details
     *   Initializes the gravity vector and sets Outlier detection to false
     */
    IMUEKF();

    /** @fn void setImuAngularVelocityBias(Vector3d bgyr_)
     *  @brief initializes the angular velocity bias state of the Error State Kalman Filter (ESKF)
     *  @param bgyr_ angular velocity bias in the base coordinates
     */
    void setImuAngularVelocityBias(Vector3d imu_angular_velocity_bias) {
		state_.imu_gyro_rate_bias_ = std::move(imu_angular_velocity_bias);
    }

    /** @fn void setImuLinearAccelerationBias(Vector3d bacc_)
     *  @brief initializes the acceleration bias state of the Error State Kalman Filter (ESKF)
     *  @param bacc_ acceleration bias in the base coordinates
     */
    void setImuLinearAccelerationBias(Vector3d imu_linear_acceleration_bias) {
       state_.imu_accelaration_bias_ = std::move(imu_linear_acceleration_bias);
    }

    /** @fn void setBasePosition(Eigen::Vector3d base_position)
     *  @brief initializes the base position state of the Error State Kalman Filter (ESKF)
     *  @param base_position Position of the base in the world frame
     */
    void setBasePosition(Eigen::Vector3d base_position) {
		state_.base_position_ = std::move(base_position);
    }

    /** @fn void setBaseOrientation(Eigen::Matrix3d base_orientation)
     *  @brief initializes the base rotation state of the Error State Kalman Filter (ESKF)
     *  @param base_orientation Rotation of the base in the world frame
     */
    void setBaseOrientation(Eigen::Matrix3d base_orientation) {
        state_.base_orientation_  = std::move(Eigen::Quaterniond(base_orientation));
    }
    
    /** @fn void setBaseOrientation(Eigen::Quaterniond base_orientation)
     *  @brief initializes the base rotation state of the Error State Kalman Filter (ESKF)
     *  @param base_orientation Rotation of the base in the world frame
     */
	void setBaseOrientation(Eigen::Quaterniond base_orientation) {
        state_.base_orientation_  = std::move(base_orientation);
    }

    /** @fn void setBaseLinearVelocity(Vector3d bv)
     *  @brief initializes the base velocity state of the Error State Kalman Filter (ESKF)
     *  @param base_linear_velocity linear velocity of the base in the base frame
     */
    void setBaseLinearVelocity(Vector3d base_linear_velocity) {
        state_.base_linear_velocity_ = std::move(base_linear_velocity);
    }

    /** @fn void predict(Eigen::Vector3d imu_angular_velocity, Eigen::Vector3d imu_linear_acceleration);
     *  @brief realises the predict step of the Error State Kalman Filter (ESKF)
     *  @param imu_angular_velocity angular velocity of the base in the base frame 
     *  @param imu_linear_acceleration linear acceleration of the base in the base frame
     */
    void predict(Eigen::Vector3d imu_angular_velocity, Eigen::Vector3d imu_linear_acceleration);
    
	/** @fn void updateWithOdom(const Eigen::Vector3d& y, const Eigen::Quaterniond& qy, bool useOutlierDetection);
     *  @brief realises the pose update step of the Error State Kalman Filter (ESKF)
     *  @param y 3D base position measurement in the world frame
     *  @param qy orientation of the base w.r.t the world frame in quaternion
     *  @param useOutlierDetection check if the measurement is an outlier
     */
    bool updateWithBaseOdometry(const Eigen::Vector3d& y, const Eigen::Quaterniond& qy, bool useOutlierDetection);

    /** @fn void updateWithBaseLinearVelocity(const Eigen::Vector3d& y);
     *  @brief realises the  update step of the Error State Kalman Filter (ESKF) with a base linear velocity measurement
     *  @param y 3D base velocity measurement in the world frame
     */
    void updateWithBaseLinearVelocity(const Eigen::Vector3d& y);
    
	/** @fn void updateWithBaseOrientation(const Eigen::Quaterniond& qy);
     *  @brief realises the  update step of the Error State Kalman Filter (ESKF) with a base orientation measurement
     * 	@param qy 3D orientation of the base w.r.t the world frame in quaternion
     */
    void updateWithBaseOrientation(const Eigen::Quaterniond& qy);
    
	/** @fn void updateWithBasePosition(const Eigen::Vector3d& y);
     *  @brief realises the  update step of the Error State Kalman Filter (ESKF) with a base position measurement
     * 	@param y 3D position of the base w.r.t the world frame in quaternion
     */    
	void updateWithBasePosition(const Eigen::Vector3d& y);

	/**
     *  @fn void init()
     *  @brief Initializes the Base Estimator
     *  @details Initializes:  State-Error Covariance  P, State x, Linearization Matrices for process and measurement models Acf, Lcf, Hf and rest class variables
     */
    void init();
    
	/** @fn Eigen::Matrix3d wedge(const Eigen::Vector3d& v)
     * 	@brief Computes the skew symmetric matrix of a 3-D vector
     *  @param v  3D Twist vector
     *  @return   3x3 skew symmetric representation
     */
    Eigen::Matrix3d wedge(const Eigen::Vector3d& v) {
		Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
        skew(0, 1) = -v(2);
        skew(0, 2) = v(1);
        skew(1, 2) = -v(0);
        skew(1, 0) = v(2);
        skew(2, 0) = -v(1);
        skew(2, 1) = v(0);
        return skew;
    }
    /** @fn Eigen::Vector3d vec(const Eigen::Matrix3d& M)
     *  @brief Computes the vector represation of a skew symmetric matrix
     *  @param M  3x3 skew symmetric matrix
     *  @return   3D Twist vector
     */
    Eigen::Vector3d vec(const Eigen::Matrix3d& M) {
        return Eigen::Vector3d(M(2, 1), M(0, 2), M(1, 0));
    }

    /** @fn Eigen::Matrix3d expMap(const Eigen::Vector3d& omega)
	 *  @brief Computes the exponential map according to the Rodriquez Formula for component in SO(3)
     *  @param omega 3D twist in so(3) algebra
     *  @return 3x3 Rotation in  SO(3) group
     */
    Eigen::Matrix3d expMap(const Eigen::Vector3d& omega) {
    	Eigen::Matrix3d res = Eigen::Matrix3d::Identity();
        const double omeganorm = omega.norm();

        if (omeganorm > std::numeric_limits<double>::epsilon()) {
            Eigen::Matrix3d omega_skew = Eigen::Matrix3d::Zero();
            omega_skew = wedge(omega);
            res += omega_skew * (sin(omeganorm) / omeganorm);
            res += (omega_skew * omega_skew) * ((1.000 - cos(omeganorm)) / (omeganorm * omeganorm));
        }
        return res;
    }

    /** @fn Eigen::Vector3d logMap(const Eigen::Matrix3d& Rt)
	 *  @brief Computes the logarithmic map for a component in SO(3) group
     *  @param Rt 3x3 Rotation in SO(3) group
     *  @return 3D twist in so(3) algebra
     */
    Eigen::Vector3d logMap(const Eigen::Matrix3d& Rt) {
        Eigen::Vector3d res = Eigen::Vector3d::Zero();
        const double costheta = (Rt.trace() - 1.0) / 2.0;
        const double theta = acos(costheta);

        if (std::fabs(theta) > std::numeric_limits<double>::epsilon()) {
            Eigen::Matrix3d lnR = Rt - Rt.transpose();
            lnR *= theta / (2.0 * sin(theta));
            res = vec(lnR);
        }

        return res;
    }

    /**  Eigen::Vector3d logMap(const Eigen::Quaterniond& q)
	 *  @brief Computes the logarithmic map for a component in SO(3) group
     *  @param q Quaternion in SO(3) group
     *  @return   3D twist in so(3) algebra
     */
    Eigen::Vector3d logMap(const Eigen::Quaterniond& q) {
        Eigen::Vector3d omega = Eigen::Vector3d::Zero();
		// Get the vector part
        Eigen::Vector3d qv = Eigen::Vector3d(q.x(), q.y(), q.z());
        qv *= (1.000 / qv.norm());
        omega = qv * (2.0 * std::acos(q.w() / q.norm()));
        if (std::isnan(omega(0) + omega(1) + omega(2))) {
            omega = Eigen::Vector3d::Zero();
		}
        return omega;
    }
};
