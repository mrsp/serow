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
 * @brief IMU Orientation Estimation with the Mahony Filter
 * @author Stylianos Piperakis
 * @details estimates the IMU frame orientation with respect to a world frame of reference with IMU measurements
 * @note updateIMU() is based on the https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/ repository
 */


#ifndef MAHONY_H
#define MAHONY_H
#include <eigen3/Eigen/Dense>
#include <iostream>

namespace serow{
    class Mahony
    {
    private:
        /// IMU orientation w.r.t the world frame as a quaternion
        Eigen::Quaterniond q;
        /// IMU orientation w.r.t the world frame as a rotation matrix
        Eigen::Matrix3d R;
        /// IMU linear acceleration and angular velocity in the world frame
        Eigen::Vector3d acc,gyro;
        /// Algorithm propotional gain
        double twoKp;	
        /// Algorithm Integral gain			
        double twoKi;
        /// Quaternion of sensor frame relative to auxiliary frame
        double q0, q1, q2, q3;	
        /// Integral of angular velocity in x,y,z
        double integralFBx,  integralFBy, integralFBz;
        /// Sampling frequency
        double sampleFreq;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        

        /** @fn  Mahony(double sampleFreq_, double Kp, double Ki = 0.0f)
        *  @brief Initializes parameters and sets the sampling frequency and gains of the algorithm
        *  @param sampleFreq_ sampling frequency
        *  @param Kp Proportional gain
        *  @param Ki Integral gain
        */
        Mahony( double sampleFreq_, double Kp, double Ki = 0.0f)
        {
            R.Identity();
            acc.Zero();
            gyro.Zero();
            sampleFreq = sampleFreq_;
            q.Identity();
            q0 = 1.0f;
            q1 = 0.0f;
            q2 = 0.0f;
            q3 = 0.0f;
            twoKp = 2.0 * Kp;
            twoKi = 2.0 * Ki;
            integralFBx = 0.0;
            integralFBy = 0.0;
            integralFBz = 0.0;
        }
        /** @fn Eigen::Quaterniond getQ()
         *  @returns the orientation of IMU  w.r.t the world frame as a quaternion
         */
        Eigen::Quaterniond getQ()
        {
            return q;
        }
        /** @fn  Eigen::Vector3d getAcc()
         *  @returns the linear acceleration of IMU  in the world frame 
         */        
        Eigen::Vector3d getAcc()
        {
            return acc;
        }
        /** @fn  Eigen::Vector3d getGyro()
         *  @returns the angular velocity  of IMU  in the world frame 
         */        
        Eigen::Vector3d getGyro()
        {
            return gyro;
        }
        /** @fn Eigen::Matrix3d getR()
         *  @returns the orientation of IMU  w.r.t the world frame as a rotation matrix
         */

        Eigen::Matrix3d getR()
        {
            return R;
        }
         /** @fn Eigen::Matrix3d getEuler()
         *  @returns the orientation of IMU  w.r.t the world frame as  euler angles in the RPY convention
         */
        Eigen::Vector3d getEuler()
        {
            return q.toRotationMatrix().eulerAngles(0, 1, 2);
        }
        

        /** @fn updateIMU(Eigen::Vector3d gyro_, Eigen::Vector3d acc_)
         *  @brief Computes the IMU orientation w.r.t the world frame of reference
         *  @param gyro_ angular velocity as measured by the IMU
         *  @param acc_ linea acceleration as measured by the IMU
         */
        void updateIMU(Eigen::Vector3d gyro_, Eigen::Vector3d acc_) {
            double recipNorm;
            double halfvx, halfvy, halfvz;
            double halfex, halfey, halfez;
            double qa, qb, qc;
            double ax,ay,az,gx,gy,gz;
	    
            gx = gyro_(0);
            gy = gyro_(1);
            gz = gyro_(2);
            ax = acc_(0);
            ay = acc_(1);
            az = acc_(2);
            // Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
            if(!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {
                
                // Normalise accelerometer measurement
                recipNorm = 1.0/sqrt(ax * ax + ay * ay + az * az);
                ax *= recipNorm;
                ay *= recipNorm;
                az *= recipNorm;
                
                // Estimated direction of gravity and vector perpendicular to magnetic flux
                halfvx = q1 * q3 - q0 * q2;
                halfvy = q0 * q1 + q2 * q3;
                halfvz = q0 * q0 - 0.5f + q3 * q3;
                
                // Error is sum of cross product between estimated and measured direction of gravity
                halfex = (ay * halfvz - az * halfvy);
                halfey = (az * halfvx - ax * halfvz);
                halfez = (ax * halfvy - ay * halfvx);
                
                // Compute and apply integral feedback if enabled
                if(twoKi > 0.0f) {
                    integralFBx += twoKi * halfex * (1.0f / sampleFreq);    // integral error scaled by Ki
                    integralFBy += twoKi * halfey * (1.0f / sampleFreq);
                    integralFBz += twoKi * halfez * (1.0f / sampleFreq);
                    gx += integralFBx;    // apply integral feedback
                    gy += integralFBy;
                    gz += integralFBz;
                }
                else {
                    integralFBx = 0.0f;    // prevent integral windup
                    integralFBy = 0.0f;
                    integralFBz = 0.0f;
                }
                
                // Apply proportional feedback
                gx += twoKp * halfex;
                gy += twoKp * halfey;
                gz += twoKp * halfez;
            }
            
            // Integrate rate of change of quaternion
            gx *= (0.5f * (1.0f / sampleFreq));        // pre-multiply common factors
            gy *= (0.5f * (1.0f / sampleFreq));
            gz *= (0.5f * (1.0f / sampleFreq));
            qa = q0;
            qb = q1;
            qc = q2;
            q0 += (-qb * gx - qc * gy - q3 * gz);
            q1 += (qa * gx + qc * gz - q3 * gy);
            q2 += (qa * gy - qb * gz + q3 * gx);
            q3 += (qa * gz + qb * gy - qc * gx);
            
            // Normalise quaternion
            recipNorm = 1.0/sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
            q0 *= recipNorm;
            q1 *= recipNorm;
            q2 *= recipNorm;
            q3 *= recipNorm;
            q.x() = q1;
            q.y() = q2;
            q.z() = q3;
            q.w() = q0;
            R = q.toRotationMatrix();
            acc = R*acc_;
            gyro = R*gyro_;
        }
    };
}
#endif