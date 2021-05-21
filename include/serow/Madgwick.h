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
 * @brief IMU Orientation Estimation with the Madgwick Filter
 * @author Stylianos Piperakis
 * @details estimates the IMU frame orientation with respect to a world frame of reference with IMU measurements
 * @note updateIMU() is based on the https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/ repository
 */


#ifndef MADGWICK_H
#define MADGWICK_H
#include <eigen3/Eigen/Dense>
#include <iostream>

namespace serow{
    class Madgwick
    {
    private:
        // IMU orientation w.r.t the world frame as a quaternion
        Eigen::Quaterniond q;
        // IMU orientation w.r.t the world frame as a rotation matrix
        Eigen::Matrix3d R;
        // IMU linear acceleration and angular velocity in the world frame
        Eigen::Vector3d acc,gyro;
        /// Algorithm gain
        double beta;				
        /// Sampling frequency
        double freq;          
        /// Quaternion of sensor frame relative to auxiliary frame
        double q0, q1, q2, q3;	
        
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        

        /** @fn   Madgwick(double freq_, double beta_)
        *  @brief Initializes parameters and sets the sampling frequency and gain of the algorithm
        *  @param sampleFreq_ sampling frequency
        *  @param beta_ Proportional gain
        */
        Madgwick(double freq_, double beta_)
        {
            R.Identity();
            acc.Zero();
            gyro.Zero();
            freq = freq_;
            beta = beta_;  //2 * proportional gain (Kp)
            q.Identity();
            q0 = 1.0f;
            q1 = 0.0f;
            q2 = 0.0f;
            q3 = 0.0f;
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
        void updateIMU(Eigen::Vector3d gyro_, Eigen::Vector3d acc_)
        {
            
            
            double gx = gyro_(0);
            double gy = gyro_(1);
            double gz = gyro_(2);
            
            double ax = acc_(0);
            double ay = acc_(1);
            double az = acc_(2);
            
            double recipNorm;
            double s0, s1, s2, s3;
            double qDot1, qDot2, qDot3, qDot4;
            double _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2 ,_8q1, _8q2, q0q0, q1q1, q2q2, q3q3;
            
            // Rate of change of quaternion from gyroscope
            qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
            qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
            qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
            qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);
            
            // Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
            if(!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {
                
                // Normalise accelerometer measurement
                recipNorm = 1.0/sqrt(ax * ax + ay * ay + az * az);
                ax *= recipNorm;
                ay *= recipNorm;
                az *= recipNorm;
                
                // Auxiliary variables to avoid repeated arithmetic
                _2q0 = 2.0f * q0;
                _2q1 = 2.0f * q1;
                _2q2 = 2.0f * q2;
                _2q3 = 2.0f * q3;
                _4q0 = 4.0f * q0;
                _4q1 = 4.0f * q1;
                _4q2 = 4.0f * q2;
                _8q1 = 8.0f * q1;
                _8q2 = 8.0f * q2;
                q0q0 = q0 * q0;
                q1q1 = q1 * q1;
                q2q2 = q2 * q2;
                q3q3 = q3 * q3;
                
                // Gradient decent algorithm corrective step
                s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
                s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
                s2 = 4.0f * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
                s3 = 4.0f * q1q1 * q3 - _2q1 * ax + 4.0f * q2q2 * q3 - _2q2 * ay;
                recipNorm = 1.0/sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3); // normalise step magnitude
                s0 *= recipNorm;
                s1 *= recipNorm;
                s2 *= recipNorm;
                s3 *= recipNorm;
                
                // Apply feedback step
                qDot1 -= beta * s0;
                qDot2 -= beta * s1;
                qDot3 -= beta * s2;
                qDot4 -= beta * s3;
            }
            
            // Integrate rate of change of quaternion to yield quaternion
            q0 += qDot1 * (1.0f / freq);
            q1 += qDot2 * (1.0f / freq);
            q2 += qDot3 * (1.0f / freq);
            q3 += qDot4 * (1.0f / freq);
            
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