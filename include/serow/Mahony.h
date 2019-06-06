#ifndef MAHONY_H
#define MAHONY_H
#include <eigen3/Eigen/Dense>
#include <iostream>

namespace serow{
    class Mahony
    {
    private:
        Eigen::Quaterniond q;
        Eigen::Vector3d acc,gyro;
        Eigen::Matrix3d R;
        double twoKp;				// algorithm propotional gain
        double twoKi;               // algorithm Integral gain
        double q0, q1, q2, q3;	// quaternion of sensor frame relative to auxiliary frame
        double integralFBx,  integralFBy, integralFBz;
        double sampleFreq;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
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
        
        Eigen::Quaterniond getQ()
        {
            return q;
        }
        
        Eigen::Vector3d getAcc()
        {
            return acc;
        }
        Eigen::Vector3d getGyro()
        {
            return gyro;
        }
        Eigen::Matrix3d getR()
        {
            return R;
        }
        Eigen::Vector3d getEuler()
        {
            return q.toRotationMatrix().eulerAngles(0, 1, 2);
        }
        
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