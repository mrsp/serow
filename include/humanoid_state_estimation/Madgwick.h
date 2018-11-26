#include <eigen3/Eigen/Dense>
#include <iostream>

namespace serow{
    class Madgwick
    {
        private:
            Eigen::Quaterniond q;
            Eigen::Vector3d acc,gyro;
            Eigen::Matrix3d R;
            double beta;				// algorithm gain
            double freq;
            double q0, q1, q2, q3;	// quaternion of sensor frame relative to auxiliary frame
        
        public:

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
                cout<<"R "<<R<<endl;
                return R;
            }
            Eigen::Vector3d getEuler()
            {
                return q.toRotationMatrix().eulerAngles(0, 1, 2);
            }
            void MadgwickAHRSupdateIMU(Eigen::Vector3d gyro_, Eigen::Vector3d acc_)
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