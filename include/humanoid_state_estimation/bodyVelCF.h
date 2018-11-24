#include <math.h>
#include <eigen3/Eigen/Dense>

namespace serow{

    class bodyVelCF{
        private:
            Eigen::Vector3d vwb,vwb0;
            Eigen::Vector3d acc0;
            Eigen::Vector3d vwbKCFS0;

            //IMU preprocessing
            Eigen::Vector3d G, acc_bias;
            //Filter sampling time and cross-over frequency
            double Ts, Tv;
            double freqvmin,freqvmax;
            double mass, g;

            double computeCrossoverFreq(double GRFz)
            {
                if(GRFz <= 0.0)
                    Tv = 2.0*M_PI * freqvmin;
                else if(GRFz>=mass*g)
                    Tv = 2.0*M_PI * freqvmax;
                else
                {
                    Tv = (freqvmax-freqvmin)*GRFz/(mass*g) + freqvmin; //linear Interpolation
                    Tv *= (2.0*M_PI);
                }
            }

        public:
            bodyVelCF(double freq_, double mass_, double freqvmin_ = 0.001, double freqvmax_ = 0.5, double g_ = 9.81, Eigen::Vector3d acc_bias_ = Eigen::Vector3d::Zero())
            {
                acc_bias = acc_bias_; //Bias is in the world frame
                freqvmax = freqvmax_;
                freqvmin = freqvmin_;
                mass = mass_;
                g = g_;
                Ts = 1.0/freq_;
                Tv = 2.0*M_PI * freqvmax;
                G.Zero();
                G(2) = g;
                vwb.Zero();
                vwb0.Zero();
                vwbKCFS0.Zero();
                acc0.Zero();
            }

            Eigen:: Vector3d filter(Eigen::Vector3d vwbKCFS, Eigen::Vector3d acc, double GRFz)
            {
                acc -=  G;
                acc -= acc_bias;

                Tv = computeCrossoverFreq(GRFz);

                double temp = Ts+2.0*Tv;
                vwb = -(Ts-2.0*Tv)/temp * vwb0 + Ts*Tv/temp * (acc + acc0) + Ts/temp * (vwbKCFS + vwbKCFS0);

                acc0 = acc;
                vwb0 = vwb;
                vwbKCFS0 = vwbKCFS;
                return vwb;
            }

        
    };
}