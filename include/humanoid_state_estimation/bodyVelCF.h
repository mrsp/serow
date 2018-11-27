#include <math.h>
#include <eigen3/Eigen/Dense>
#include <fstream>
using namespace std;
namespace serow{

    class bodyVelCF{
        private:
            Eigen::Vector3d vwb,vwb0;
            Eigen::Vector3d acc0;
            Eigen::Vector3d vwbKCFS0;

            //IMU preprocessing
            Eigen::Vector3d G;
            //Filter sampling time and cross-over frequency
            double Ts, Tv;
            double freqvmin,freqvmax;
            double mass, g;
            ofstream outFileX,outFileY,outFileZ,outFileKX,outFileKY,outFileKZ;

            double computeCrossoverFreq(double GRFz)
            {
                double Tv_;
   

                if(GRFz <= 0.0)
                    Tv_ = 1.0/(2.0*M_PI * freqvmin);
                else if(GRFz>=mass*g)
                    Tv_ = 1.0/(2.0*M_PI * freqvmax);
                else
                {
                    Tv_ = (freqvmax-freqvmin)*GRFz/(mass*g) + freqvmin; //linear Interpolation
                    Tv_ = 1.0/ (2.0*M_PI * Tv_);
                }
                return Tv_;
            }

        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            bodyVelCF(double freq_, double mass_, double freqvmin_ = 0.1, double freqvmax_ = 2.5, double g_ = 9.81)
            {
                outFileX.open("/home/master/outputX.txt");
                outFileY.open("/home/master/outputY.txt");
                outFileZ.open("/home/master/outputZ.txt");
                outFileKX.open("/home/master/outputKX.txt");
                outFileKY.open("/home/master/outputKY.txt");
                outFileKZ.open("/home/master/outputKZ.txt");
                freqvmax = freqvmax_;
                freqvmin = freqvmin_;
                mass = mass_;
                g = g_;
                Ts = 1.0/freq_;
                Tv = 1.0/(2.0*M_PI * freqvmax);
                G = Eigen::Vector3d::Zero();
                G(2) = g;
                vwb = Eigen::Vector3d::Zero();
                vwb0 = Eigen::Vector3d::Zero();
                vwbKCFS0 = Eigen::Vector3d::Zero();
                acc0 = Eigen::Vector3d::Zero();
            }

            Eigen:: Vector3d filter(Eigen::Vector3d vwbKCFS, Eigen::Vector3d acc, double GRFz)
            {
                
                acc -=  G;
                Tv=computeCrossoverFreq(GRFz);

                double temp = Ts+2.0*Tv;
                //Tustin Transform
                vwb = -(Ts-2.0*Tv)/temp * vwb0 + Ts*Tv/temp * (acc + acc0) + Ts/temp * (vwbKCFS + vwbKCFS0);


	            outFileKX<<vwbKCFS(0)<<std::endl;
	            outFileKY<<vwbKCFS(1)<<std::endl;
	            outFileKZ<<vwbKCFS(2)<<std::endl;
	            outFileX<<vwb(0)<<std::endl;
	            outFileY<<vwb(1)<<std::endl;
	            outFileZ<<vwb(2)<<std::endl;
                acc0 = acc;
                vwb0 = vwb;
                vwbKCFS0 = vwbKCFS;
                return vwb;
            }

        
    };
}