#include <eigen3/Eigen/Dense>
#include "humanoid_state_estimation/bodyVelCF.h"

namespace serow{
    
    class deadReckoning{
    private:
        double Tm, Tm2, ef, wl, wr, mass, g, freq, GRF;
        Eigen::Matrix3d C1l, C2l, C1r, C2r;
        Eigen::Vector3d RpRm, LpLm;
        
        Eigen::Vector3d pwr,pwl,pwb;
        Eigen::Vector3d vwr,vwl,vwb;

        Eigen::Matrix3d Rwr,Rwl;
        
        Eigen::Vector3d pwl_,pwr_;
        Eigen::Vector3d pb_l, pb_r;
        Eigen::Matrix3d Rwl_,Rwr_;
        Eigen::Vector3d vwbKCFS;
        Eigen::Vector3d Lomega, Romega;
        Eigen::Vector3d omegawl, omegawr;

        bodyVelCF* bvcf;
    public:
        deadReckoning(Eigen::Vector3d pwl0, Eigen::Vector3d pwr0, Eigen::Matrix3d Rwl0, Eigen::Matrix3d Rwr0,
                      double mass_, double Tm_ = 0.4, double ef_ = 0.3, double freq_ = 100.0, double g_= 9.81,  double freqvmin_ = 0.001, double freqvmax_ = 0.5)
        {
            
            pwl_ = pwl0;
            pwr_ = pwr0;
            Rwl_ = Rwl0;
            Rwr_ = Rwr0;
            Rwl = Rwl_;
            Rwr = Rwr_;
            pwl = pwl_;
            pwr = pwr_;

            freq = freq_;
            mass = mass_;
            Tm = Tm_;
            Tm2 = Tm*Tm;
            ef = ef_;
            g =  g_;
            C1l.Zero();
            C2l.Zero();
            C1r.Zero();
            C2r.Zero();
            RpRm.Zero();
            LpLm.Zero();
            Lomega.Zero();
            Romega.Zero();
            vwb.Zero();
            pb_l.Zero();
            pb_r.Zero();
            vwbKCFS.Zero();
            bvcf = new bodyVelCF(freq_, mass_, freqvmin_ ,  freqvmax_ ,  g_);

        }
        
        void computeBodyVelKCFS(Eigen::Matrix3d Rwb,Eigen::Vector3d omegawb, Eigen::Vector3d pbl, Eigen::Vector3d pbr,
                             Eigen::Vector3d vbl, Eigen::Vector3d vbr, double lfz, double rfz)
        {
            

            if(lfz>rfz)
                vwbKCFS= -wedge(omegawb) * Rwb * pbl - Rwb * vbl;
            else
                vwbKCFS= -wedge(omegawb) * Rwb * pbr - Rwb * vbr;

            vwb = vwbKCFS;
           
        }
        
        void computeLegKCFS(Eigen::Matrix3d Rwb,Eigen::Matrix3d Rbl, Eigen::Matrix3d Rbr, Eigen::Vector3d omegawb, Eigen::Vector3d omegabl, Eigen::Vector3d omegabr,
                            Eigen::Vector3d pbl, Eigen::Vector3d pbr, Eigen::Vector3d vbl, Eigen::Vector3d vbr)
        {
            Rwl = Rwb * Rbl;
            Rwr = Rwb * Rbr;
            omegawl = omegawb + Rwb * omegabl;
            omegawr = omegawb + Rwb * omegabr;
            
            vwl = vwb + wedge(omegawb) * Rwb * pbl + Rwb * vbl;
            vwr = vwb + wedge(omegawb) * Rwb * pbr + Rwb * vbr;


        }
        void computeIMVP()
        {
            Lomega=Rwl.transpose()*omegawl;
            Romega=Rwr.transpose()*omegawr;

            double temp =Tm2/(Lomega.squaredNorm()*Tm2+1.0);

            cout<<" L O "<<endl;
            C1l = temp * wedge(Lomega);
            C2l = temp * (Lomega * Lomega.transpose() + 1.0/Tm2 * Matrix3d::Identity());
            cout<<C1l<<endl;
            cout<<C2l<<endl;


            temp = Tm2/(Romega.squaredNorm()*Tm2+1.0);
            cout<<" R O "<<endl;

            C1r = temp * wedge(Romega);
            C2r = temp * (Romega * Romega.transpose() + 1.0/Tm2 * Matrix3d::Identity());
            cout<<C1r<<endl;
            cout<<C2r<<endl;
            //IMVP Computations
            LpLm = C2l * LpLm;
            LpLm += C1l * Rwl.transpose() * vwl;
            
            RpRm = C2r * RpRm;
            RpRm += C1r * Rwr.transpose() * vwr;
            
            cout<<" RPRM "<<RpRm<<endl;
            cout<<" LPRM "<<LpLm<<endl;
            if(RpRm(0)>1.0)
            {
                 unsigned int *ip = NULL;
                printf("Value stored at Address stored in pointer: %d \n",*ip); // gives "Segmentation fault (core dumped)"
            }

		
        }
        
        
        void computeDeadReckoning(Eigen::Matrix3d Rwb, Eigen::Matrix3d Rbl, Eigen::Matrix3d Rbr,
                                  Eigen::Vector3d omegawb,
                                  Eigen::Vector3d pbl, Eigen::Vector3d pbr,
                                  Eigen::Vector3d vbl, Eigen::Vector3d vbr,
                                  Eigen::Vector3d omegabl, Eigen::Vector3d omegabr,
                                  double lfz, double rfz, Eigen::Vector3d  acc, string support_leg)
        {
            
            //Compute Body position
            double GRF=lfz+rfz;
            computeBodyVelKCFS(Rwb, omegawb,  pbl,  pbr, vbl, vbr, lfz, rfz);
            GRF = cropGRF(GRF);

            bvcf->filter(vwbKCFS, acc, GRF);
            computeLegKCFS(Rwb,  Rbl, Rbr, omegawb,  omegabl, omegabr, pbl,  pbr,  vbl,  vbr);
            computeIMVP();
            
            //Temp estimate of Leg position w.r.t Inertial Frame
            pwl = pwl_ - Rwl*LpLm + Rwl_*LpLm;
            pwr = pwr_ - Rwr*RpRm + Rwr_*RpRm;
            
            //Leg odometry with left foot
            pb_l = pwl - Rwb * pbl;
            //Leg odometry with right foot
            pb_r = pwr - Rwb * pbr;
            
            //Cropping the vertical GRF
            lfz = cropGRF(lfz);
            rfz = cropGRF(rfz);
            
            //GRF Coefficients
            wl = (lfz + ef)/(lfz+rfz+2.0*ef);
            wr = (rfz + ef)/(lfz+rfz+2.0*ef);
            
            
            //Leg Odometry Estimate
            pwb = wl * pb_l + wr * pb_r;
            //Leg Position Estimate w.r.t Inertial Frame
            pwl += pwb - pb_l;
            pwr += pwb - pb_r;
            //Needed in the next iteration
            Rwl_ = Rwl;
            Rwr_ = Rwr;
            pwl_ = pwl;
            pwr_ = pwr;
            cout<<"DEAD RECKONING "<<endl;
            cout<<pwb<<endl;
        }
        
        double cropGRF(double f_)
        {
            return  max(0.0,min(f_,mass*g));
        }
	//Computes the skew symmetric matrix of a 3-D vector
    	Matrix3d wedge(Vector3d v)
    	{
		Matrix3d res;
		res.Zero();
		
		res(0, 1) = -v(2);
		res(0, 2) = v(1);
		res(1, 2) = -v(0);
		res(1, 0) = v(2);
		res(2, 0) = -v(1);
		res(2, 1) = v(0);
		
		return res;
	}

        
    };
    

    
}
