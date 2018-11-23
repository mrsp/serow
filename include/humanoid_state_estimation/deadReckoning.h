#include <eigen3/Eigen/Dense>


namespace serow{
    
    class deadReckoning{
    private:
        double Tm, Tm2, ef, wl, wr, mass, g;
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
        Eigen::Vector3d omegawl, omegawl;

    public:
        deadReckoning(Eigen::Vector3d pwl, Eigen::Vector3d pwl, Eigen::Matrix3d Rwl, Eigen::Matrix3d Rwr,
                      double mass_, double Tm_ = 0.3, double ef_ = 0.3, double g_= 9.81)
        {
            
            pwl_ = pwl;
            pwr_ = pwr;
            Rwl_ = Rwl;
            Rwr_ = Rwr;
            
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
        }
        
        void computeBodyVelKCFS(Eigen::Matrix3d Rwb,Eigen::Vector3d omegawb, Eigen::Vector3d pbl, Eigen::Vector3d pbr,
                             Eigen::Vector3d vbl, Eigen::Vector3d vbr, string support_leg)
        {
            if(support_leg == "LLeg")
                vwbKCFS= -serow::wedge(omegawb) * Rwb * pbl - Rwb * vbl;
            else
                vwbKCFS= -serow::wedge(omegawb) * Rwb * pbr - Rwb * vbr;

            vwb = vwbKCFS;
        }
        
        void computeLegKCFS(Eigen::Matrix3d Rwb, Eigen::Vector3d omegawb, Eigen::Vector3d omegabl, Eigen::Vector3d omegabr,
                            Eigen::Vector3d pbl, Eigen::Vector3d pbr, Eigen::Vector3d vbl, Eigen::Vector3d vbr)
        {
            Rwl = Rwb * Rbl;
            Rwr = Rwb * Rbr;
            omegawl = omegawb + Rwb * omegabl;
            omegawr = omegawb + Rwb * omegabr;
            
            vwl = vwb + serow::wedge(omegawb) * Rwb * pbl + Rwb * vbl;
            vwr = vwb + serow::wedge(omegawb) * Rwb * pbr + Rwb * vbr;

        }
        void computeIMVP()
        {
            Lomega=Rwl.transpose()*omegawl;
            Romega=Rwr.transpose()*omegawr;

            
            double temp =Tm2/(Lomega.squaredNorm()*Tm2+1.0f);
            C1l = temp * serow::wedge(Lomega);
            C2l = temp * (Lomega * Lomega.transpose() + 1.0f/Tm2 * Matrix3d::Identity());

            temp = Tm2/(Romega.squaredNorm()*Tm2+1.0f);
            C1r = temp * serow::wedge(Romega);
            C2r = temp * Romega * Romega.transpose() + 1.0f/Tm2 * Matrix3d::Identity();
            
            //IMVP Computations
            LpLm = C2l * LpLm;
            LpLm += C1l * Rwl.transpose() * vwl;
            
            RpRm = C2r * RpRm;
            RpRm += C1r * Rwr.transpose() * vwr;
        }
        
        
        void computeDeadReckoning(Eigen::Matrix3d Rwb, Eigen::Vector3d omegawb,
                                  Eigen::Vector3d pbl, Eigen::Vector3d pbr,
                                  Eigen::Vector3d vbl, Eigen::Vector3d vbr,
                                  Eigen::Vector3d omegabl, Eigen::Vector3d omegabr,
                                  double lfz, double rfz, string support_leg)
        {
            
            //Compute Body position
            computeBodyVelKCFS(Rwb, omegawb,  pbl,  pbr, vbl, vbr, support_leg);
            computeLegKCFS(Rwb,  omegawb,  omegabl, omegabr, pbl,  pbr,  vbl,  vbr);
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
            
        }
        
        double cropGRF(double f_)
        {
            if(f_ > mass*g)
                f_ = mass*g;
            if(f_ < 0.0)
                f_ = 0.0;
            
            return f_;
        }
        
    };
    
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

    
}
