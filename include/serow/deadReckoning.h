#include <eigen3/Eigen/Dense>
namespace serow
{

class deadReckoning
{
  private:
    double Tm, Tm2, ef, wl, wr, mass, g, freq, GRF, Tm3, alpha1, alpha3;
    Eigen::Matrix3d C1l, C2l, C1r, C2r;
    Eigen::Vector3d RpRm, LpLm, RpRmb, LpLmb;

    Eigen::Vector3d pwr, pwl, pwb, pwb_;
    Eigen::Vector3d vwr, vwl, vwb, vwb_r, vwb_l;
    Eigen::Matrix3d Rwr, Rwl, vwb_cov;
    Eigen::Vector3d pwl_, pwr_;
    Eigen::Vector3d pb_l, pb_r;
    Eigen::Matrix3d Rwl_, Rwr_;
    Eigen::Vector3d vwbKCFS;
    Eigen::Vector3d Lomega, Romega;
    Eigen::Vector3d omegawl, omegawr;

    Eigen::Matrix3d AL, AR, wedgerf, wedgelf, RRpRm, RLpLm;
    Eigen::Vector3d bL, bR, plf, prf; //FT w.r.t Foot Frame;
    bool firstrun;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    deadReckoning(Eigen::Vector3d pwl0, Eigen::Vector3d pwr0, Eigen::Matrix3d Rwl0, Eigen::Matrix3d Rwr0,
                  double mass_, double alpha1_ = 1.0, double alpha3_ = 0.01, double freq_ = 100.0, double g_ = 9.81,
                  Eigen::Vector3d plf_ = Eigen::Vector3d::Zero(), Eigen::Vector3d prf_ = Eigen::Vector3d::Zero())
    {
        firstrun = true;
        vwb_cov = Eigen::Matrix3d::Zero();
        vwb_l = Eigen::Vector3d::Zero();
        vwb_r = Eigen::Vector3d::Zero();

        pwl_ = pwl0;
        pwr_ = pwr0;
        Rwl_ = Rwl0;
        Rwr_ = Rwr0;
        Rwl = Rwl_;
        Rwr = Rwr_;
        pwl = pwl_;
        pwr = pwr_;
        pwb_ = Eigen::Vector3d::Zero();
        pwb = pwb_;
        freq = freq_;
        mass = mass_;
        //Joint Freq
        Tm = 1.0 / freq;
        Tm2 = Tm * Tm;
        ef = 0.1;
        g = g_;
        C1l = Eigen::Matrix3d::Zero();
        C2l = Eigen::Matrix3d::Zero();
        C1r = Eigen::Matrix3d::Zero();
        C2r = Eigen::Matrix3d::Zero();

        Lomega = Eigen::Vector3d::Zero();
        Romega = Eigen::Vector3d::Zero();
        vwb = Eigen::Vector3d::Zero();
        pb_l = Eigen::Vector3d::Zero();
        pb_r = Eigen::Vector3d::Zero();
        vwbKCFS = Eigen::Vector3d::Zero();
        plf = plf_;
        prf = prf_;
        RpRm = prf;
        LpLm = plf;
        alpha1 = alpha1_;
        alpha3 = alpha3_;
        Tm3 = (mass * mass * g * g) * Tm2;

    }
    Eigen::Vector3d getOdom()
    {
        return pwb;
    }
    Eigen::Vector3d getLinearVel()
    {
        return vwb;
    }

    void computeBodyVelKCFS(Eigen::Matrix3d Rwb, Eigen::Vector3d omegawb, Eigen::Vector3d pbl, Eigen::Vector3d pbr,
                            Eigen::Vector3d vbl, Eigen::Vector3d vbr, double wl_, double wr_)
    {

        
        //vwb_l.noalias() = -wedge(omegab)  * pbl - vbl;
        //vwb_l =  Rwb * vwb_l;
        //vwb_r.noalias() = -wedge(omegab)  * pbr - vbr;
        //vwb_r =  Rwb * vwb_r;

        vwb_l = wl*(-wedge(omegawb) * Rwb * pbl - Rwb * vbl);
        vwb_r = wr*(-wedge(omegawb) * Rwb * pbr - Rwb * vbr);

        vwb = wl_ * vwb_l;
        vwb += wr_ * vwb_r;
    }

    Eigen::Matrix3d getVelocityCovariance()
    {

        return vwb_cov;
    }

    void computeLegKCFS(Eigen::Matrix3d Rwb, Eigen::Matrix3d Rbl, Eigen::Matrix3d Rbr, Eigen::Vector3d omegawb, Eigen::Vector3d omegabl, Eigen::Vector3d omegabr,
                        Eigen::Vector3d pbl, Eigen::Vector3d pbr, Eigen::Vector3d vbl, Eigen::Vector3d vbr)
    {
        Rwl = Rwb * Rbl;
        Rwr = Rwb * Rbr;
        omegawl = omegawb + Rwb * omegabl;
        omegawr = omegawb + Rwb * omegabr;

        vwl = vwb + wedge(omegawb) * Rwb * pbl + Rwb * vbl;
        vwr = vwb + wedge(omegawb) * Rwb * pbr + Rwb * vbr;
    }

    Eigen::Vector3d getLFootLinearVel()
    {
        return vwl;
    }
    Eigen::Vector3d getRFootLinearVel()
    {
        return vwr;
    }

    Eigen::Vector3d getLFootAngularVel()
    {
        return omegawl;
    }
    Eigen::Vector3d getRFootAngularVel()
    {
        return omegawr;
    }

    Eigen::Vector3d getLFootIMVPPosition()
    {
        return LpLmb;
    }

    Eigen::Vector3d getRFootIMVPPosition()
    {
        return RpRmb;
    }

    Eigen::Matrix3d getLFootIMVPOrientation()
    {
        return RLpLm;
    }

    Eigen::Matrix3d getRFootIMVPOrientation()
    {
        return RRpRm;
    }


    void computeIMVP()
    {
        Lomega = Rwl.transpose() * omegawl;
        Romega = Rwr.transpose() * omegawr;

        double temp = Tm2 / (Lomega.squaredNorm() * Tm2 + 1.0);

        C1l = temp * wedge(Lomega);
        C2l = temp * (Lomega * Lomega.transpose() + 1.0 / Tm2 * Matrix3d::Identity());

        temp = Tm2 / (Romega.squaredNorm() * Tm2 + 1.0);

        C1r = temp * wedge(Romega);
        C2r = temp * (Romega * Romega.transpose() + 1.0 / Tm2 * Matrix3d::Identity());

        //IMVP Computations
        LpLm = C2l * LpLm;

        LpLm = LpLm + C1l * Rwl.transpose() * vwl;

        RpRm = C2r * RpRm;

        RpRm = RpRm + C1r * Rwr.transpose() * vwr;
    }
    void computeIMVPFT(Eigen::Vector3d lf, Eigen::Vector3d rf, Eigen::Vector3d lt, Eigen::Vector3d rt)
    {
        Lomega = Rwl.transpose() * omegawl;
        Romega = Rwr.transpose() * omegawr;

        wedgerf = wedge(rf);
        wedgelf = wedge(lf);

        AL.noalias() = 1.0 / Tm2 * Matrix3d::Identity();
        AL.noalias() -= alpha1 * wedge(Lomega) * wedge(Lomega);
        AL.noalias() -= alpha3 / Tm3 * wedgelf * wedgelf;

        bL.noalias() = 1.0 / Tm2 * LpLm;
        bL.noalias() += alpha1 * wedge(Lomega) * Rwl.transpose() * vwl;
        bL.noalias() += alpha3 / Tm3 * (wedgelf * lt - wedgelf * wedgelf * plf);

        LpLm.noalias() = AL.inverse() * bL;

        AR.noalias() = 1.0 / Tm2 * Matrix3d::Identity();
        AR.noalias() -= alpha1 * wedge(Romega) * wedge(Romega);
        AR.noalias() -= alpha3 / Tm3 * wedgerf * wedgerf;

        bR.noalias() = 1.0 / Tm2 * RpRm;
        bR.noalias() += alpha1 * wedge(Romega) * Rwr.transpose() * vwr;
        bR.noalias() += alpha3 / Tm3 * (wedgerf * rt - wedgerf * wedgerf * prf);

        RpRm.noalias() = AR.inverse() * bR;


    }

    void computeDeadReckoning(Eigen::Matrix3d Rwb, Eigen::Matrix3d Rbl, Eigen::Matrix3d Rbr,
                              Eigen::Vector3d omegawb, Eigen::Vector3d bomegab,
                              Eigen::Vector3d pbl, Eigen::Vector3d pbr,
                              Eigen::Vector3d vbl, Eigen::Vector3d vbr,
                              Eigen::Vector3d omegabl, Eigen::Vector3d omegabr,
                              double lfz, double rfz,  Eigen::Vector3d lf, Eigen::Vector3d rf, Eigen::Vector3d lt, Eigen::Vector3d rt)
    {

        //Compute Body position
        //Cropping the vertical GRF
        lfz = cropGRF(lfz);
        rfz = cropGRF(rfz);
        //GRF Coefficients
        wl = (lfz + ef) / (lfz + rfz + 2.0 * ef);
        wr = (rfz + ef) / (lfz + rfz + 2.0 * ef);

        computeBodyVelKCFS(Rwb, omegawb, pbl, pbr, vbl, vbr, wl, wr);
 


        computeLegKCFS(Rwb, Rbl, Rbr, omegawb, omegabl, omegabr, pbl, pbr, vbl, vbr);



        if(alpha3>0)
            computeIMVPFT(lf, rf, lt, rt);
        else
            computeIMVP();

      
 
        //Temp estimate of Leg position w.r.t Inertial Frame
        pwl = pwl_ - Rwl * LpLm + Rwl_ * LpLm;
        pwr = pwr_ - Rwr * RpRm + Rwr_ * RpRm;

        //Leg odometry with left foot
        pb_l = pwl - Rwb * pbl;
        //Leg odometry with right foot
        pb_r = pwr - Rwb * pbr;


        //Leg Odometry Estimate
        pwb_ = pwb;
        pwb = wl * pb_l + wr * pb_r;
        //Leg Position Estimate w.r.t Inertial Frame
        pwl += pwb - pb_l;
        pwr += pwb - pb_r;

        
        RpRmb =  pbr;
        LpLmb =  pbl;

        // RpRmb =  Rwb.transpose()*(pwr-pwb);
        // LpLmb =  Rwb.transpose()*(pwl-pwb);
        RLpLm = Rbl;
        RRpRm = Rbr;

        //Needed in the next iteration
        Rwl_ = Rwl;
        Rwr_ = Rwr;
        pwl_ = pwl;
        pwr_ = pwr;
        if (!firstrun)
            vwb = (pwb - pwb_) * freq;
        else
            firstrun = false;
       
    
        vwb_cov.noalias() =  wl * (vwb_l - vwb) * (vwb_l - vwb).transpose();
        vwb_cov.noalias() += wr * (vwb_r - vwb) * (vwb_r - vwb).transpose();

        

 
    }


    void computeDeadReckoningGEM(Eigen::Matrix3d Rwb, Eigen::Matrix3d Rbl, Eigen::Matrix3d Rbr,
                              Eigen::Vector3d omegawb,
                              Eigen::Vector3d pbl, Eigen::Vector3d pbr,
                              Eigen::Vector3d vbl, Eigen::Vector3d vbr,
                              Eigen::Vector3d omegabl, Eigen::Vector3d omegabr,
                              double wl_, double wr_, Eigen::Vector3d lf, Eigen::Vector3d rf, Eigen::Vector3d lt, Eigen::Vector3d rt)
    {

        
        wl = (wl_ + ef)/(wl_ + wr_ + 2.0 *ef);
        wr = (wr_ + ef)/(wl_ + wr_ + 2.0 *ef);

        computeBodyVelKCFS(Rwb, omegawb, pbl, pbr, vbl, vbr, wl, wr);



        computeLegKCFS(Rwb, Rbl, Rbr, omegawb, omegabl, omegabr, pbl, pbr, vbl, vbr);
       
        if(alpha3>0)
            computeIMVPFT(lf, rf, lt, rt);
        else
            computeIMVP();
        //Temp estimate of Leg position w.r.t Inertial Frame
        pwl = pwl_ - Rwl * LpLm + Rwl_ * LpLm;
        pwr = pwr_ - Rwr * RpRm + Rwr_ * RpRm;

        //Leg odometry with left foot
        pb_l = pwl - Rwb * pbl;
        //Leg odometry with right foot
        pb_r = pwr - Rwb * pbr;


        //Leg Odometry Estimate
        pwb_ = pwb;
        pwb = wl * pb_l + wr * pb_r;
        //Leg Position Estimate w.r.t Inertial Frame
        pwl += pwb - pb_l;
        pwr += pwb - pb_r;
        //Needed in the next iteration
        Rwl_ = Rwl;
        Rwr_ = Rwr;
        pwl_ = pwl;
        pwr_ = pwr;
        if (!firstrun)
            vwb = (pwb - pwb_) * freq;
        else
            firstrun = false;
        //cout<<"DEAD RECKONING "<<endl;
        //cout<<pwb<<endl;
    }

    double cropGRF(double f_)
    {
        return max(0.0, min(f_, mass * g));
    }
    //Computes the skew symmetric matrix of a 3-D vector
    Eigen::Matrix3d wedge(Eigen::Vector3d v)
    {
        Eigen::Matrix3d res = Eigen::Matrix3d::Zero();

        res(0, 1) = -v(2);
        res(0, 2) = v(1);
        res(1, 2) = -v(0);
        res(1, 0) = v(2);
        res(2, 0) = -v(1);
        res(2, 1) = v(0);

        return res;
    }
};

} // namespace serow
