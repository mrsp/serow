#include <eigen3/Eigen/Dense>
namespace serow
{

class deadReckoningQuad
{
  private:
    double Tm, Tm2, ef, wLF, wLH, wRF, wRH, mass, g, freq, Tm3, alpha1, alpha3;
    Eigen::Matrix3d C1l, C2l, C1r, C2r;
    Eigen::Vector3d RFpRm, RHpRm, LFpLm, LHpLm, LFpLmb,  RHpRmb, LHpLmb, RFpRmb;
    Eigen::Vector3d pwRF, pwRH, pwLF, pwLH, pwb, pwb_;
    Eigen::Vector3d vwRF, vwRH, vwLF, vwLH, vwb, vwb_RF, vwb_RH, vwb_LF, vwb_LH;
    Eigen::Matrix3d RwRF, RwRH, RwLF, RwLH, vwb_cov;
    Eigen::Vector3d pwLF_, pwLH_, pwRF_, pwRH_;
    Eigen::Vector3d pb_LF, pb_RF, pb_LH, pb_RH;
    Eigen::Matrix3d RwLF_, RwLH_, RwRF_, RwRH_;
    Eigen::Vector3d vwbKCFS;
    Eigen::Vector3d LFomega, LHomega, RFomega, RHomega;
    Eigen::Vector3d omegawLH, omegawRH, omegawLF, omegawRF;

    Eigen::Matrix3d AL, AR, wedgeLFf, wedgeLHf, wedgeRHf, wedgeRFf, RRFpRm, RRHpRm, RLFpLm, RLHpLm;
    Eigen::Vector3d bL, bR, pLFf, pLHf, pRFf, pRHf; //FT w.r.t Foot Frame;
    bool firstrun;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    deadReckoningQuad(Eigen::Vector3d pwLF0, Eigen::Vector3d pwLH0, Eigen::Vector3d pwRF0, Eigen::Vector3d pwRH0,
                  Eigen::Matrix3d RwLF0, Eigen::Matrix3d RwLH0, Eigen::Matrix3d RwRF0, Eigen::Matrix3d RwRH0,
                  double mass_, double alpha1_ = 1.0, double alpha3_ = 0.01, double freq_ = 100.0, double g_ = 9.81,
                  Eigen::Vector3d pLFf_ = Eigen::Vector3d::Zero(),  Eigen::Vector3d pLHf_ = Eigen::Vector3d::Zero(), 
                  Eigen::Vector3d pRFf_ = Eigen::Vector3d::Zero(), Eigen::Vector3d pRHf_ = Eigen::Vector3d::Zero())
    {
        firstrun = true;
        vwb_cov = Eigen::Matrix3d::Zero();
        vwb_LF = Eigen::Vector3d::Zero();
        vwb_RF = Eigen::Vector3d::Zero();
        vwb_LH = Eigen::Vector3d::Zero();
        vwb_RH = Eigen::Vector3d::Zero();

        pwLF_ = pwLF0;
        pwLH_ = pwLH0;
        pwRH_ = pwRH0;
        pwRF_ = pwRF0;

        RwLF_ = RwLF0;
        RwLH_ = RwLH0;

        RwRF_ = RwRF0;
        RwRH_ = RwRH0;

        pwLF = pwLF0;
        pwLH = pwLH0;
        pwRH = pwRH0;
        pwRF = pwRF0;

        RwLF = RwLF0;
        RwLH = RwLH0;

        RwRF = RwRF0;
        RwRH = RwRH0;

        pwb_ = Eigen::Vector3d::Zero();
        pwb = pwb_;
        freq = freq_;
        mass = mass_;
        //Joint Freq
        Tm = 1.0 / freq;
        Tm2 = Tm * Tm;
        Tm3 = (mass * mass * g * g) * Tm2;
        ef = 0.1;
        
        g = g_;
        C1l = Eigen::Matrix3d::Zero();
        C2l = Eigen::Matrix3d::Zero();
        C1r = Eigen::Matrix3d::Zero();
        C2r = Eigen::Matrix3d::Zero();

        LFomega = Eigen::Vector3d::Zero();
        RFomega = Eigen::Vector3d::Zero();

        LHomega = Eigen::Vector3d::Zero();
        RHomega = Eigen::Vector3d::Zero();
        vwb = Eigen::Vector3d::Zero();
        pb_LF = Eigen::Vector3d::Zero();
        pb_RF = Eigen::Vector3d::Zero();
        pb_LH = Eigen::Vector3d::Zero();
        pb_RH = Eigen::Vector3d::Zero();
        vwbKCFS = Eigen::Vector3d::Zero();
        pLFf = pLFf_;
        pRFf = pRFf_;
        pLHf = pLHf_;
        pRHf = pRHf_;

        RFpRm = pRFf;
        LFpLm = pLFf;
        RHpRm = pRHf;
        LHpLm = pLHf;

        alpha1 = alpha1_;
        alpha3 = alpha3_;

    }
    Eigen::Vector3d getOdom()
    {
        return pwb;
    }
    Eigen::Vector3d getLinearVel()
    {
        return vwb;
    }

    void computeBodyVelKCFS(Eigen::Matrix3d Rwb, Eigen::Vector3d omegab, Eigen::Vector3d pbLF, Eigen::Vector3d pbLH,
                            Eigen::Vector3d pbRF, Eigen::Vector3d pbRH, 
                            Eigen::Vector3d vbLF, Eigen::Vector3d vbLH,
                            Eigen::Vector3d vbRF, Eigen::Vector3d vbRH,
                            double wLF_, double wLH_, double wRF_, double wRH_)
    {

        
        vwb_LF.noalias() = -wedge(omegab)  * pbLF - vbLF;
        vwb_LF =  Rwb * vwb_LF;
        vwb_LH.noalias() = -wedge(omegab)  * pbLH - vbLH;
        vwb_LH =  Rwb * vwb_LH;

        vwb_RF.noalias() = -wedge(omegab)  * pbRF - vbRF;
        vwb_RF =  Rwb * vwb_RF;
        vwb_RH.noalias() = -wedge(omegab)  * pbRH - vbRH;
        vwb_RH =  Rwb * vwb_RH;


        vwb  = wLF_ * vwb_LF;
        vwb += wLH_ * vwb_LH;
        vwb += wRF_ * vwb_RF;
        vwb += wRH_ * vwb_RH;

   
    }

    Eigen::Matrix3d getVelocityCovariance()
    {

        return vwb_cov;
    }

    void computeLegKCFS(Eigen::Matrix3d Rwb, Eigen::Matrix3d RbLF, Eigen::Matrix3d RbLH, Eigen::Matrix3d RbRF, Eigen::Matrix3d RbRH,
                        Eigen::Vector3d omegawb, Eigen::Vector3d omegabLF, Eigen::Vector3d omegabLH, Eigen::Vector3d omegabRF, Eigen::Vector3d omegabRH,
                        Eigen::Vector3d pbLF, Eigen::Vector3d pbLH, Eigen::Vector3d pbRF, Eigen::Vector3d pbRH,
                        Eigen::Vector3d vbLF, Eigen::Vector3d vbLH, Eigen::Vector3d vbRF, Eigen::Vector3d vbRH)
    {
        RwLF = Rwb * RbLF;
        RwLH = Rwb * RbLH;
        RwRF = Rwb * RbRF;
        RwRH = Rwb * RbRH;

        omegawLF = omegawb + Rwb * omegabLF;
        omegawLH = omegawb + Rwb * omegabLH;

        omegawRF = omegawb + Rwb * omegabRF;
        omegawRH = omegawb + Rwb * omegabRH;  

        vwLH = vwb + wedge(omegawb) * Rwb * pbLH + Rwb * vbLH;
        vwLF = vwb + wedge(omegawb) * Rwb * pbLF + Rwb * vbLF;

        vwRF = vwb + wedge(omegawb) * Rwb * pbRF + Rwb * vbRF;
        vwRH = vwb + wedge(omegawb) * Rwb * pbRH + Rwb * vbRH;
    }

    Eigen::Vector3d getLFFootLinearVel()
    {
        return vwLF;
    }
    Eigen::Vector3d getLHFootLinearVel()
    {
        return vwLH;
    }
    Eigen::Vector3d getRFFootLinearVel()
    {
        return vwRF;
    }
    Eigen::Vector3d getRHFootLinearVel()
    {
        return vwRH;
    }

    Eigen::Vector3d getLFFootAngularVel()
    {
        return omegawLF;
    }
    Eigen::Vector3d getLHFootAngularVel()
    {
        return omegawLH;
    }

    Eigen::Vector3d getRFFootAngularVel()
    {
        return omegawRF;
    }
    Eigen::Vector3d getRHFootAngularVel()
    {
        return omegawRH;
    }

    Eigen::Vector3d getLFFootIMVPPosition()
    {
        return LFpLmb;
    }

    Eigen::Vector3d getLHFootIMVPPosition()
    {
        return LHpLmb;
    }


    Eigen::Vector3d getRFFootIMVPPosition()
    {
        return RFpRmb;
    }
    
    Eigen::Vector3d getRHFootIMVPPosition()
    {
        return RHpRmb;
    }

    Eigen::Matrix3d getLFFootIMVPOrientation()
    {
        return RLFpLm;
    }

    Eigen::Matrix3d getLHFootIMVPOrientation()
    {
        return RLHpLm;
    }

    Eigen::Matrix3d getRFFootIMVPOrientation()
    {
        return RRFpRm;
    }

    Eigen::Matrix3d getRHFootIMVPOrientation()
    {
        return RRHpRm;
    }

    void computeIMVP()
    {
        LFomega = RwLF.transpose() * omegawLF;
        LHomega = RwLH.transpose() * omegawLH;

        RFomega = RwRF.transpose() * omegawRF;
        RHomega = RwRH.transpose() * omegawRH;

        double temp = Tm2 / (LFomega.squaredNorm() * Tm2 + 1.0);
        C1l = temp * wedge(LFomega);
        C2l = temp * (LFomega * LFomega.transpose() + 1.0 / Tm2 * Matrix3d::Identity());
        //IMVP Computations
        LFpLm = C2l * LFpLm;
        LFpLm = LFpLm + C1l * RwLF.transpose() * vwLF;


        temp = Tm2 / (LHomega.squaredNorm() * Tm2 + 1.0);
        C1l = temp * wedge(LHomega);
        C2l = temp * (LHomega * LHomega.transpose() + 1.0 / Tm2 * Matrix3d::Identity());
        //IMVP Computations
        LHpLm = C2l * LHpLm;
        LHpLm = LHpLm + C1l * RwLH.transpose() * vwLH;


        temp = Tm2 / (RFomega.squaredNorm() * Tm2 + 1.0);
        C1r = temp * wedge(RFomega);
        C2r = temp * (RFomega * RFomega.transpose() + 1.0 / Tm2 * Matrix3d::Identity());
        RFpRm = C2r * RFpRm;
        RFpRm = RFpRm + C1r * RwRF.transpose() * vwRF;

        temp = Tm2 / (RHomega.squaredNorm() * Tm2 + 1.0);
        C1r = temp * wedge(RHomega);
        C2r = temp * (RHomega * RHomega.transpose() + 1.0 / Tm2 * Matrix3d::Identity());
        RHpRm = C2r * RHpRm;
        RHpRm = RHpRm + C1r * RwRH.transpose() * vwRH;
    }


    void computeIMVPFT(Eigen::Vector3d LFf, Eigen::Vector3d LHf, Eigen::Vector3d RFf, Eigen::Vector3d RHf,
                       Eigen::Vector3d LFt, Eigen::Vector3d LHt, Eigen::Vector3d RFt, Eigen::Vector3d RHt)
    {
        LFomega = RwLF.transpose() * omegawLF;
        LHomega = RwLH.transpose() * omegawLH;
        RFomega = RwRF.transpose() * omegawRF;
        RHomega = RwRH.transpose() * omegawRH;

        wedgeLFf = wedge(LFf);
        wedgeLHf = wedge(LHf);
        wedgeRFf = wedge(RFf);
        wedgeRHf = wedge(RHf);

        //LF
        AL.noalias() = 1.0 / Tm2 * Matrix3d::Identity();
        AL.noalias() -= alpha1 * wedge(LFomega) * wedge(LFomega);
        AL.noalias() -= alpha3 / Tm3 * wedgeLFf * wedgeLFf;

        bL.noalias() = 1.0 / Tm2 * LFpLm;
        bL.noalias() += alpha1 * wedge(LFomega) * RwLF.transpose() * vwLF;
        bL.noalias() += alpha3 / Tm3 * (wedgeLFf * LFt - wedgeLFf * wedgeLFf * pLFf);

        LFpLm.noalias() = AL.inverse() * bL;

        //LH
        AL.noalias() = 1.0 / Tm2 * Matrix3d::Identity();
        AL.noalias() -= alpha1 * wedge(LHomega) * wedge(LHomega);
        AL.noalias() -= alpha3 / Tm3 * wedgeLHf * wedgeLHf;

        bL.noalias() = 1.0 / Tm2 * LHpLm;
        bL.noalias() += alpha1 * wedge(LHomega) * RwLH.transpose() * vwLH;
        bL.noalias() += alpha3 / Tm3 * (wedgeLHf * LHt - wedgeLHf * wedgeLHf * pLHf);

        LHpLm.noalias() = AL.inverse() * bL;


        //RF
        AR.noalias() = 1.0 / Tm2 * Matrix3d::Identity();
        AR.noalias() -= alpha1 * wedge(RFomega) * wedge(RFomega);
        AR.noalias() -= alpha3 / Tm3 * wedgeRFf * wedgeRFf;

        bR.noalias() = 1.0 / Tm2 * RFpRm;
        bR.noalias() += alpha1 * wedge(RFomega) * RwRF.transpose() * vwRF;
        bR.noalias() += alpha3 / Tm3 * (wedgeRFf * RFt - wedgeRFf * wedgeRFf * pRFf);

        RFpRm.noalias() = AR.inverse() * bR;

        //RH
        AR.noalias() = 1.0 / Tm2 * Matrix3d::Identity();
        AR.noalias() -= alpha1 * wedge(RHomega) * wedge(RHomega);
        AR.noalias() -= alpha3 / Tm3 * wedgeRHf * wedgeRHf;

        bR.noalias() = 1.0 / Tm2 * RHpRm;
        bR.noalias() += alpha1 * wedge(RHomega) * RwRH.transpose() * vwRH;
        bR.noalias() += alpha3 / Tm3 * (wedgeRHf * RHt - wedgeRHf * wedgeRHf * pRHf);

        RHpRm.noalias() = AR.inverse() * bR;

    }

    void computeDeadReckoning(Eigen::Matrix3d Rwb, Eigen::Matrix3d RbLF, Eigen::Matrix3d RbLH, Eigen::Matrix3d RbRF, Eigen::Matrix3d RbRH,
                              Eigen::Vector3d omegawb, Eigen::Vector3d bomegab, 
                              Eigen::Vector3d pbLF, Eigen::Vector3d pbRF, Eigen::Vector3d pbLH, Eigen::Vector3d pbRH,
                              Eigen::Vector3d vbLF, Eigen::Vector3d vbLH, Eigen::Vector3d vbRF, Eigen::Vector3d vbRH,
                              Eigen::Vector3d omegabLF, Eigen::Vector3d omegabLH, Eigen::Vector3d omegabRF, Eigen::Vector3d omegabRH,
                              double LFfz, double LHfz, double RFfz, double RHfz, 
                              Eigen::Vector3d LFf, Eigen::Vector3d LHf, Eigen::Vector3d RFf, Eigen::Vector3d RHf,
                              Eigen::Vector3d LFt, Eigen::Vector3d LHt, Eigen::Vector3d RFt, Eigen::Vector3d RHt )
    {

        //Compute Body position
        //Cropping the vertical GRF
        LFfz = cropGRF(LFfz);
        LHfz = cropGRF(LHfz);
        RFfz = cropGRF(RFfz);
        RHfz = cropGRF(RHfz);
        //GRF Coefficients
        wLF = (LFfz + ef) / (LFfz + LHfz + RFfz + RHfz + 4.0 * ef);
        wLH = (LHfz + ef) / (LFfz + LHfz + RFfz + RHfz + 4.0 * ef);
        wRH = (RHfz + ef) / (LFfz + LHfz + RFfz + RHfz + 4.0 * ef);
        wRF = (RFfz + ef) / (LFfz + LHfz + RFfz + RHfz + 4.0 * ef);

        computeBodyVelKCFS(Rwb,  bomegab, pbLF,  pbLH, pbRF, pbRH, vbLF, vbLH, vbLF,  vbLH, wLF,  wLH,  wRF,  wRH);
 


        computeLegKCFS(Rwb, RbLF, RbLH, RbRF, RbRH, omegawb, omegabLF, omegabLH, omegabRF, omegabRH, pbLF, pbLH, pbRF, pbRH, vbLF, vbLH, vbRF, vbRH);


        RLFpLm = RbLF;
        RRFpRm = RbRF;
        RLHpLm = RbLH;
        RRHpRm = RbRH;


        //computeIMVP();
        computeIMVPFT(LFf, LHf, RFf, RHf, LFt, LHt,  RFt, RHt);


        RFpRmb = RbRF * RFpRm + pbRF;
        RHpRmb = RbRH * RHpRm + pbRH;

        LFpLmb = RbLF * LFpLm + pbLF;
        LHpLmb = RbLH * LHpLm + pbLH;

 
        RFpRmb =  pbRF;
        RHpRmb =  pbRH;
        LHpLmb =  pbLH;
        LFpLmb =  pbLF;


        //Temp estimate of Leg position w.r.t Inertial Frame
        pwLF = pwLF_ - RwLF * LFpLm + RwLF_ * LFpLm;
        pwLH = pwLH_ - RwLH * LHpLm + RwLH_ * LHpLm;

        pwRF = pwRF_ - RwRF * RFpRm + RwRF_ * RFpRm;
        pwRH = pwRH_ - RwRH * RHpRm + RwRH_ * RHpRm;

        //Leg odometry with left foot
        pb_LF = pwLF - Rwb * pbLF;
        pb_LH = pwLH - Rwb * pbLH;
        //Leg odometry with right foot
        pb_RF = pwRF - Rwb * pbRF;
        pb_RH = pwRH - Rwb * pbRH;


        //Leg Odometry Estimate
        pwb_ = pwb;
        pwb = wLF * pb_LF + wRF * pb_RF + wLH * pb_LH + wRH * pb_RH;
        //Leg Position Estimate w.r.t Inertial Frame
        pwLF += pwb - pb_LF;
        pwLH += pwb - pb_LH;

        pwRF += pwb - pb_RF;
        pwRH += pwb - pb_RH;

        


        //Needed in the next iteration
        RwLF_ = RwLF;
        RwLH_ = RwLH;
        RwRF_ = RwRF;
        RwRH_ = RwRH;

        pwLF_ = pwLF;
        pwRF_ = pwRF;
        pwLH_ = pwLH;
        pwRH_ = pwRH;
        if (!firstrun)
            vwb = (pwb - pwb_) * freq;
        else
            firstrun = false;
       
    
        vwb_cov.noalias() =  wLF * (vwb_LF - vwb) * (vwb_LF - vwb).transpose();
        vwb_cov.noalias() += wLH * (vwb_LH - vwb) * (vwb_LH - vwb).transpose();
        vwb_cov.noalias() += wRF * (vwb_RF - vwb) * (vwb_RF - vwb).transpose();
        vwb_cov.noalias() += wRH * (vwb_RH - vwb) * (vwb_RH - vwb).transpose();

        

 
    }


    void computeDeadReckoningGEM(Eigen::Matrix3d Rwb, Eigen::Matrix3d RbLF, Eigen::Matrix3d RbLH, Eigen::Matrix3d RbRF, Eigen::Matrix3d RbRH,
                              Eigen::Vector3d omegawb, Eigen::Vector3d bomegab, 
                              Eigen::Vector3d pbLF, Eigen::Vector3d pbRF, Eigen::Vector3d pbLH, Eigen::Vector3d pbRH,
                              Eigen::Vector3d vbLF, Eigen::Vector3d vbLH, Eigen::Vector3d vbRF, Eigen::Vector3d vbRH,
                              Eigen::Vector3d omegabLF, Eigen::Vector3d omegabLH, Eigen::Vector3d omegabRF, Eigen::Vector3d omegabRH,
                              double wLF_, double wLH_, double wRF_, double wRH_, 
                              Eigen::Vector3d LFf, Eigen::Vector3d LHf, Eigen::Vector3d RFf, Eigen::Vector3d RHf,
                              Eigen::Vector3d LFt, Eigen::Vector3d LHt, Eigen::Vector3d RFt, Eigen::Vector3d RHt)
    {

        
        wLF = wLF_;
        wRF = wRF_;
        wLH = wLH_;
        wRH = wRH_;
       computeBodyVelKCFS(Rwb,  bomegab, pbLF,  pbLH, pbRF, pbRH, vbLF, vbLH, vbLF,  vbLH, wLF,  wLH,  wRF,  wRH);
 


        computeLegKCFS(Rwb, RbLF, RbLH, RbRF, RbRH, omegawb, omegabLF, omegabLH, omegabRF, omegabRH, pbLF, pbLH, pbRF, pbRH,vbLF, vbLH, vbRF, vbRH);


        RLFpLm = RbLF;
        RRFpRm = RbRF;
        RLHpLm = RbLH;
        RRHpRm = RbRH;


        //computeIMVP();
        computeIMVPFT(LFf, LHf, RFf, RHf, LFt, LHt,  RFt, RHt);


        RFpRmb = RbRF * RFpRm + pbRF;
        RHpRmb = RbRH * RHpRm + pbRH;

        LFpLmb = RbLF * LFpLm + pbLF;
        LHpLmb = RbLH * LHpLm + pbLH;

 
        RFpRmb =  pbRF;
        RHpRmb =  pbRH;
        LHpLmb =  pbLH;
        LFpLmb =  pbLF;


        //Temp estimate of Leg position w.r.t Inertial Frame
        pwLF = pwLF_ - RwLF * LFpLm + RwLF_ * LFpLm;
        pwLH = pwLH_ - RwLH * LHpLm + RwLH_ * LHpLm;

        pwRF = pwRF_ - RwRF * RFpRm + RwRF_ * RFpRm;
        pwRH = pwRH_ - RwRH * RHpRm + RwRH_ * RHpRm;

        //Leg odometry with left foot
        pb_LF = pwLF - Rwb * pbLF;
        pb_LH = pwLH - Rwb * pbLH;
        //Leg odometry with right foot
        pb_RF = pwRF - Rwb * pbRF;
        pb_RH = pwRH - Rwb * pbRH;


        //Leg Odometry Estimate
        pwb_ = pwb;
        pwb = wLF * pb_LF + wRF * pb_RF + wLH * pb_LH + wRH * pb_RH;
        //Leg Position Estimate w.r.t Inertial Frame
        pwLF += pwb - pb_LF;
        pwLH += pwb - pb_LH;

        pwRF += pwb - pb_RF;
        pwRH += pwb - pb_RH;

        


        //Needed in the next iteration
        RwLF_ = RwLF;
        RwLH_ = RwLH;
        RwRF_ = RwRF;
        RwRH_ = RwRH;

        pwLF_ = pwLF;
        pwRF_ = pwRF;
        pwLH_ = pwLH;
        pwRH_ = pwRH;
        if (!firstrun)
            vwb = (pwb - pwb_) * freq;
        else
            firstrun = false;
       
    
        vwb_cov.noalias() =  wLF * (vwb_LF - vwb) * (vwb_LF - vwb).transpose();
        vwb_cov.noalias() += wLH * (vwb_LH - vwb) * (vwb_LH - vwb).transpose();
        vwb_cov.noalias() += wRF * (vwb_RF - vwb) * (vwb_RF - vwb).transpose();
        vwb_cov.noalias() += wRH * (vwb_RH - vwb) * (vwb_RH - vwb).transpose();

        

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
