#include <eigen3/Eigen/Dense>
#include "serow/bodyVelCF.h"
namespace serow
{

class deadReckoning
{
  private:
    double Tm, Tm2, ef, wl, wr, mass, g, freq, GRF, Tm3, alpha1, alpha3;
    Eigen::Matrix3d C1l, C2l, C1r, C2r;
    Eigen::Vector3d RpRm, LpLm;

    Eigen::Vector3d pwr, pwl, pwb, pwb_;
    Eigen::Vector3d vwr, vwl, vwb;
    Eigen::Matrix3d Rwr, Rwl;
    Eigen::Vector3d pwl_, pwr_;
    Eigen::Vector3d pb_l, pb_r;
    Eigen::Matrix3d Rwl_, Rwr_;
    Eigen::Vector3d vwbKCFS;
    Eigen::Vector3d Lomega, Romega;
    Eigen::Vector3d omegawl, omegawr;

    Eigen::Matrix3d AL, AR, wedgerf, wedgelf;
    Eigen::Vector3d bL, bR, plf, prf; //FT w.r.t Foot Frame;
    bool USE_CF, firstrun;
    bodyVelCF *bvcf;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    deadReckoning(Eigen::Vector3d pwl0, Eigen::Vector3d pwr0, Eigen::Matrix3d Rwl0, Eigen::Matrix3d Rwr0,
                  double mass_, double alpha1_ = 1.0, double alpha3_ = 0.01, double freq_ = 100.0, double g_ = 9.81, bool USE_CF_ = false, double freqvmin_ = 0.1, double freqvmax_ = 2.5,
                  Eigen::Vector3d plf_ = Eigen::Vector3d::Zero(), Eigen::Vector3d prf_ = Eigen::Vector3d::Zero())
    {
        firstrun = true;
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
        USE_CF = USE_CF_;
        plf = plf_;
        prf = prf_;
        RpRm = prf;
        LpLm = plf;
        alpha1 = alpha1_;
        alpha3 = alpha3_;
        Tm3 = (mass * mass * g * g) * Tm2;
        if (USE_CF)
            bvcf = new bodyVelCF(freq_, mass_, freqvmin_, freqvmax_, g_);
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
                            Eigen::Vector3d vbl, Eigen::Vector3d vbr, double lfz, double rfz)
    {

        /*
            if(lfz>rfz)
                vwbKCFS= -wedge(omegawb) * Rwb * pbl - Rwb * vbl;
            else
                vwbKCFS= -wedge(omegawb) * Rwb * pbr - Rwb * vbr;
            */

        //Cropping the vertical GRF
        lfz = cropGRF(lfz);
        rfz = cropGRF(rfz);

        //GRF Coefficients
        wl = (lfz + ef) / (lfz + rfz + 2.0 * ef);
        wr = (rfz + ef) / (lfz + rfz + 2.0 * ef);

        vwbKCFS.noalias() = wl * (-wedge(omegawb) * Rwb * pbl - Rwb * vbl);
        vwbKCFS.noalias() += wr * (-wedge(omegawb) * Rwb * pbr - Rwb * vbr);

        //if(!USE_CF)
        vwb = vwbKCFS;
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
                              Eigen::Vector3d omegawb,
                              Eigen::Vector3d pbl, Eigen::Vector3d pbr,
                              Eigen::Vector3d vbl, Eigen::Vector3d vbr,
                              Eigen::Vector3d omegabl, Eigen::Vector3d omegabr,
                              double lfz, double rfz, Eigen::Vector3d acc, Eigen::Vector3d lf, Eigen::Vector3d rf, Eigen::Vector3d lt, Eigen::Vector3d rt)
    {

        //Compute Body position
        computeBodyVelKCFS(Rwb, omegawb, pbl, pbr, vbl, vbr, lfz, rfz);
        /*
            if(USE_CF)
            {
                double GRF=lfz+rfz;
                GRF = cropGRF(GRF);
                vwb=bvcf->filter(vwbKCFS, acc, GRF);
            }
        */
        computeLegKCFS(Rwb, Rbl, Rbr, omegawb, omegabl, omegabr, pbl, pbr, vbl, vbr);
        //computeIMVP();
        computeIMVPFT(lf, rf, lt, rt);

        //Temp estimate of Leg position w.r.t Inertial Frame
        pwl = pwl_ - Rwl * LpLm + Rwl_ * LpLm;
        pwr = pwr_ - Rwr * RpRm + Rwr_ * RpRm;

        //Leg odometry with left foot
        pb_l = pwl - Rwb * pbl;
        //Leg odometry with right foot
        pb_r = pwr - Rwb * pbr;

        //Cropping the vertical GRF
        lfz = cropGRF(lfz);
        rfz = cropGRF(rfz);

        //GRF Coefficients
        wl = (lfz + ef) / (lfz + rfz + 2.0 * ef);
        wr = (rfz + ef) / (lfz + rfz + 2.0 * ef);

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
