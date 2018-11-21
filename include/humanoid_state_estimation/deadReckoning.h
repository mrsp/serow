#include <eigen3/Eigen/Dense>


namespace serow{

    class deadReckoning{
        private:
            double Tm, Tm2, ef;
            Eigen::Matrix3d C1l, C2l, C1r, C2r;
            Eigen::Vector3d RpRm, LpLm;

            Eigen::Vector3d pwl_,pwr_;
            Eigen::Matrix3d Rwl_,Rwr_;
            Eigen::Vector3d vwbKCFS;

        public:
            deadReckoning(double Tm_, double ef_)
            {
                
                Tm = Tm_;
                Tm2 = Tm*Tm;
                ef = ef_;
                C1l.Zero();
                C2l.Zero();
                C1r.Zero();
                C2r.Zero();
                RpRm.Zero();
                LpLm.Zero();

            }

            void computeKCFS(Eigen::Vector3d pbs, Eigen::Vector3d vbs)
            {
                 vwbKCFS= -serow::skew(omegawb) * Rwb * pbs - Rwb * vbs;
            }
            void computeIMVP()
            {
                Lomega=Rwl.transpose()*omegawl;
                
                C1l = Tm2/(Lomega.squaredNorm()*Tm2+1.0f);
                C1l *= serow::skew(Lomega);

                C2l = C1l;
                C2l *= Lomega * Lomega.transpose() + 1.0f/Tm2 * Matrix3d::Identity();

                Romega=Rwr.transpose()*omegawr;
                
                C1r = Tm2/(Romega.squaredNorm()*Tm2+1.0f);
                C1r *= serow::skew(Romega);

                C2r = C1r;
                C2r *= Romega * Romega.transpose() + 1.0f/Tm2 * Matrix3d::Identity();

                LpLm = C2l * LpLm;

                LpLm += C1l * Rwl.transpose() * vwl;

                RpRm = C2r * RpRm;

                RpRm += C1r * Rwr.transpose() * vwr;
            }


            void computeDeadReckoning(Rwl,Rwr,pbl,pbr,lfz,rfz,)
            {
                pwl = pwl_ - Rwl*LpLm + Rwl_*LpLm;
                pwr = pwr_ - Rwr*RpRm + Rwr_*RpRm;


                pb_l = pwl - Rwb * pbl;
                pb_r = pwr - Rwb * pbr;

                lfz = cropGRF(lfz);
                rfz = cropGRF(rfz);

                wl = (lfz + ef)/(lfz+rfz+2.0*ef);
                wr = (rfz + ef)/(lfz+rfz+2.0*ef);

                pb = wl * pb_l + wr * pb_r;

                pwl += pb - pb_l;
                pwr += pb - pb_r;


            }

    }


}