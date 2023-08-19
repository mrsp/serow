#include <Eigen/Dense>

namespace lie
{
    /** @fn Eigen::Matrix3d wedge(const Eigen::Vector3d& v)
     * 	@brief Computes the skew symmetric matrix of a 3-D vector
     *  @param v  3D Twist vector
     *  @return   3x3 skew symmetric representation
     */
    Eigen::Matrix3d wedge(const Eigen::Vector3d &v)
    {
        Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
        skew(0, 1) = -v(2);
        skew(0, 2) = v(1);
        skew(1, 2) = -v(0);
        skew(1, 0) = v(2);
        skew(2, 0) = -v(1);
        skew(2, 1) = v(0);
        return skew;
    }
    /** @fn Eigen::Vector3d vec(const Eigen::Matrix3d& M)
     *  @brief Computes the vector represation of a skew symmetric matrix
     *  @param M  3x3 skew symmetric matrix
     *  @return   3D Twist vector
     */
    Eigen::Vector3d vec(const Eigen::Matrix3d &M)
    {
        return Eigen::Vector3d(M(2, 1), M(0, 2), M(1, 0));
    }

    /** @fn Eigen::Matrix3d expMap(const Eigen::Vector3d& omega)
     *  @brief Computes the exponential map according to the Rodriquez Formula for component in SO(3)
     *  @param omega 3D twist in so(3) algebra
     *  @return 3x3 Rotation in  SO(3) group
     */
    Eigen::Matrix3d expMap(const Eigen::Vector3d &omega)
    {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity();
        const double omeganorm = omega.norm();

        if (omeganorm > std::numeric_limits<double>::epsilon())
        {
            Eigen::Matrix3d omega_skew = Eigen::Matrix3d::Zero();
            omega_skew = wedge(omega);
            res += omega_skew * (sin(omeganorm) / omeganorm);
            res += (omega_skew * omega_skew) * ((1.000 - cos(omeganorm)) / (omeganorm * omeganorm));
        }
        return res;
    }

    /** @fn Eigen::Vector3d logMap(const Eigen::Matrix3d& Rt)
     *  @brief Computes the logarithmic map for a component in SO(3) group
     *  @param Rt 3x3 Rotation in SO(3) group
     *  @return 3D twist in so(3) algebra
     */
    Eigen::Vector3d logMap(const Eigen::Matrix3d &Rt)
    {
        Eigen::Vector3d res = Eigen::Vector3d::Zero();
        const double costheta = (Rt.trace() - 1.0) / 2.0;
        const double theta = acos(costheta);

        if (std::fabs(theta) > std::numeric_limits<double>::epsilon())
        {
            Eigen::Matrix3d lnR = Rt - Rt.transpose();
            lnR *= theta / (2.0 * sin(theta));
            res = vec(lnR);
        }

        return res;
    }

    /**  Eigen::Vector3d logMap(const Eigen::Quaterniond& q)
     *  @brief Computes the logarithmic map for a component in SO(3) group
     *  @param q Quaternion in SO(3) group
     *  @return   3D twist in so(3) algebra
     */
    Eigen::Vector3d logMap(const Eigen::Quaterniond &q)
    {
        Eigen::Vector3d omega = Eigen::Vector3d::Zero();
        // Get the vector part
        Eigen::Vector3d qv = Eigen::Vector3d(q.x(), q.y(), q.z());
        qv *= (1.000 / qv.norm());
        omega = qv * (2.0 * std::acos(q.w() / q.norm()));
        if (std::isnan(omega(0) + omega(1) + omega(2)))
        {
            omega = Eigen::Vector3d::Zero();
        }
        return omega;
    }

    Eigen::Matrix3d plus(const Eigen::Matrix3d R, const Eigen::Vector3d tau)
    {
        return R * expMap(tau);
    }

    Eigen::Vector3d minus(const Eigen::Matrix3d& R0, const Eigen::Matrix3d& R1)
    {
        return logMap(R0.transpose() * R1);
    }

    Eigen::Vector3d minus(const Eigen::Quaterniond& q0, const Eigen::Queternion& q1)
    {
        return logMap(q0.inverse() * q1);
    }
}