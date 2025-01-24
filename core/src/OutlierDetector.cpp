/**
 * Copyright (C) 2024 Stylianos Piperakis, Ownage Dynamics L.P.
 * Serow is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, version 3.
 *
 * Serow is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Serow. If not,
 * see <https://www.gnu.org/licenses/>.
 **/
#include "OutlierDetector.hpp"


namespace serow {

double OutlierDetector::computePsi(double xxx) {
    double result = 0;
    double xx, xx2, xx4;
    for (; xxx < 7; ++xxx) {
        result -= 1 / xxx;
    }

    xxx -= 1.0 / 2.0;
    xx = 1.0 / xxx;
    xx2 = xx * xx;
    xx4 = xx2 * xx2;
    result += std::log(xxx) + (1.0 / 24.0) * xx2 - (7.0 / 960.0) * xx4 +
              (31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4;
    return result;
}

void OutlierDetector::init() {
    zeta = 1.0;
    e_t = e_0;
    f_t = f_0;
}

void OutlierDetector::estimate(const Eigen::Matrix3d& BetaT, const Eigen::Matrix3d& R) {
    double efpsi = computePsi(e_t + f_t);
    double lnp = computePsi(e_t) - efpsi;
    double ln1_p = computePsi(f_t) - efpsi;

    double pzeta_1 = std::exp(lnp - 0.5 * (BetaT * R.inverse()).trace());
    double pzeta_0 = std::exp(ln1_p);

    // Normalization factor
    double norm_factor = 1.0 / (pzeta_1 + pzeta_0);

    // p(zeta) are now proper probabilities
    pzeta_1 = norm_factor * pzeta_1;
    pzeta_0 = norm_factor * pzeta_0;

    // Mean of Bernulli
    zeta = pzeta_1 / (pzeta_1 + pzeta_0);

    // Update epsilon and f
    e_t = e_0 + zeta;
    f_t = f_0 + 1.0 - zeta;
}


}  // namespace serow
