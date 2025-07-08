/**
 * Copyright (C) 2025 Stylianos Piperakis, Ownage Dynamics L.P.
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
#include <gtest/gtest.h>
#include <serow/lie.hpp>

TEST(SerowTests, SO3LieTest) {
    for (size_t i = 0; i < 1000; i++) {
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

        // Check the expMap operation of SO3
        Eigen::Vector3d w = Eigen::Vector3d::Random();
        w = lie::so3::wrapToSO3(w);
        Eigen::Matrix3d R_plus = lie::so3::plus(R, w);
        EXPECT_NEAR(R_plus.determinant(), 1.0, 1e-6);

        // Check the logMap operation of SO3
        Eigen::Vector3d w_minus = lie::so3::logMap(R_plus);
        EXPECT_TRUE(w.isApprox(w_minus));

        // Check the hat and vee operations of SO3
        const Eigen::Matrix3d R_hat = lie::so3::wedge(w);
        const Eigen::Vector3d w_vee = lie::so3::vec(R_hat);
        EXPECT_TRUE(w.isApprox(w_vee));

        // Check the plus and minus operations of SO3
        R = Eigen::Matrix3d::Identity();
        R_plus = lie::so3::plus(R, w);
        w_minus = lie::so3::minus(R_plus, R);
        EXPECT_TRUE(w.isApprox(w_minus));
    }
}