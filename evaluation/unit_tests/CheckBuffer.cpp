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
#include <serow/Serow.hpp>

namespace serow {

class ImuMeasurementBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test IMU measurements
        createTestMeasurements();
    }

    void createTestMeasurements() {
        // Create IMU measurements with different timestamps
        m1.timestamp = 1.0;
        m1.linear_acceleration = Eigen::Vector3d(1.0, 2.0, 3.0);
        m1.angular_velocity = Eigen::Vector3d(0.1, 0.2, 0.3);
        m1.orientation = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);  // Identity quaternion
        m1.angular_acceleration = Eigen::Vector3d(0.01, 0.02, 0.03);

        m2.timestamp = 2.0;
        m2.linear_acceleration = Eigen::Vector3d(2.0, 4.0, 6.0);
        m2.angular_velocity = Eigen::Vector3d(0.2, 0.4, 0.6);
        m2.orientation = Eigen::Quaterniond(0.7071, 0.0, 0.7071, 0.0);  // 90° rotation around Y
        m2.angular_acceleration = Eigen::Vector3d(0.02, 0.04, 0.06);

        m3.timestamp = 3.0;
        m3.linear_acceleration = Eigen::Vector3d(3.0, 6.0, 9.0);
        m3.angular_velocity = Eigen::Vector3d(0.3, 0.6, 0.9);
        m3.orientation = Eigen::Quaterniond(0.0, 0.0, 1.0, 0.0);  // 180° rotation around Z
        m3.angular_acceleration = Eigen::Vector3d(0.03, 0.06, 0.09);

        m4.timestamp = 4.0;
        m4.linear_acceleration = Eigen::Vector3d(4.0, 8.0, 12.0);
        m4.angular_velocity = Eigen::Vector3d(0.4, 0.8, 1.2);
        m4.orientation = Eigen::Quaterniond(0.7071, 0.0, 0.0, 0.7071);  // 90° rotation around X
        m4.angular_acceleration = Eigen::Vector3d(0.04, 0.08, 0.12);
    }

    serow::ImuMeasurement m1, m2, m3, m4;
};

TEST_F(ImuMeasurementBufferTest, Constructor) {
    serow::ImuMeasurementBuffer buffer(100);
    EXPECT_EQ(buffer.size(), 0);

    serow::ImuMeasurementBuffer buffer2(50);
    EXPECT_EQ(buffer2.size(), 0);
}

TEST_F(ImuMeasurementBufferTest, AddAndSize) {
    serow::ImuMeasurementBuffer buffer(3);

    EXPECT_EQ(buffer.size(), 0);

    buffer.add(m1);
    EXPECT_EQ(buffer.size(), 1);

    buffer.add(m2);
    EXPECT_EQ(buffer.size(), 2);

    buffer.add(m3);
    EXPECT_EQ(buffer.size(), 3);

    // Adding more should maintain max size
    buffer.add(m4);
    EXPECT_EQ(buffer.size(), 3);
}

TEST_F(ImuMeasurementBufferTest, Clear) {
    serow::ImuMeasurementBuffer buffer(10);

    buffer.add(m1);
    buffer.add(m2);
    EXPECT_EQ(buffer.size(), 2);

    buffer.clear();
    EXPECT_EQ(buffer.size(), 0);

    // Should be able to add after clearing
    buffer.add(m1);
    EXPECT_EQ(buffer.size(), 1);
}

TEST_F(ImuMeasurementBufferTest, IsSorted) {
    serow::ImuMeasurementBuffer buffer(10);

    // Empty buffer should be sorted
    EXPECT_TRUE(buffer.isSorted());

    // Single measurement should be sorted
    buffer.add(m1);
    EXPECT_TRUE(buffer.isSorted());

    // Multiple measurements in order should be sorted
    buffer.add(m2);
    buffer.add(m3);
    EXPECT_TRUE(buffer.isSorted());

    // Clear and add out of order
    buffer.clear();
    buffer.add(m3);  // timestamp 3.0
    buffer.add(m1);  // timestamp 1.0
    buffer.add(m2);  // timestamp 2.0
    EXPECT_FALSE(buffer.isSorted());
}

TEST_F(ImuMeasurementBufferTest, GetTimeRange) {
    serow::ImuMeasurementBuffer buffer(10);

    // Empty buffer should return nullopt
    auto time_range = buffer.getTimeRange();
    EXPECT_FALSE(time_range.has_value());

    // Single measurement
    buffer.add(m1);
    time_range = buffer.getTimeRange();
    EXPECT_TRUE(time_range.has_value());
    EXPECT_DOUBLE_EQ(time_range->first, 1.0);
    EXPECT_DOUBLE_EQ(time_range->second, 1.0);

    // Multiple measurements in order
    buffer.add(m2);
    buffer.add(m3);
    time_range = buffer.getTimeRange();
    EXPECT_TRUE(time_range.has_value());
    EXPECT_DOUBLE_EQ(time_range->first, 1.0);
    EXPECT_DOUBLE_EQ(time_range->second, 3.0);

    // Out of order measurements
    buffer.clear();
    buffer.add(m3);  // timestamp 3.0
    buffer.add(m1);  // timestamp 1.0
    buffer.add(m2);  // timestamp 2.0
    time_range = buffer.getTimeRange();
    EXPECT_TRUE(time_range.has_value());
    EXPECT_DOUBLE_EQ(time_range->first, 1.0);
    EXPECT_DOUBLE_EQ(time_range->second, 3.0);
}

TEST_F(ImuMeasurementBufferTest, IsTimestampInRange) {
    serow::ImuMeasurementBuffer buffer(10);

    // Empty buffer should return false
    EXPECT_FALSE(buffer.isTimestampInRange(1.0));

    buffer.add(m1);  // timestamp 1.0
    buffer.add(m3);  // timestamp 3.0

    // Timestamps within range
    EXPECT_TRUE(buffer.isTimestampInRange(1.0));
    EXPECT_TRUE(buffer.isTimestampInRange(2.0));
    EXPECT_TRUE(buffer.isTimestampInRange(3.0));

    // Timestamps outside range
    EXPECT_FALSE(buffer.isTimestampInRange(0.5));
    EXPECT_FALSE(buffer.isTimestampInRange(3.5));

    // Timestamps within tolerance
    EXPECT_TRUE(buffer.isTimestampInRange(0.995, 0.01));  // Within 10ms tolerance
    EXPECT_TRUE(buffer.isTimestampInRange(3.005, 0.01));  // Within 10ms tolerance
}

TEST_F(ImuMeasurementBufferTest, GetExactTimestamp) {
    serow::ImuMeasurementBuffer buffer(10);

    // Empty buffer should return nullopt
    auto result = buffer.get(1.0);
    EXPECT_FALSE(result.has_value());

    buffer.add(m1);  // timestamp 1.0
    buffer.add(m2);  // timestamp 2.0
    buffer.add(m3);  // timestamp 3.0

    // Get exact timestamps
    result = buffer.get(1.0);
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->timestamp, 1.0);

    result = buffer.get(2.0);
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->timestamp, 2.0);

    result = buffer.get(3.0);
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->timestamp, 3.0);

    // Timestamps outside range should return nullopt
    result = buffer.get(0.5);
    EXPECT_FALSE(result.has_value());

    result = buffer.get(3.5);
    EXPECT_FALSE(result.has_value());
}

TEST_F(ImuMeasurementBufferTest, GetInterpolated) {
    serow::ImuMeasurementBuffer buffer(10);

    buffer.add(m1);  // timestamp 1.0
    buffer.add(m3);  // timestamp 3.0

    // Interpolate at timestamp 2.0 (middle) with reasonable tolerance
    auto result = buffer.get(2.0, 2.0);  // Allow up to 2.0 second difference
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->timestamp, 2.0);

    // Check interpolated values
    Eigen::Vector3d expected_acc = (m1.linear_acceleration + m3.linear_acceleration) / 2.0;
    EXPECT_TRUE(result->linear_acceleration.isApprox(expected_acc, 1e-6));

    Eigen::Vector3d expected_vel = (m1.angular_velocity + m3.angular_velocity) / 2.0;
    EXPECT_TRUE(result->angular_velocity.isApprox(expected_vel, 1e-6));

    // Interpolate at timestamp 1.5 with reasonable tolerance
    result = buffer.get(1.5, 2.0);  // Allow up to 2.0 second difference
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->timestamp, 1.5);

    // Check that interpolation is within tolerance
    // At timestamp 1.5, alpha = (1.5 - 1.0) / (3.0 - 1.0) = 0.25
    Eigen::Vector3d expected_acc_15 =
        m1.linear_acceleration + 0.25 * (m3.linear_acceleration - m1.linear_acceleration);
    EXPECT_TRUE(result->linear_acceleration.isApprox(expected_acc_15, 1e-6));
}

TEST_F(ImuMeasurementBufferTest, GetClosest) {
    serow::ImuMeasurementBuffer buffer(10);

    // Empty buffer should return nullopt
    auto result = buffer.getClosest(1.0);
    EXPECT_FALSE(result.has_value());

    buffer.add(m1);  // timestamp 1.0
    buffer.add(m3);  // timestamp 3.0

    // Get closest to existing timestamps
    result = buffer.getClosest(1.0);
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->timestamp, 1.0);

    result = buffer.getClosest(3.0);
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->timestamp, 3.0);

    // Get closest to intermediate timestamps with reasonable tolerance
    result = buffer.getClosest(1.8, 2.0);  // Closer to 1.0, allow up to 2.0 second difference
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->timestamp, 1.0);

    result = buffer.getClosest(2.2, 2.0);  // Closer to 3.0, allow up to 2.0 second difference
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->timestamp, 3.0);

    // Test with max_time_diff constraint
    result = buffer.getClosest(5.0, 0.5);  // 5.0 is 2.0 away from 3.0, tolerance 0.5
    EXPECT_FALSE(result.has_value());

    result = buffer.getClosest(5.0, 2.5);  // 5.0 is 2.0 away from 3.0, tolerance 2.5
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->timestamp, 3.0);
}

TEST_F(ImuMeasurementBufferTest, MaxSizeEnforcement) {
    serow::ImuMeasurementBuffer buffer(2);

    buffer.add(m1);
    buffer.add(m2);
    EXPECT_EQ(buffer.size(), 2);

    // Adding third measurement should remove the oldest (m1)
    buffer.add(m3);
    EXPECT_EQ(buffer.size(), 2);

    // Check that m1 was removed
    auto result = buffer.get(1.0);
    EXPECT_FALSE(result.has_value());

    // Check that m2 and m3 are still there
    result = buffer.get(2.0);
    EXPECT_TRUE(result.has_value());

    result = buffer.get(3.0);
    EXPECT_TRUE(result.has_value());
}

TEST_F(ImuMeasurementBufferTest, EdgeCases) {
    serow::ImuMeasurementBuffer buffer(10);

    // Test with very small time differences
    serow::ImuMeasurement m_small1, m_small2;
    m_small1.timestamp = 1.0;
    m_small1.linear_acceleration = Eigen::Vector3d::Zero();
    m_small1.angular_velocity = Eigen::Vector3d::Zero();
    m_small1.orientation = Eigen::Quaterniond::Identity();
    m_small1.angular_acceleration = Eigen::Vector3d::Zero();

    m_small2.timestamp = 1.0 + 1e-10;  // Very small difference
    m_small2.linear_acceleration = Eigen::Vector3d::Ones();
    m_small2.angular_velocity = Eigen::Vector3d::Ones();
    m_small2.orientation = Eigen::Quaterniond::Identity();
    m_small2.angular_acceleration = Eigen::Vector3d::Ones();

    buffer.add(m_small1);
    buffer.add(m_small2);

    // Should be able to get both measurements
    auto result1 = buffer.get(1.0);
    auto result2 = buffer.get(1.0 + 1e-10);
    EXPECT_TRUE(result1.has_value());
    EXPECT_TRUE(result2.has_value());
}

TEST_F(ImuMeasurementBufferTest, QuaternionInterpolation) {
    serow::ImuMeasurementBuffer buffer(10);

    // Create measurements with different orientations
    serow::ImuMeasurement m_rot1, m_rot2;
    m_rot1.timestamp = 0.0;
    m_rot1.orientation = Eigen::Quaterniond::Identity();  // No rotation
    m_rot1.linear_acceleration = Eigen::Vector3d::Zero();
    m_rot1.angular_velocity = Eigen::Vector3d::Zero();
    m_rot1.angular_acceleration = Eigen::Vector3d::Zero();

    m_rot2.timestamp = 1.0;
    m_rot2.orientation = Eigen::Quaterniond(0.0, 0.0, 0.0, 1.0);  // 180° rotation around Z
    m_rot2.linear_acceleration = Eigen::Vector3d::Zero();
    m_rot2.angular_velocity = Eigen::Vector3d::Zero();
    m_rot2.angular_acceleration = Eigen::Vector3d::Zero();

    buffer.add(m_rot1);
    buffer.add(m_rot2);

    // Interpolate at middle timestamp with reasonable tolerance
    auto result = buffer.get(0.5, 2.0);  // Allow up to 2.0 second difference
    EXPECT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(result->timestamp, 0.5);

    // Check that orientation is interpolated (should be 90° rotation around Z)
    Eigen::Quaterniond expected_rot(0.7071, 0.0, 0.0, 0.7071);
    EXPECT_TRUE(result->orientation.isApprox(expected_rot, 1e-3));
}

}  // namespace serow

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
