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
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <serow/NaiveLocalTerrainMapper.hpp>
#include <serow/LocalTerrainMapper.hpp>
#include <serow/common.hpp>

namespace serow {

class TerrainElevationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize both implementations with the same parameters
        terrain.initializeLocalMap(initial_height, initial_variance, min_variance);
        naive_terrain.initializeLocalMap(initial_height, initial_variance, min_variance);
    }

    // Test parameters
    const float initial_height = 0.0f;
    const float initial_variance = 1.0f;
    const float min_variance = 0.001f;

    LocalTerrainMapper terrain;
    NaiveLocalTerrainMapper naive_terrain;

    // Utility function to compare floating point values
    bool floatEqual(float a, float b, float epsilon = 1e-5f) {
        return std::abs(a - b) < epsilon;
    }

    // Utility function to compare ElevationCell objects
    bool elevationCellEqual(const ElevationCell& a, const ElevationCell& b, float epsilon = 1e-5f) {
        return floatEqual(a.height, b.height, epsilon) &&
            floatEqual(a.variance, b.variance, epsilon) && a.contact == b.contact &&
            a.updated == b.updated;
    }
};

// Test fast_mod function
TEST_F(TerrainElevationTest, FastMod) {
    EXPECT_EQ(fast_mod<16>(15), 15 % 16);
    EXPECT_EQ(fast_mod<16>(-1), -1 % 16);
    EXPECT_EQ(fast_mod<16>(16), 16 % 16);
    EXPECT_EQ(fast_mod<16>(-17), -17 % 16);
}

TEST_F(TerrainElevationTest, FastModRandom) {
    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(-1000000, 1000000);

    // Test with different powers of 2
    const std::vector<int> powers = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    for (int power : powers) {
        // Generate 100 random numbers for each power
        for (int i = 0; i < 100; ++i) {
            int64_t num = dist(gen);

            // Test with template parameter
            switch (power) {
                case 2:
                    EXPECT_EQ(fast_mod<2>(num), num % 2);
                    break;
                case 4:
                    EXPECT_EQ(fast_mod<4>(num), num % 4);
                    break;
                case 8:
                    EXPECT_EQ(fast_mod<8>(num), num % 8);
                    break;
                case 16:
                    EXPECT_EQ(fast_mod<16>(num), num % 16);
                    break;
                case 32:
                    EXPECT_EQ(fast_mod<32>(num), num % 32);
                    break;
                case 64:
                    EXPECT_EQ(fast_mod<64>(num), num % 64);
                    break;
                case 128:
                    EXPECT_EQ(fast_mod<128>(num), num % 128);
                    break;
                case 256:
                    EXPECT_EQ(fast_mod<256>(num), num % 256);
                    break;
                case 512:
                    EXPECT_EQ(fast_mod<512>(num), num % 512);
                    break;
                case 1024:
                    EXPECT_EQ(fast_mod<1024>(num), num % 1024);
                    break;
            }
        }
    }
}

TEST_F(TerrainElevationTest, FastModNegative) {
    // Test with different powers of 2
    const std::vector<int> powers = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    // Test specific negative numbers
    const std::vector<int64_t> negative_numbers = {
        -1,     -2,     -3,     -4,      -5,      -8,      -16,     -32,
        -64,    -128,   -256,   -512,    -1024,   -2048,   -4096,   -8192,
        -16384, -32768, -65536, -131072, -262144, -524288, -1048576};

    for (int power : powers) {
        for (int64_t num : negative_numbers) {
            // Test with template parameter
            switch (power) {
                case 2:
                    EXPECT_EQ(fast_mod<2>(num), num % 2);
                    break;
                case 4:
                    EXPECT_EQ(fast_mod<4>(num), num % 4);
                    break;
                case 8:
                    EXPECT_EQ(fast_mod<8>(num), num % 8);
                    break;
                case 16:
                    EXPECT_EQ(fast_mod<16>(num), num % 16);
                    break;
                case 32:
                    EXPECT_EQ(fast_mod<32>(num), num % 32);
                    break;
                case 64:
                    EXPECT_EQ(fast_mod<64>(num), num % 64);
                    break;
                case 128:
                    EXPECT_EQ(fast_mod<128>(num), num % 128);
                    break;
                case 256:
                    EXPECT_EQ(fast_mod<256>(num), num % 256);
                    break;
                case 512:
                    EXPECT_EQ(fast_mod<512>(num), num % 512);
                    break;
                case 1024:
                    EXPECT_EQ(fast_mod<1024>(num), num % 1024);
                    break;
            }
        }
    }
}

// Test normalize function
TEST_F(TerrainElevationTest, Normalize) {
    static auto normalize_fn = [](int x, int a, int b) -> int {
        int range = b - a + 1;
        int y = (x - a) % range;
        return (y < 0 ? y + range : y) + a;
    };

    EXPECT_EQ(normalize(15), normalize_fn(15, -map_dim, map_dim));
    EXPECT_EQ(normalize(-1), normalize_fn(-1, -map_dim, map_dim));
    EXPECT_EQ(normalize(16), normalize_fn(16, -map_dim, map_dim));
    EXPECT_EQ(normalize(-17), normalize_fn(-17, -map_dim, map_dim));
}

TEST_F(TerrainElevationTest, NormalizeRandomAndEdgeCases) {
    static auto normalize_fn = [](int x, int a, int b) -> int {
        int range = b - a + 1;
        int y = (x - a) % range;
        return (y < 0 ? y + range : y) + a;
    };

    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(-1000000, 1000000);

    // Test random numbers
    for (int i = 0; i < 100; ++i) {
        int64_t num = dist(gen);
        EXPECT_EQ(normalize(num), normalize_fn(num, -half_map_dim, half_map_dim));
    }

    // Edge cases
    const std::vector<int64_t> edge_cases = {
        // Boundary values
        -map_dim,      // Lower bound
        map_dim,       // Upper bound
        -map_dim - 1,  // Just below lower bound
        map_dim + 1,   // Just above upper bound

        // Powers of 2
        -2048, -1024, -512, -256, -128, -64, -32, -16, -8, -4, -2, 2, 4, 8, 16, 32, 64, 128, 256,
        512, 1024, 2048,

        // Large numbers
        -1000000, -500000, -100000, -50000, -10000, 10000, 50000, 100000, 500000, 1000000,

        // Special values
        0, 1, -1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(),
        std::numeric_limits<int>::min() + 1, std::numeric_limits<int>::max() - 1};

    for (int64_t num : edge_cases) {
        EXPECT_EQ(normalize(num), normalize_fn(num, -half_map_dim, half_map_dim))
            << "Failed for number: " << num;
    }

    // Test sequences of numbers
    for (int i = -half_map_dim - 10; i <= half_map_dim + 10; ++i) {
        EXPECT_EQ(normalize(i), normalize_fn(i, -half_map_dim, half_map_dim))
            << "Failed for number: " << i;
    }
}

}  // namespace serow
