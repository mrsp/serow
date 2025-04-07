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

TEST_F(TerrainElevationTest, LocalTerrainMapperCoordinateConversions) {
    // Test the coordinate conversion functions
    // Test with valid global indices
    std::array<int, 2> valid_indices[] = {
        {0, 0},                    // Origin
        {100, 100},               // Positive indices
        {half_map_dim-1, 0},      // Just inside edge x
        {0, half_map_dim-1},      // Just inside edge y
        {-half_map_dim+1, 0},     // Just inside edge -x
        {0, -half_map_dim+1}      // Just inside edge -y
    };

    for (const auto& idx : valid_indices) {
        EXPECT_TRUE(terrain.inside(idx)) << "Failed for index: " << idx[0] << ", " << idx[1];
    }

    for (const auto& idx : valid_indices) {
        const auto local_idx = terrain.globalIndexToLocalIndex(idx);
        const auto global_idx = terrain.localIndexToGlobalIndex(local_idx);
        EXPECT_EQ(idx[0], global_idx[0]) << "Failed x conversion for index: " << idx[0] << ", " << idx[1];
        EXPECT_EQ(idx[1], global_idx[1]) << "Failed y conversion for index: " << idx[0] << ", " << idx[1];
    }
    
    // Test with invalid global indices
    std::array<int, 2> invalid_indices[] = {
        {half_map_dim+1, 0},      // Just outside edge x
        {0, half_map_dim+1},      // Just outside edge y
        {-half_map_dim-1, 0},     // Just outside edge -x
        {0, -half_map_dim-1}      // Just outside edge -y
    };

    for (const auto& idx : invalid_indices) {
        EXPECT_FALSE(terrain.inside(idx)) << "Failed for index: " << idx[0] << ", " << idx[1];
    }

    // Test with valid map locations
    std::array<float, 2> valid_locations[] = {
        {0.0f, 0.0f},                    // Origin
        {1.0f, 1.0f},                    // 1 meter in x and y
        {-1.0f, -1.0f},                  // -1 meter in x and y
        {(half_map_dim-1) * resolution, 0.0f},  // Just inside edge x
        {0.0f, (half_map_dim-1) * resolution},  // Just inside edge y
        {(-half_map_dim+1) * resolution, 0.0f}, // Just inside edge -x
        {0.0f, (-half_map_dim+1) * resolution}  // Just inside edge -y
    };

    for (const auto& loc : valid_locations) {
        EXPECT_TRUE(terrain.inside(loc)) << "Failed for location: " << loc[0] << ", " << loc[1];
        
        // Test coordinate conversion
        const auto idx = terrain.locationToGlobalIndex(loc);
        const auto loc_back = terrain.globalIndexToLocation(idx);
        EXPECT_NEAR(loc[0], loc_back[0], 1e-5f) << "Failed x conversion for location: " << loc[0] << ", " << loc[1];
        EXPECT_NEAR(loc[1], loc_back[1], 1e-5f) << "Failed y conversion for location: " << loc[0] << ", " << loc[1];
    }

    // Test with invalid map locations
    std::array<float, 2> invalid_locations[] = {
        {(half_map_dim+1) * resolution, 0.0f},      // Just outside edge x
        {0.0f, (half_map_dim+1) * resolution},      // Just outside edge y
        {(-half_map_dim-1) * resolution, 0.0f},     // Just outside edge -x
        {0.0f, (-half_map_dim-1) * resolution}      // Just outside edge -y
    };

    for (const auto& loc : invalid_locations) {
        EXPECT_FALSE(terrain.inside(loc)) << "Failed for location: " << loc[0] << ", " << loc[1];
    }

    // Test with valid hash ids
    for (const auto& idx : valid_indices) {
        const int hash_id = terrain.globalIndexToHashId(idx);
        EXPECT_TRUE(terrain.isHashIdValid(hash_id)) << "Failed for local index: " << idx[0] << ", " << idx[1];
        
        // Test hash ID to local index conversion
        const auto idx_back = terrain.hashIdToGlobalIndex(hash_id);
        EXPECT_EQ(idx[0], idx_back[0]) << "Failed x conversion for hash_id: " << hash_id;
        EXPECT_EQ(idx[1], idx_back[1]) << "Failed y conversion for hash_id: " << hash_id;
    }

    // Test with invalid hash ids
    const int invalid_hash_ids[] = {
        -1,                   // Negative
        map_size,             // Equal to size
        map_size + 1,         // Just above size
        map_size * 2,         // Way above size
        std::numeric_limits<int>::max(),  // Max int
        std::numeric_limits<int>::min()   // Min int
    };

    for (const int hash_id : invalid_hash_ids) {
        EXPECT_FALSE(terrain.isHashIdValid(hash_id)) << "Failed for hash_id: " << hash_id;
    }
}

TEST_F(TerrainElevationTest, TerrainMapperRecentering) {
    //Test recentering with valid origin
    std::array<float, 2> valid_origin = {1.0f, 1.0f};
    terrain.recenter(valid_origin);
    naive_terrain.recenter(valid_origin);
    EXPECT_EQ(terrain.getMapOrigin(), valid_origin);
    EXPECT_EQ(naive_terrain.getMapOrigin(), valid_origin);

    // Update a few cells in the map and check if they are updated and that they are consistent 
    // after recentering
    std::array<float, 2> update_location = {0.1f, 0.1f};
    float update_height = 12.0f;
    float update_variance = 3.1f;
    const auto set_cell = ElevationCell(update_height, update_variance);
    EXPECT_TRUE(terrain.setElevation(update_location, set_cell));
    EXPECT_TRUE(naive_terrain.setElevation(update_location, set_cell));

    const auto cell = terrain.getElevation(update_location);
    const auto naive_cell = naive_terrain.getElevation(update_location);
    std::cout << "cell: " << cell.value().height << ", " << cell.value().variance << std::endl;
    std::cout << "naive cell: " << naive_cell.value().height << ", " << naive_cell.value().variance << std::endl;
    EXPECT_TRUE(cell.has_value());
    EXPECT_TRUE(naive_cell.has_value());
    EXPECT_TRUE(elevationCellEqual(cell.value(), naive_cell.value()));

    // Test recentering again with valid origin
    std::array<float, 2> new_valid_origin = {1.5f, 1.5f};
    terrain.recenter(new_valid_origin);
    naive_terrain.recenter(new_valid_origin);
    EXPECT_EQ(terrain.getMapOrigin(), new_valid_origin);
    EXPECT_EQ(naive_terrain.getMapOrigin(), new_valid_origin);

    // Check if the cells are still consistent
    // Compute the shifted update location relative to the new origin
    std::array<float, 2> shift = {
        new_valid_origin[0] - valid_origin[0],
        new_valid_origin[1] - valid_origin[1]
    };

    std::array<float, 2> shifted_update_location = {
        update_location[0] - shift[0],
        update_location[1] - shift[1]
    };
    std::cout << "shift: " << shift[0] << ", " << shift[1] << std::endl;
    std::cout << "location prior to shift: " << update_location[0] << ", " << update_location[1] << std::endl;
    std::cout << "location after shift: " << shifted_update_location[0] << ", " << shifted_update_location[1] << std::endl;
    const auto new_cell = terrain.getElevation(shifted_update_location);
    const auto new_naive_cell = naive_terrain.getElevation(shifted_update_location);
    EXPECT_TRUE(new_cell.has_value());
    EXPECT_TRUE(new_naive_cell.has_value());
    std::cout << "new cell: " << new_cell.value().height << ", " << new_cell.value().variance << std::endl;
    std::cout << "new naive cell: " << new_naive_cell.value().height << ", " << new_naive_cell.value().variance << std::endl;
    
    EXPECT_TRUE(elevationCellEqual(new_cell.value(), new_naive_cell.value()));
    EXPECT_TRUE(floatEqual(new_cell.value().height, update_height));
    EXPECT_TRUE(floatEqual(new_cell.value().variance, update_variance));
    EXPECT_TRUE(floatEqual(new_naive_cell.value().height, update_height));
    EXPECT_TRUE(floatEqual(new_naive_cell.value().variance, update_variance));
}

}  // namespace serow
    