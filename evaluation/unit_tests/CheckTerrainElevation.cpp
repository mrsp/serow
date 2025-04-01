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
#include <serow/NaiveTerrainElevation.hpp>
#include <serow/TerrainElevation.hpp>

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

    TerrainElevation terrain;
    NaiveTerrainElevation naive_terrain;

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
                case 2: EXPECT_EQ(fast_mod<2>(num), num % 2); break;
                case 4: EXPECT_EQ(fast_mod<4>(num), num % 4); break;
                case 8: EXPECT_EQ(fast_mod<8>(num), num % 8); break;
                case 16: EXPECT_EQ(fast_mod<16>(num), num % 16); break;
                case 32: EXPECT_EQ(fast_mod<32>(num), num % 32); break;
                case 64: EXPECT_EQ(fast_mod<64>(num), num % 64); break;
                case 128: EXPECT_EQ(fast_mod<128>(num), num % 128); break;
                case 256: EXPECT_EQ(fast_mod<256>(num), num % 256); break;
                case 512: EXPECT_EQ(fast_mod<512>(num), num % 512); break;
                case 1024: EXPECT_EQ(fast_mod<1024>(num), num % 1024); break;
            }
        }
    }
}

TEST_F(TerrainElevationTest, FastModNegative) {
    // Test with different powers of 2
    const std::vector<int> powers = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
    // Test specific negative numbers
    const std::vector<int64_t> negative_numbers = {
        -1, -2, -3, -4, -5, -8, -16, -32, -64, -128,
        -256, -512, -1024, -2048, -4096, -8192, -16384,
        -32768, -65536, -131072, -262144, -524288, -1048576
    };

    for (int power : powers) {
        for (int64_t num : negative_numbers) {
            // Test with template parameter
            switch (power) {
                case 2: EXPECT_EQ(fast_mod<2>(num), num % 2); break;
                case 4: EXPECT_EQ(fast_mod<4>(num), num % 4); break;
                case 8: EXPECT_EQ(fast_mod<8>(num), num % 8); break;
                case 16: EXPECT_EQ(fast_mod<16>(num), num % 16); break;
                case 32: EXPECT_EQ(fast_mod<32>(num), num % 32); break;
                case 64: EXPECT_EQ(fast_mod<64>(num), num % 64); break;
                case 128: EXPECT_EQ(fast_mod<128>(num), num % 128); break;
                case 256: EXPECT_EQ(fast_mod<256>(num), num % 256); break;
                case 512: EXPECT_EQ(fast_mod<512>(num), num % 512); break;
                case 1024: EXPECT_EQ(fast_mod<1024>(num), num % 1024); break;
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
        -map_dim,           // Lower bound
        map_dim,            // Upper bound
        -map_dim - 1,       // Just below lower bound
        map_dim + 1,        // Just above upper bound
        
        // Powers of 2
        -2048, -1024, -512, -256, -128, -64, -32, -16, -8, -4, -2,
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
        
        // Large numbers
        -1000000, -500000, -100000, -50000, -10000,
        10000, 50000, 100000, 500000, 1000000,
        
        // Special values
        0, 1, -1,
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::min() + 1,
        std::numeric_limits<int>::max() - 1
    };

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

// Test basic initialization
TEST_F(TerrainElevationTest, Initialization) {
    // Check if both implementations initialize the same by checking a point at origin
    std::array<float, 2> origin = {0.0f, 0.0f};
    auto elevation_terrain = terrain.getElevation(origin);
    auto elevation_naive = naive_terrain.getElevation(origin);

    ASSERT_TRUE(elevation_terrain.has_value());
    ASSERT_TRUE(elevation_naive.has_value());
    EXPECT_TRUE(elevationCellEqual(elevation_terrain.value(), elevation_naive.value()));
}

// Test location to global index conversion
TEST_F(TerrainElevationTest, LocationToGlobalIndex) {
    // Test various locations by checking if they're inside the map
    std::vector<float> test_locations = {-10.0f, -5.0f, -1.0f, -0.5f, -0.01f, 0.0f,
                                         0.01f,  0.5f,  1.0f,  5.0f,  10.0f};

    for (const auto& loc : test_locations) {
        std::array<float, 2> point = {loc, loc};
        bool inside_terrain = terrain.getElevation(point).has_value();
        bool inside_naive = naive_terrain.getElevation(point).has_value();

        // Log the difference for analysis
        std::cout << "Location: " << loc << ", TerrainElevation inside: " << inside_terrain
                  << ", NaiveTerrainElevation inside: " << inside_naive << std::endl;
    }
}

// Test inside function
TEST_F(TerrainElevationTest, Inside) {
    // Test various 2D locations
    std::vector<std::array<float, 2>> test_locations = {
        {0.0f, 0.0f}, {1.0f, 1.0f}, {-1.0f, -1.0f}, {5.0f, -5.0f}, {-5.0f, 5.0f}};

    for (const auto& loc : test_locations) {
        bool inside_terrain = terrain.getElevation(loc).has_value();
        bool inside_naive = naive_terrain.getElevation(loc).has_value();

        std::cout << "Location: [" << loc[0] << ", " << loc[1] << "], "
                  << "TerrainElevation inside: " << inside_terrain
                  << ", NaiveTerrainElevation inside: " << inside_naive << std::endl;
    }
}

// Test getting elevation at specific locations
TEST_F(TerrainElevationTest, GetElevation) {
    std::vector<std::array<float, 2>> test_locations = {
        {0.0f, 0.0f}, {0.5f, 0.5f}, {-0.5f, -0.5f}, {1.0f, -1.0f}, {-1.0f, 1.0f}};

    for (const auto& loc : test_locations) {
        auto elevation_terrain = terrain.getElevation(loc);
        auto elevation_naive = naive_terrain.getElevation(loc);

        if (elevation_terrain.has_value() && elevation_naive.has_value()) {
            EXPECT_TRUE(elevationCellEqual(elevation_terrain.value(), elevation_naive.value()));
        } else {
            EXPECT_EQ(elevation_terrain.has_value(), elevation_naive.has_value());
        }
    }
}

// Test updating elevations
TEST_F(TerrainElevationTest, UpdateElevation) {
    // A set of locations and heights to test
    std::vector<std::tuple<std::array<float, 2>, float, float>> test_updates = {
        {{0.0f, 0.0f}, 0.1f, 0.05f},
        {{0.5f, 0.5f}, 0.2f, 0.02f},
        {{-0.5f, -0.5f}, -0.1f, 0.01f},
        {{0.02f, 0.03f}, 0.15f, 0.03f},
        {{-0.02f, -0.03f}, -0.05f, 0.04f}};

    double timestamp = 0.0;
    // Apply updates to both implementations
    for (const auto& [loc, height, variance] : test_updates) {
        bool success_terrain = terrain.update(loc, height, variance, timestamp);
        bool success_naive = naive_terrain.update(loc, height, variance, timestamp);
        timestamp += 0.1;  // Increment timestamp for each update

        EXPECT_EQ(success_terrain, success_naive);

        if (success_terrain && success_naive) {
            auto elevation_terrain = terrain.getElevation(loc);
            auto elevation_naive = naive_terrain.getElevation(loc);

            ASSERT_TRUE(elevation_terrain.has_value());
            ASSERT_TRUE(elevation_naive.has_value());

            EXPECT_TRUE(elevationCellEqual(elevation_terrain.value(), elevation_naive.value()));

            // Also check nearby cells affected by the radius
            for (float dx = -radius; dx <= radius; dx += resolution) {
                for (float dy = -radius; dy <= radius; dy += resolution) {
                    if (dx == 0.0f && dy == 0.0f)
                        continue;

                    std::array<float, 2> nearby_loc = {loc[0] + dx, loc[1] + dy};
                    auto nearby_terrain = terrain.getElevation(nearby_loc);
                    auto nearby_naive = naive_terrain.getElevation(nearby_loc);

                    if (nearby_terrain.has_value() && nearby_naive.has_value()) {
                        EXPECT_TRUE(
                            elevationCellEqual(nearby_terrain.value(), nearby_naive.value()));
                    } else {
                        EXPECT_EQ(nearby_terrain.has_value(), nearby_naive.has_value());
                    }
                }
            }
        }
    }
}

// Test recenter operation (only implemented in TerrainElevation)
TEST_F(TerrainElevationTest, Recenter) {
    // First, set up some test data
    double timestamp = 0.0;
    std::vector<std::tuple<std::array<float, 2>, float, float>> test_points = {
        {{0.0f, 0.0f}, 0.1f, 0.05f}, {{0.5f, 0.5f}, 0.2f, 0.02f}, {{-0.5f, -0.5f}, -0.1f, 0.01f}};

    // Store initial data
    std::vector<std::pair<std::array<float, 2>, std::optional<ElevationCell>>> initial_data;
    for (const auto& [loc, height, variance] : test_points) {
        terrain.update(loc, height, variance, timestamp);
        initial_data.push_back({loc, terrain.getElevation(loc)});
        timestamp += 0.1;
    }

    // Perform recenter
    std::array<float, 2> new_center = {2.0f, 3.0f};
    terrain.recenter(new_center);

    // Verify the origin was updated correctly
    auto origin_terrain = terrain.getMapOrigin();
    std::cout << "After recenter, TerrainElevation origin: [" << origin_terrain[0] << ", "
              << origin_terrain[1] << "]" << std::endl;

    // Check that data is consistent after recenter
    for (const auto& [original_loc, original_elevation] : initial_data) {
        // Calculate the new location relative to the new center
        std::array<float, 2> new_loc = {original_loc[0] + new_center[0],
                                        original_loc[1] + new_center[1]};

        // Get the elevation at the new location
        auto new_elevation = terrain.getElevation(new_loc);

        // Verify the data is consistent
        ASSERT_TRUE(original_elevation.has_value());
        ASSERT_TRUE(new_elevation.has_value());
        EXPECT_TRUE(elevationCellEqual(original_elevation.value(), new_elevation.value()));
    }

    // Test that points outside the map are still outside
    std::array<float, 2> far_point = {100.0f, 100.0f};
    auto far_elevation = terrain.getElevation(far_point);
    EXPECT_FALSE(far_elevation.has_value());

    // Test that we can still update points in the new coordinate system
    std::array<float, 2> new_point = {new_center[0] + 0.5f, new_center[1] + 0.5f};
    bool update_success = terrain.update(new_point, 0.3f, 0.02f, timestamp);
    EXPECT_TRUE(update_success);

    auto updated_elevation = terrain.getElevation(new_point);
    ASSERT_TRUE(updated_elevation.has_value());
    EXPECT_TRUE(floatEqual(updated_elevation.value().height, 0.3f));
    EXPECT_TRUE(floatEqual(updated_elevation.value().variance, 0.02f));
}

// Test random updates
TEST_F(TerrainElevationTest, RandomUpdates) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> loc_dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> height_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> var_dist(0.01f, 0.1f);

    const int num_updates = 100;
    int successful_updates = 0;
    double timestamp = 0.0;

    for (int i = 0; i < num_updates; ++i) {
        std::array<float, 2> loc = {loc_dist(gen), loc_dist(gen)};
        float height = height_dist(gen);
        float variance = var_dist(gen);

        bool success_terrain = terrain.update(loc, height, variance, timestamp);
        bool success_naive = naive_terrain.update(loc, height, variance, timestamp);
        timestamp += 0.1;  // Increment timestamp for each update

        EXPECT_EQ(success_terrain, success_naive);

        if (success_terrain && success_naive) {
            successful_updates++;
            auto elevation_terrain = terrain.getElevation(loc);
            auto elevation_naive = naive_terrain.getElevation(loc);

            ASSERT_TRUE(elevation_terrain.has_value());
            ASSERT_TRUE(elevation_naive.has_value());

            EXPECT_TRUE(elevationCellEqual(elevation_terrain.value(), elevation_naive.value()));
        }
    }

    std::cout << "Successfully updated " << successful_updates << " out of " << num_updates
              << " random locations." << std::endl;
}

// Test edge cases
TEST_F(TerrainElevationTest, EdgeCases) {
    double timestamp = 0.0;
    // Test very small values
    std::array<float, 2> small_loc = {0.001f, 0.001f};
    terrain.update(small_loc, 0.0001f, 0.0001f, timestamp);
    naive_terrain.update(small_loc, 0.0001f, 0.0001f, timestamp);
    timestamp += 0.1;

    auto small_terrain = terrain.getElevation(small_loc);
    auto small_naive = naive_terrain.getElevation(small_loc);

    if (small_terrain.has_value() && small_naive.has_value()) {
        EXPECT_TRUE(elevationCellEqual(small_terrain.value(), small_naive.value()));
    } else {
        EXPECT_EQ(small_terrain.has_value(), small_naive.has_value());
    }

    // Test very large values
    std::array<float, 2> large_loc = {100.0f, 100.0f};
    bool large_success_terrain = terrain.update(large_loc, 10.0f, 1.0f, timestamp);
    bool large_success_naive = naive_terrain.update(large_loc, 10.0f, 1.0f, timestamp);

    EXPECT_EQ(large_success_terrain, large_success_naive);
}

// Comprehensive test that iterates through a grid of locations
TEST_F(TerrainElevationTest, ComprehensiveGridTest) {
    const float grid_start = -1.0f;
    const float grid_end = 1.0f;
    const float grid_step = 0.1f;

    int total_points = 0;
    int matching_points = 0;
    double timestamp = 0.0;

    for (float x = grid_start; x <= grid_end; x += grid_step) {
        for (float y = grid_start; y <= grid_end; y += grid_step) {
            std::array<float, 2> loc = {x, y};
            float height = 0.1f * (x + y);
            float variance = 0.01f;

            bool success_terrain = terrain.update(loc, height, variance, timestamp);
            bool success_naive = naive_terrain.update(loc, height, variance, timestamp);
            timestamp += 0.1;  // Increment timestamp for each update

            EXPECT_EQ(success_terrain, success_naive);
            total_points++;

            if (success_terrain && success_naive) {
                auto elevation_terrain = terrain.getElevation(loc);
                auto elevation_naive = naive_terrain.getElevation(loc);

                if (elevation_terrain.has_value() && elevation_naive.has_value() &&
                    elevationCellEqual(elevation_terrain.value(), elevation_naive.value())) {
                    matching_points++;
                }
            }
        }
    }

    std::cout << "Grid test: " << matching_points << " out of " << total_points
              << " points match between implementations." << std::endl;

    // Check if a significant portion of points match
    float match_percentage = static_cast<float>(matching_points) / total_points * 100.0f;
    std::cout << "Match percentage: " << match_percentage << "%" << std::endl;

    // Expect a high match percentage, but not necessarily 100% due to implementation differences
    EXPECT_GT(match_percentage, 99.0f);
}

// Test coordinate conversion and hashing
TEST_F(TerrainElevationTest, CoordinateConversionAndHashing) {
    // Test various locations
    std::vector<std::array<float, 2>> test_locations = {
        {0.0f, 0.0f},           // Origin
        {0.5f, 0.5f},           // Positive quadrant
        {-0.5f, -0.5f},         // Negative quadrant
        {1.0f, -1.0f},          // Mixed signs
        {-1.0f, 1.0f},          // Mixed signs
        {0.01f, 0.01f},         // Small values
        {-0.01f, -0.01f},       // Small negative values
        {5.0f, 5.0f},           // Large values
        {-5.0f, -5.0f}          // Large negative values
    };

    for (const auto& loc : test_locations) {
        // 1. Test location -> global index -> location
        auto global_idx = terrain.locationToGlobalIndex(loc);
        auto loc_from_global = terrain.globalIndexToLocation(global_idx);
        EXPECT_TRUE(floatEqual(loc[0], loc_from_global[0]));
        EXPECT_TRUE(floatEqual(loc[1], loc_from_global[1]));

        // 2. Test global index -> local index -> global index
        auto local_idx = terrain.globalIndexToLocalIndex(global_idx);
        auto global_from_local = terrain.localIndexToGlobalIndex(local_idx);
        EXPECT_EQ(global_idx[0], global_from_local[0]);
        EXPECT_EQ(global_idx[1], global_from_local[1]);

        // 3. Test local index -> hash ID -> local index
        auto hash_id = terrain.localIndexToHashId(local_idx);
        auto local_from_hash = terrain.hashIdToLocalIndex(hash_id);
        EXPECT_EQ(local_idx[0], local_from_hash[0]);
        EXPECT_EQ(local_idx[1], local_from_hash[1]);

        // 4. Test hash ID -> global index -> hash ID
        auto global_from_hash = terrain.hashIdToGlobalIndex(hash_id);
        auto hash_from_global = terrain.globalIndexToHashId(global_from_hash);
        EXPECT_EQ(hash_id, hash_from_global);

        // 5. Test hash ID -> location -> hash ID
        auto loc_from_hash = terrain.hashIdToLocation(hash_id);
        auto hash_from_loc = terrain.locationToHashId(loc_from_hash);
        EXPECT_EQ(hash_id, hash_from_loc);

        // 6. Test location -> hash ID -> location
        auto hash_from_loc_direct = terrain.locationToHashId(loc);
        auto loc_from_hash_direct = terrain.hashIdToLocation(hash_from_loc_direct);
        EXPECT_TRUE(floatEqual(loc[0], loc_from_hash_direct[0]));
        EXPECT_TRUE(floatEqual(loc[1], loc_from_hash_direct[1]));

        // 7. Verify hash ID is valid
        EXPECT_TRUE(terrain.isHashIdValid(hash_id));

        // 8. Verify the conversions work with the naive implementation
        auto naive_global_idx = naive_terrain.locationToGlobalIndex(loc);
        auto naive_loc = naive_terrain.globalIndexToLocation(naive_global_idx);
        EXPECT_TRUE(floatEqual(loc[0], naive_loc[0]));
        EXPECT_TRUE(floatEqual(loc[1], naive_loc[1]));
    }

    // Test edge cases
    std::vector<std::array<float, 2>> edge_cases = {
        {0.0f, 0.0f},           // Origin
        {map_dim * resolution, map_dim * resolution},           // Map boundary
        {-map_dim * resolution, -map_dim * resolution},         // Map boundary
        {map_dim * resolution + 1.0f, map_dim * resolution + 1.0f},  // Outside map
        {-map_dim * resolution - 1.0f, -map_dim * resolution - 1.0f} // Outside map
    };

    for (const auto& loc : edge_cases) {
        // Test location -> hash ID conversion for edge cases
        auto hash_id = terrain.locationToHashId(loc);
        if (terrain.inside(loc)) {
            EXPECT_TRUE(terrain.isHashIdValid(hash_id));
        } else {
            // For points outside the map, we expect the hash ID to be invalid
            // or the point to be wrapped around to a valid position
            auto wrapped_loc = terrain.hashIdToLocation(hash_id);
            EXPECT_TRUE(terrain.inside(wrapped_loc));
        }
    }
}

}  // namespace serow
