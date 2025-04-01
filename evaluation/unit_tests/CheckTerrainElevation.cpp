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

// // Test fast_mod function
// TEST_F(TerrainElevationTest, FastMod) {
//     EXPECT_EQ(fast_mod<16>(15), 15 % 16);
//     EXPECT_EQ(fast_mod<16>(-1), -1 % 16);
//     EXPECT_EQ(fast_mod<16>(16), 16 % 16);
//     EXPECT_EQ(fast_mod<16>(-17), -17 % 16);
// }

// TEST_F(TerrainElevationTest, FastModRandom) {
//     // Set up random number generation
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<int64_t> dist(-1000000, 1000000);

//     // Test with different powers of 2
//     const std::vector<int> powers = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
//     for (int power : powers) {
//         // Generate 100 random numbers for each power
//         for (int i = 0; i < 100; ++i) {
//             int64_t num = dist(gen);
            
//             // Test with template parameter
//             switch (power) {
//                 case 2: EXPECT_EQ(fast_mod<2>(num), num % 2); break;
//                 case 4: EXPECT_EQ(fast_mod<4>(num), num % 4); break;
//                 case 8: EXPECT_EQ(fast_mod<8>(num), num % 8); break;
//                 case 16: EXPECT_EQ(fast_mod<16>(num), num % 16); break;
//                 case 32: EXPECT_EQ(fast_mod<32>(num), num % 32); break;
//                 case 64: EXPECT_EQ(fast_mod<64>(num), num % 64); break;
//                 case 128: EXPECT_EQ(fast_mod<128>(num), num % 128); break;
//                 case 256: EXPECT_EQ(fast_mod<256>(num), num % 256); break;
//                 case 512: EXPECT_EQ(fast_mod<512>(num), num % 512); break;
//                 case 1024: EXPECT_EQ(fast_mod<1024>(num), num % 1024); break;
//             }
//         }
//     }
// }

// TEST_F(TerrainElevationTest, FastModNegative) {
//     // Test with different powers of 2
//     const std::vector<int> powers = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
//     // Test specific negative numbers
//     const std::vector<int64_t> negative_numbers = {
//         -1, -2, -3, -4, -5, -8, -16, -32, -64, -128,
//         -256, -512, -1024, -2048, -4096, -8192, -16384,
//         -32768, -65536, -131072, -262144, -524288, -1048576
//     };

//     for (int power : powers) {
//         for (int64_t num : negative_numbers) {
//             // Test with template parameter
//             switch (power) {
//                 case 2: EXPECT_EQ(fast_mod<2>(num), num % 2); break;
//                 case 4: EXPECT_EQ(fast_mod<4>(num), num % 4); break;
//                 case 8: EXPECT_EQ(fast_mod<8>(num), num % 8); break;
//                 case 16: EXPECT_EQ(fast_mod<16>(num), num % 16); break;
//                 case 32: EXPECT_EQ(fast_mod<32>(num), num % 32); break;
//                 case 64: EXPECT_EQ(fast_mod<64>(num), num % 64); break;
//                 case 128: EXPECT_EQ(fast_mod<128>(num), num % 128); break;
//                 case 256: EXPECT_EQ(fast_mod<256>(num), num % 256); break;
//                 case 512: EXPECT_EQ(fast_mod<512>(num), num % 512); break;
//                 case 1024: EXPECT_EQ(fast_mod<1024>(num), num % 1024); break;
//             }
//         }
//     }
// }

// // Test normalize function
// TEST_F(TerrainElevationTest, Normalize) {
//     static auto normalize_fn = [](int x, int a, int b) -> int {
//         int range = b - a + 1;
//         int y = (x - a) % range;
//         return (y < 0 ? y + range : y) + a;
//     };

//     EXPECT_EQ(normalize(15), normalize_fn(15, -map_dim, map_dim));
//     EXPECT_EQ(normalize(-1), normalize_fn(-1, -map_dim, map_dim));
//     EXPECT_EQ(normalize(16), normalize_fn(16, -map_dim, map_dim));
//     EXPECT_EQ(normalize(-17), normalize_fn(-17, -map_dim, map_dim));
// }

// TEST_F(TerrainElevationTest, NormalizeRandomAndEdgeCases) {
//     static auto normalize_fn = [](int x, int a, int b) -> int {
//         int range = b - a + 1;
//         int y = (x - a) % range;
//         return (y < 0 ? y + range : y) + a;
//     };

//     // Set up random number generation
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<int64_t> dist(-1000000, 1000000);

//     // Test random numbers
//     for (int i = 0; i < 100; ++i) {
//         int64_t num = dist(gen);
//         EXPECT_EQ(normalize(num), normalize_fn(num, -half_map_dim, half_map_dim));
//     }

//     // Edge cases
//     const std::vector<int64_t> edge_cases = {
//         // Boundary values
//         -map_dim,           // Lower bound
//         map_dim,            // Upper bound
//         -map_dim - 1,       // Just below lower bound
//         map_dim + 1,        // Just above upper bound
        
//         // Powers of 2
//         -2048, -1024, -512, -256, -128, -64, -32, -16, -8, -4, -2,
//         2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
        
//         // Large numbers
//         -1000000, -500000, -100000, -50000, -10000,
//         10000, 50000, 100000, 500000, 1000000,
        
//         // Special values
//         0, 1, -1,
//         std::numeric_limits<int>::min(),
//         std::numeric_limits<int>::max(),
//         std::numeric_limits<int>::min() + 1,
//         std::numeric_limits<int>::max() - 1
//     };

//     for (int64_t num : edge_cases) {
//         EXPECT_EQ(normalize(num), normalize_fn(num, -half_map_dim, half_map_dim))
//             << "Failed for number: " << num;
//     }

//     // Test sequences of numbers
//     for (int i = -half_map_dim - 10; i <= half_map_dim + 10; ++i) {
//         EXPECT_EQ(normalize(i), normalize_fn(i, -half_map_dim, half_map_dim))
//             << "Failed for number: " << i;
//     }
// }

// // Test basic initialization
// TEST_F(TerrainElevationTest, Initialization) {
//     // Check if both implementations initialize the same by checking a point at origin
//     std::array<float, 2> origin = {0.0f, 0.0f};
//     auto elevation_terrain = terrain.getElevation(origin);
//     auto elevation_naive = naive_terrain.getElevation(origin);

//     ASSERT_TRUE(elevation_terrain.has_value());
//     ASSERT_TRUE(elevation_naive.has_value());
//     EXPECT_TRUE(elevationCellEqual(elevation_terrain.value(), elevation_naive.value()));
// }

// // Test location to global index conversion
// TEST_F(TerrainElevationTest, LocationToGlobalIndex) {
//     // Test various locations by checking if they're inside the map
//     std::vector<float> test_locations = {-10.0f, -5.0f, -1.0f, -0.5f, -0.01f, 0.0f,
//                                          0.01f,  0.5f,  1.0f,  5.0f,  10.0f};

//     for (const auto& loc : test_locations) {
//         std::array<float, 2> point = {loc, loc};
//         bool inside_terrain = terrain.getElevation(point).has_value();
//         bool inside_naive = naive_terrain.getElevation(point).has_value();

//         // Log the difference for analysis
//         std::cout << "Location: " << loc << ", TerrainElevation inside: " << inside_terrain
//                   << ", NaiveTerrainElevation inside: " << inside_naive << std::endl;
//     }
// }

// // Test inside function
// TEST_F(TerrainElevationTest, Inside) {
//     // Test various 2D locations
//     std::vector<std::array<float, 2>> test_locations = {
//         {0.0f, 0.0f}, {1.0f, 1.0f}, {-1.0f, -1.0f}, {5.0f, -5.0f}, {-5.0f, 5.0f}};

//     for (const auto& loc : test_locations) {
//         bool inside_terrain = terrain.getElevation(loc).has_value();
//         bool inside_naive = naive_terrain.getElevation(loc).has_value();

//         std::cout << "Location: [" << loc[0] << ", " << loc[1] << "], "
//                   << "TerrainElevation inside: " << inside_terrain
//                   << ", NaiveTerrainElevation inside: " << inside_naive << std::endl;
//     }
// }

// // Test getting elevation at specific locations
// TEST_F(TerrainElevationTest, GetElevation) {
//     std::vector<std::array<float, 2>> test_locations = {
//         {0.0f, 0.0f}, {0.5f, 0.5f}, {-0.5f, -0.5f}, {1.0f, -1.0f}, {-1.0f, 1.0f}};

//     for (const auto& loc : test_locations) {
//         auto elevation_terrain = terrain.getElevation(loc);
//         auto elevation_naive = naive_terrain.getElevation(loc);

//         if (elevation_terrain.has_value() && elevation_naive.has_value()) {
//             EXPECT_TRUE(elevationCellEqual(elevation_terrain.value(), elevation_naive.value()));
//         } else {
//             EXPECT_EQ(elevation_terrain.has_value(), elevation_naive.has_value());
//         }
//     }
// }

// // Test updating elevations
// TEST_F(TerrainElevationTest, UpdateElevation) {
//     // A set of locations and heights to test
//     std::vector<std::tuple<std::array<float, 2>, float, float>> test_updates = {
//         {{0.0f, 0.0f}, 0.1f, 0.05f},
//         {{0.5f, 0.5f}, 0.2f, 0.02f},
//         {{-0.5f, -0.5f}, -0.1f, 0.01f},
//         {{0.02f, 0.03f}, 0.15f, 0.03f},
//         {{-0.02f, -0.03f}, -0.05f, 0.04f}};

//     double timestamp = 0.0;
//     // Apply updates to both implementations
//     for (const auto& [loc, height, variance] : test_updates) {
//         bool success_terrain = terrain.update(loc, height, variance, timestamp);
//         bool success_naive = naive_terrain.update(loc, height, variance, timestamp);
//         timestamp += 0.1;  // Increment timestamp for each update

//         EXPECT_EQ(success_terrain, success_naive);

//         if (success_terrain && success_naive) {
//             auto elevation_terrain = terrain.getElevation(loc);
//             auto elevation_naive = naive_terrain.getElevation(loc);

//             ASSERT_TRUE(elevation_terrain.has_value());
//             ASSERT_TRUE(elevation_naive.has_value());

//             EXPECT_TRUE(elevationCellEqual(elevation_terrain.value(), elevation_naive.value()));

//             // Also check nearby cells affected by the radius
//             for (float dx = -radius; dx <= radius; dx += resolution) {
//                 for (float dy = -radius; dy <= radius; dy += resolution) {
//                     if (dx == 0.0f && dy == 0.0f)
//                         continue;

//                     std::array<float, 2> nearby_loc = {loc[0] + dx, loc[1] + dy};
//                     auto nearby_terrain = terrain.getElevation(nearby_loc);
//                     auto nearby_naive = naive_terrain.getElevation(nearby_loc);

//                     if (nearby_terrain.has_value() && nearby_naive.has_value()) {
//                         EXPECT_TRUE(
//                             elevationCellEqual(nearby_terrain.value(), nearby_naive.value()));
//                     } else {
//                         EXPECT_EQ(nearby_terrain.has_value(), nearby_naive.has_value());
//                     }
//                 }
//             }
//         }
//     }
// }

// // Test recenter operation (only implemented in TerrainElevation)
// TEST_F(TerrainElevationTest, Recenter) {
//     // First, set up some test data with known values
//     double timestamp = 0.0;
//     std::vector<std::tuple<std::array<float, 2>, float, float>> test_points = {
//         {{0.0f, 0.0f}, 0.1f, 0.05f},    // Origin point
//         {{0.5f, 0.5f}, 0.2f, 0.02f},    // Positive quadrant
//         {{-0.5f, -0.5f}, -0.1f, 0.01f}, // Negative quadrant
//         {{0.02f, 0.03f}, 0.15f, 0.03f}, // Small offset
//         {{-0.02f, -0.03f}, -0.05f, 0.04f} // Small negative offset
//     };

//     // Store initial data with their world coordinates
//     std::vector<std::pair<std::array<float, 2>, ElevationCell>> initial_data;
//     for (const auto& [loc, height, variance] : test_points) {
//         bool success = terrain.update(loc, height, variance, timestamp);
//         ASSERT_TRUE(success) << "Failed to update point at [" << loc[0] << ", " << loc[1] << "]";
        
//         auto elevation = terrain.getElevation(loc);
//         ASSERT_TRUE(elevation.has_value()) << "Failed to get elevation for point at [" << loc[0] << ", " << loc[1] << "]";
//         initial_data.push_back({loc, elevation.value()});
//         timestamp += 0.1;
//     }

//     // Get the current map origin before recentering
//     auto original_origin = terrain.getMapOrigin();

//     // Perform recenter
//     std::array<float, 2> new_center = {0.5f, 0.5f};
//     terrain.recenter(new_center);

//     // Verify the origin was updated correctly
//     auto new_origin = terrain.getMapOrigin();
//     EXPECT_TRUE(floatEqual(new_origin[0], new_center[0]));
//     EXPECT_TRUE(floatEqual(new_origin[1], new_center[1]));

//     // Check that data is consistent after recenter
//     for (const auto& [original_loc, original_elevation] : initial_data) {
//         // Convert original location to global indices
//         auto original_global_idx = terrain.locationToGlobalIndex(original_loc);
        
//         // Get the new center in global indices
//         auto new_center_global_idx = terrain.locationToGlobalIndex(new_center);
//         auto original_center_global_idx = terrain.locationToGlobalIndex(original_origin);
        
//         // Calculate the shift in global index space
//         std::array<int, 2> shift = {
//             new_center_global_idx[0] - original_center_global_idx[0],
//             new_center_global_idx[1] - original_center_global_idx[1]
//         };
        
//         // Apply shift to get new global index
//         std::array<int, 2> new_global_idx = {
//             original_global_idx[0] - shift[0],
//             original_global_idx[1] - shift[1]
//         };
        
//         // Convert back to location
//         auto new_loc = terrain.globalIndexToLocation(new_global_idx);

//         // Get the elevation at the new location
//         auto new_elevation = terrain.getElevation(new_loc);
//         ASSERT_TRUE(new_elevation.has_value()) 
//             << "Failed to get elevation after recenter for point at [" 
//             << new_loc[0] << ", " << new_loc[1] << "]";

//         // Compare the elevation values with more detailed error reporting
//         EXPECT_TRUE(floatEqual(original_elevation.height, new_elevation.value().height))
//             << "Height mismatch at [" << new_loc[0] << ", " << new_loc[1] 
//             << "]: original=" << original_elevation.height 
//             << ", new=" << new_elevation.value().height;
        
//         EXPECT_TRUE(floatEqual(original_elevation.variance, new_elevation.value().variance))
//             << "Variance mismatch at [" << new_loc[0] << ", " << new_loc[1] 
//             << "]: original=" << original_elevation.variance 
//             << ", new=" << new_elevation.value().variance;
        
//         EXPECT_EQ(original_elevation.contact, new_elevation.value().contact)
//             << "Contact flag mismatch at [" << new_loc[0] << ", " << new_loc[1] << "]";
        
//         EXPECT_EQ(original_elevation.updated, new_elevation.value().updated)
//             << "Updated flag mismatch at [" << new_loc[0] << ", " << new_loc[1] << "]";
//     }

//     // Test that points outside the map are still outside
//     std::array<float, 2> far_point = {100.0f, 100.0f};
//     auto far_elevation = terrain.getElevation(far_point);
//     EXPECT_FALSE(far_elevation.has_value());

//     // Test that we can still update points in the new coordinate system
//     std::array<float, 2> new_point = {new_center[0] + 0.5f, new_center[1] + 0.5f};
//     bool update_success = terrain.update(new_point, 0.3f, 0.02f, timestamp);
//     EXPECT_TRUE(update_success);

//     auto updated_elevation = terrain.getElevation(new_point);
//     ASSERT_TRUE(updated_elevation.has_value());
//     EXPECT_TRUE(floatEqual(updated_elevation.value().height, 0.3f))
//         << "Height mismatch after update: expected=0.3, got=" << updated_elevation.value().height;
//     EXPECT_TRUE(floatEqual(updated_elevation.value().variance, 0.02f))
//         << "Variance mismatch after update: expected=0.02, got=" << updated_elevation.value().variance;
// }

// // Test random updates
// TEST_F(TerrainElevationTest, RandomUpdates) {
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> loc_dist(-5.0f, 5.0f);
//     std::uniform_real_distribution<float> height_dist(-0.5f, 0.5f);
//     std::uniform_real_distribution<float> var_dist(0.01f, 0.1f);

//     const int num_updates = 100;
//     int successful_updates = 0;
//     double timestamp = 0.0;

//     for (int i = 0; i < num_updates; ++i) {
//         std::array<float, 2> loc = {loc_dist(gen), loc_dist(gen)};
//         float height = height_dist(gen);
//         float variance = var_dist(gen);

//         bool success_terrain = terrain.update(loc, height, variance, timestamp);
//         bool success_naive = naive_terrain.update(loc, height, variance, timestamp);
//         timestamp += 0.1;  // Increment timestamp for each update

//         EXPECT_EQ(success_terrain, success_naive);

//         if (success_terrain && success_naive) {
//             successful_updates++;
//             auto elevation_terrain = terrain.getElevation(loc);
//             auto elevation_naive = naive_terrain.getElevation(loc);

//             ASSERT_TRUE(elevation_terrain.has_value());
//             ASSERT_TRUE(elevation_naive.has_value());

//             EXPECT_TRUE(elevationCellEqual(elevation_terrain.value(), elevation_naive.value()));
//         }
//     }

//     std::cout << "Successfully updated " << successful_updates << " out of " << num_updates
//               << " random locations." << std::endl;
// }

// // Test edge cases
// TEST_F(TerrainElevationTest, EdgeCases) {
//     double timestamp = 0.0;
//     // Test very small values
//     std::array<float, 2> small_loc = {0.001f, 0.001f};
//     terrain.update(small_loc, 0.0001f, 0.0001f, timestamp);
//     naive_terrain.update(small_loc, 0.0001f, 0.0001f, timestamp);
//     timestamp += 0.1;

//     auto small_terrain = terrain.getElevation(small_loc);
//     auto small_naive = naive_terrain.getElevation(small_loc);

//     if (small_terrain.has_value() && small_naive.has_value()) {
//         EXPECT_TRUE(elevationCellEqual(small_terrain.value(), small_naive.value()));
//     } else {
//         EXPECT_EQ(small_terrain.has_value(), small_naive.has_value());
//     }

//     // Test very large values
//     std::array<float, 2> large_loc = {100.0f, 100.0f};
//     bool large_success_terrain = terrain.update(large_loc, 10.0f, 1.0f, timestamp);
//     bool large_success_naive = naive_terrain.update(large_loc, 10.0f, 1.0f, timestamp);

//     EXPECT_EQ(large_success_terrain, large_success_naive);
// }

// // Comprehensive test that iterates through a grid of locations
// TEST_F(TerrainElevationTest, ComprehensiveGridTest) {
//     const float grid_start = -1.0f;
//     const float grid_end = 1.0f;
//     const float grid_step = 0.1f;

//     int total_points = 0;
//     int matching_points = 0;
//     double timestamp = 0.0;

//     for (float x = grid_start; x <= grid_end; x += grid_step) {
//         for (float y = grid_start; y <= grid_end; y += grid_step) {
//             std::array<float, 2> loc = {x, y};
//             float height = 0.1f * (x + y);
//             float variance = 0.01f;

//             bool success_terrain = terrain.update(loc, height, variance, timestamp);
//             bool success_naive = naive_terrain.update(loc, height, variance, timestamp);
//             timestamp += 0.1;  // Increment timestamp for each update

//             EXPECT_EQ(success_terrain, success_naive);
//             total_points++;

//             if (success_terrain && success_naive) {
//                 auto elevation_terrain = terrain.getElevation(loc);
//                 auto elevation_naive = naive_terrain.getElevation(loc);

//                 if (elevation_terrain.has_value() && elevation_naive.has_value() &&
//                     elevationCellEqual(elevation_terrain.value(), elevation_naive.value())) {
//                     matching_points++;
//                 }
//             }
//         }
//     }

//     std::cout << "Grid test: " << matching_points << " out of " << total_points
//               << " points match between implementations." << std::endl;

//     // Check if a significant portion of points match
//     float match_percentage = static_cast<float>(matching_points) / total_points * 100.0f;
//     std::cout << "Match percentage: " << match_percentage << "%" << std::endl;

//     // Expect a high match percentage, but not necessarily 100% due to implementation differences
//     EXPECT_GT(match_percentage, 99.0f);
// }

// // Test coordinate conversion and hashing
// TEST_F(TerrainElevationTest, CoordinateConversionAndHashing) {
//     // Test various locations
//     std::vector<std::array<float, 2>> test_locations = {
//         {0.0f, 0.0f},           // Origin
//         {0.5f, 0.5f},           // Positive quadrant
//         {-0.5f, -0.5f},         // Negative quadrant
//         {1.0f, -1.0f},          // Mixed signs
//         {-1.0f, 1.0f},          // Mixed signs
//         {0.01f, 0.01f},         // Small values
//         {-0.01f, -0.01f},       // Small negative values
//         {5.0f, 5.0f},           // Large values
//         {-5.0f, -5.0f}          // Large negative values
//     };

//     for (const auto& loc : test_locations) {
//         // 1. Test location -> global index -> location
//         auto global_idx = terrain.locationToGlobalIndex(loc);
//         auto loc_from_global = terrain.globalIndexToLocation(global_idx);
//         EXPECT_TRUE(floatEqual(loc[0], loc_from_global[0]));
//         EXPECT_TRUE(floatEqual(loc[1], loc_from_global[1]));

//         // 2. Test global index -> local index -> global index
//         auto local_idx = terrain.globalIndexToLocalIndex(global_idx);
//         auto global_from_local = terrain.localIndexToGlobalIndex(local_idx);
//         EXPECT_EQ(global_idx[0], global_from_local[0]);
//         EXPECT_EQ(global_idx[1], global_from_local[1]);

//         // 3. Test local index -> hash ID -> local index
//         auto hash_id = terrain.localIndexToHashId(local_idx);
//         auto local_from_hash = terrain.hashIdToLocalIndex(hash_id);
//         EXPECT_EQ(local_idx[0], local_from_hash[0]);
//         EXPECT_EQ(local_idx[1], local_from_hash[1]);

//         // 4. Test hash ID -> global index -> hash ID
//         auto global_from_hash = terrain.hashIdToGlobalIndex(hash_id);
//         auto hash_from_global = terrain.globalIndexToHashId(global_from_hash);
//         EXPECT_EQ(hash_id, hash_from_global);

//         // 5. Test hash ID -> location -> hash ID
//         auto loc_from_hash = terrain.hashIdToLocation(hash_id);
//         auto hash_from_loc = terrain.locationToHashId(loc_from_hash);
//         EXPECT_EQ(hash_id, hash_from_loc);

//         // 6. Test location -> hash ID -> location
//         auto hash_from_loc_direct = terrain.locationToHashId(loc);
//         auto loc_from_hash_direct = terrain.hashIdToLocation(hash_from_loc_direct);
//         EXPECT_TRUE(floatEqual(loc[0], loc_from_hash_direct[0]));
//         EXPECT_TRUE(floatEqual(loc[1], loc_from_hash_direct[1]));

//         // 7. Verify hash ID is valid
//         EXPECT_TRUE(terrain.isHashIdValid(hash_id));

//         // 8. Verify the conversions work with the naive implementation
//         auto naive_global_idx = naive_terrain.locationToGlobalIndex(loc);
//         auto naive_loc = naive_terrain.globalIndexToLocation(naive_global_idx);
//         EXPECT_TRUE(floatEqual(loc[0], naive_loc[0]));
//         EXPECT_TRUE(floatEqual(loc[1], naive_loc[1]));
//     }

//     // Test edge cases
//     std::vector<std::array<float, 2>> edge_cases = {
//         {0.0f, 0.0f},           // Origin
//         {map_dim * resolution, map_dim * resolution},           // Map boundary
//         {-map_dim * resolution, -map_dim * resolution},         // Map boundary
//         {map_dim * resolution + 1.0f, map_dim * resolution + 1.0f},  // Outside map
//         {-map_dim * resolution - 1.0f, -map_dim * resolution - 1.0f} // Outside map
//     };

//     for (const auto& loc : edge_cases) {
//         // Test location -> hash ID conversion for edge cases
//         auto hash_id = terrain.locationToHashId(loc);
//         if (terrain.inside(loc)) {
//             EXPECT_TRUE(terrain.isHashIdValid(hash_id));
//         } else {
//             // For points outside the map, we expect the hash ID to be invalid
//             // or the point to be wrapped around to a valid position
//             auto wrapped_loc = terrain.hashIdToLocation(hash_id);
//             EXPECT_TRUE(terrain.inside(wrapped_loc));
//         }
//     }
// }



TEST_F(TerrainElevationTest, NaiveRecenter) {
    naive_terrain.printMapInformation();

    // Set up test points with known values
    const std::vector<std::array<float, 2>> test_points = {
        {0.0f, 0.0f},    // Origin
        {0.5f, 0.5f},    // Positive quadrant
        {-0.5f, -0.5f},  // Negative quadrant
        {1.0f, 1.0f},    // Edge case
        {-1.0f, -1.0f}   // Edge case
    };

    const float test_height = 0.5f;
    const float test_variance = 0.05f;
    const ElevationCell elevation(test_height, test_variance);

    std::cout << "Setting test data" << std::endl;
    // Set map with test data
    for (const auto& point : test_points) {
        bool success = naive_terrain.setElevation(point, elevation);
        ASSERT_TRUE(success) << "Failed to update point at [" << point[0] << ", " << point[1] << "]";
    }

    // Store initial data for comparison
    std::vector<std::pair<std::array<float, 2>, ElevationCell>> initial_data;
    for (const auto& point : test_points) {
        auto cell = naive_terrain.getElevation(point);
        ASSERT_TRUE(cell.has_value()) << "Failed to get elevation for point at [" << point[0] << ", " << point[1] << "]";
        EXPECT_TRUE(elevationCellEqual(cell.value(), elevation)) << "Elevation mismatch at [" << point[0] << ", " << point[1] << "]";
        initial_data.push_back({point, cell.value()});
    }
    
    // Test recentering with different shifts
    const std::vector<std::array<float, 2>> recenter_points = {
        {0.2f, 0.2f},    // Small shift
        {0.5f, 0.5f},    // Medium shift
        {1.0f, 1.0f},    // Large shift
        {-0.5f, -0.5f}   // Negative shift
    };

    // Get the current map origin before recentering
    auto original_origin = naive_terrain.getMapOrigin();
    std::cout << "Initial origin: [" << original_origin[0] << ", " << original_origin[1] << "]" << std::endl;
    
    // Track cumulative shift for point transformation
    std::array<float, 2> cumulative_shift = {0.0f, 0.0f};

    for (const std::array<float, 2>& recenter_point : recenter_points) {
        // Calculate shift from last origin
        std::array<float, 2> shift = {
            recenter_point[0] - original_origin[0],
            recenter_point[1] - original_origin[1]
        };
        
        // Update cumulative shift
        cumulative_shift[0] += shift[0];
        cumulative_shift[1] += shift[1];
        
        std::cout << "Recentering to [" << recenter_point[0] << ", " << recenter_point[1] 
                  << "], shift: [" << shift[0] << ", " << shift[1] << "]" << std::endl;
                  
        // Recenter the map
        naive_terrain.recenter(recenter_point);

        // Verify the origin was updated correctly
        auto new_origin = naive_terrain.getMapOrigin();
        EXPECT_TRUE(floatEqual(new_origin[0], recenter_point[0]))
            << "Origin x coordinate mismatch after recentering to " << recenter_point[0];
        EXPECT_TRUE(floatEqual(new_origin[1], recenter_point[1]))
            << "Origin y coordinate mismatch after recentering to " << recenter_point[1];

        // Check data consistency for each test point
        for (const auto& [original_point, original_cell] : initial_data) {
            // Calculate the new location after recentering
            // We need to shift the points in the opposite direction as the origin
            const std::array<float, 2> new_point = {
                original_point[0] - cumulative_shift[0],
                original_point[1] - cumulative_shift[1]
            };
            
            std::cout << "Original point [" << original_point[0] << ", " << original_point[1] 
                      << "] should now be at [" << new_point[0] << ", " << new_point[1] << "]" << std::endl;

            // Check if the transformed point is inside the map bounds
            if (naive_terrain.inside(new_point)) {
                // Get the cell at the new location
                auto new_cell = naive_terrain.getElevation(new_point);
                ASSERT_TRUE(new_cell.has_value())
                    << "Failed to get elevation after recenter for point at [" 
                    << new_point[0] << ", " << new_point[1] << "]";

                // Compare the elevation values
                EXPECT_TRUE(floatEqual(original_cell.height, new_cell.value().height))
                    << "Height mismatch at [" << new_point[0] << ", " << new_point[1] 
                    << "]: original=" << original_cell.height 
                    << ", new=" << new_cell.value().height;
                
                EXPECT_TRUE(floatEqual(original_cell.variance, new_cell.value().variance))
                    << "Variance mismatch at [" << new_point[0] << ", " << new_point[1] 
                    << "]: original=" << original_cell.variance 
                    << ", new=" << new_cell.value().variance;
            } else {
                std::cout << "Point [" << new_point[0] << ", " << new_point[1] 
                          << "] is now outside the map, skipping checks" << std::endl;
            }
        }

        // Update original_origin for next iteration
        original_origin = new_origin;
    }
}

// TEST_F(TerrainElevationTest, CompareRecenterWithNaive) {
//     // Initialize both maps with the same parameters
//     const float height = 0.0f;
//     const float variance = 0.1f;
//     const float min_variance = 0.01f;

//     terrain.initializeLocalMap(height, variance, min_variance);
//     naive_terrain.initializeLocalMap(height, variance, min_variance);

//     // Set up test points with known values
//     const std::vector<std::array<float, 2>> test_points = {
//         {0.0f, 0.0f},    // Origin
//         {0.5f, 0.5f},    // Positive quadrant
//         {-0.5f, -0.5f},  // Negative quadrant
//         {1.0f, 1.0f},    // Edge case
//         {-1.0f, -1.0f}   // Edge case
//     };

//     // Update both maps with the same data
//     for (const auto& point : test_points) {
//         const float test_height = 0.5f;
//         const float test_variance = 0.05f;
//         const double timestamp = 0.0;

//         terrain.update(point, test_height, test_variance, timestamp);
//         naive_terrain.update(point, test_height, test_variance, timestamp);
//     }

//     // Store initial data for comparison
//     std::vector<std::pair<std::array<float, 2>, std::pair<ElevationCell, ElevationCell>>> initial_data;
//     for (const auto& point : test_points) {
//         auto terrain_cell = terrain.getElevation(point);
//         auto naive_cell = naive_terrain.getElevation(point);
//         if (terrain_cell && naive_cell) {
//             initial_data.push_back({point, {terrain_cell.value(), naive_cell.value()}});
//         }
//     }

//     // Test recentering with different shifts
//     const std::vector<std::array<float, 2>> recenter_points = {
//         {0.2f, 0.2f},    // Small shift
//         {0.5f, 0.5f},    // Medium shift
//         {1.0f, 1.0f},    // Large shift
//         {-0.5f, -0.5f}   // Negative shift
//     };

//     for (const auto& recenter_point : recenter_points) {
//         // Recenter both maps
//         terrain.recenter(recenter_point);
//         naive_terrain.recenter(recenter_point);

//         // Verify that both maps have the same origin after recentering
//         EXPECT_EQ(terrain.getMapOrigin(), naive_terrain.getMapOrigin())
//             << "Map origins differ after recentering to " << recenter_point[0] << ", " << recenter_point[1];

//         // Check data consistency for each test point
//         for (const auto& [original_point, original_cells] : initial_data) {
//             // Calculate the new location after recentering
//             const std::array<float, 2> new_point = {
//                 original_point[0] - recenter_point[0],
//                 original_point[1] - recenter_point[1]
//             };

//             // Get the cells from both maps at the new location
//             auto terrain_cell = terrain.getElevation(new_point);
//             auto naive_cell = naive_terrain.getElevation(new_point);

//             // Verify both maps have data at the new location
//             EXPECT_TRUE(terrain_cell.has_value()) 
//                 << "Terrain map missing data at " << new_point[0] << ", " << new_point[1];
//             EXPECT_TRUE(naive_cell.has_value())
//                 << "Naive map missing data at " << new_point[0] << ", " << new_point[1];

//             if (terrain_cell && naive_cell) {
//                 // Compare the cells from both maps
//                 EXPECT_TRUE(elevationCellEqual(terrain_cell.value(), naive_cell.value()))
//                     << "Cells differ at " << new_point[0] << ", " << new_point[1];
//             }
//         }

//         // Test updating points in the new coordinate system
//         const std::array<float, 2> new_test_point = {0.3f, 0.3f};
//         const float new_height = 0.2f;
//         const float new_variance = 0.1f;
//         const double new_timestamp = 1.0;

//         terrain.update(new_test_point, new_height, new_variance, new_timestamp);
//         naive_terrain.update(new_test_point, new_height, new_variance, new_timestamp);

//         // Verify both maps store the new data correctly
//         auto terrain_cell = terrain.getElevation(new_test_point);
//         auto naive_cell = naive_terrain.getElevation(new_test_point);

//         EXPECT_TRUE(terrain_cell.has_value() && naive_cell.has_value());
//         if (terrain_cell && naive_cell) {
//             EXPECT_TRUE(elevationCellEqual(terrain_cell.value(), naive_cell.value()));
//         }
//     }
// }

TEST_F(TerrainElevationTest, NaiveCoordinateTransformations) {
    // Test location to global index conversion
    std::vector<std::pair<std::array<float, 2>, std::array<int, 2>>> test_points = {
        {{0.0f, 0.0f}, {0, 0}},           // Origin
        {{0.5f, 0.5f}, {50, 50}},         // Positive quadrant
        {{-0.5f, -0.5f}, {-50, -50}},     // Negative quadrant
        {{1.0f, 1.0f}, {100, 100}},       // Edge case
        {{-1.0f, -1.0f}, {-100, -100}},   // Edge case
        {{0.01f, 0.01f}, {1, 1}},         // Small values
        {{-0.01f, -0.01f}, {-1, -1}},     // Small negative values
        {{0.99f, 0.99f}, {99, 99}},       // Just below 1
        {{-0.99f, -0.99f}, {-99, -99}}    // Just above -1
    };

    for (const auto& [loc, expected_global] : test_points) {
        auto global_idx = naive_terrain.locationToGlobalIndex(loc);
        EXPECT_EQ(global_idx[0], expected_global[0]) 
            << "Global x index mismatch for location [" << loc[0] << ", " << loc[1] << "]";
        EXPECT_EQ(global_idx[1], expected_global[1]) 
            << "Global y index mismatch for location [" << loc[0] << ", " << loc[1] << "]";
    }

    // Test global index to array index conversion
    // First, set up a known map origin
    naive_terrain.initializeLocalMap(0.0f, 1.0f, 0.001f);
    
    // Test points with known global indices and expected array indices
    std::vector<std::pair<std::array<int, 2>, std::array<int, 2>>> test_indices = {
        {{0, 0}, {100, 100}},             // Origin
        {{50, 50}, {150, 150}},           // Positive quadrant
        {{-50, -50}, {50, 50}},           // Negative quadrant
        {{100, 100}, {200, 200}},         // Edge case
        {{-100, -100}, {0, 0}},           // Edge case
        {{1, 1}, {101, 101}},             // Small values
        {{-1, -1}, {99, 99}},             // Small negative values
        {{99, 99}, {199, 199}},           // Just below edge
        {{-99, -99}, {1, 1}}              // Just above negative edge
    };

    for (const auto& [global_idx, expected_array] : test_indices) {
        auto array_idx = naive_terrain.globalIndexToArrayIndex(global_idx);
        EXPECT_EQ(array_idx[0], expected_array[0]) 
            << "Array x index mismatch for global index [" << global_idx[0] << ", " << global_idx[1] << "]";
        EXPECT_EQ(array_idx[1], expected_array[1]) 
            << "Array y index mismatch for global index [" << global_idx[0] << ", " << global_idx[1] << "]";
    }

    // Test round-trip conversion
    for (const auto& [loc, _] : test_points) {
        // Location -> Global Index -> Location
        auto global_idx = naive_terrain.locationToGlobalIndex(loc);
        auto array_idx = naive_terrain.globalIndexToArrayIndex(global_idx);
        auto global_back = naive_terrain.globalIndexToArrayIndex(array_idx);
        
        EXPECT_EQ(global_idx[0], global_back[0]) 
            << "Round-trip x index mismatch for location [" << loc[0] << ", " << loc[1] << "]";
        EXPECT_EQ(global_idx[1], global_back[1]) 
            << "Round-trip y index mismatch for location [" << loc[0] << ", " << loc[1] << "]";
    }

    // Test boundary conditions
    std::vector<std::array<float, 2>> boundary_points = {
        {1.0f, 1.0f},     // Upper right corner
        {-1.0f, -1.0f},   // Lower left corner
        {1.0f, -1.0f},    // Upper left corner
        {-1.0f, 1.0f}     // Lower right corner
    };

    for (const auto& loc : boundary_points) {
        auto global_idx = naive_terrain.locationToGlobalIndex(loc);
        auto array_idx = naive_terrain.globalIndexToArrayIndex(global_idx);
        
        // Verify array indices are within bounds
        EXPECT_GE(array_idx[0], 0) << "Array x index out of bounds for location [" << loc[0] << ", " << loc[1] << "]";
        EXPECT_LT(array_idx[0], 200) << "Array x index out of bounds for location [" << loc[0] << ", " << loc[1] << "]";
        EXPECT_GE(array_idx[1], 0) << "Array y index out of bounds for location [" << loc[0] << ", " << loc[1] << "]";
        EXPECT_LT(array_idx[1], 200) << "Array y index out of bounds for location [" << loc[0] << ", " << loc[1] << "]";
    }
}

}  // namespace serow
