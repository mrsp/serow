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

#include <chrono>
#include <iostream>
#include <serow/TerrainElevation.hpp>

namespace serow {

class TerrainElevationTest : public ::testing::Test {
   protected:
    TerrainElevation elevation_map;
};


// Test the fast modulo function
TEST_F(TerrainElevationTest, FastModulo) {
   for (int i=-10000; i <=10000; i++) {
      EXPECT_EQ(fast_mod<half_map_dim>(i), i % half_map_dim);
      EXPECT_EQ(fast_mod<map_dim>(i), i % map_dim);
   }
}

// Test initialization and basic properties
TEST_F(TerrainElevationTest, Initialization) {
   // Test initialization with specific values
   float default_height = 5.0f;
   float default_stdev = 2.0f;
   elevation_map.initializeLocalMap(default_height, default_stdev);
   
   // Check a random point to verify initialization
   std::array<float, 2> test_loc = {0.0f, 0.0f};
   auto elevation = elevation_map.getElevation(test_loc);
   ASSERT_TRUE(elevation.has_value());
   EXPECT_FLOAT_EQ(elevation->height, default_height);
   EXPECT_FLOAT_EQ(elevation->variance, default_stdev);
}

// Test coordinate transformation functions
TEST_F(TerrainElevationTest, CoordinateTransformations) {
   // Test location to global index conversion
   std::array<float, 2> loc = {1.025f, -2.975f};
   std::array<int, 2> global_index = elevation_map.locationToGlobalIndex(loc);
   EXPECT_EQ(global_index[0], 21);    // 1.025 / 0.05 + 0.5 = 21
   EXPECT_EQ(global_index[1], -60);   // -2.975 / 0.05 - 0.5 = -60
   
   // Test global index to location conversion
   std::array<float, 2> loc_back = elevation_map.globalIndexToLocation(global_index);
   EXPECT_FLOAT_EQ(loc_back[0], 1.05f);   // 21 * 0.05 = 1.05
   EXPECT_FLOAT_EQ(loc_back[1], -3.0f);   // -60 * 0.05 = -3.0
   
   // Test local index conversions
   elevation_map.recenter({0.0f, 0.0f});  // Reset map origin
   std::array<int, 2> local_index = {10, -10};
   std::array<int, 2> global_from_local = elevation_map.localIndexToGlobalIndex(local_index);
   std::array<int, 2> local_back = elevation_map.globalIndexToLocalIndex(global_from_local);
   EXPECT_EQ(local_back[0], local_index[0]);
   EXPECT_EQ(local_back[1], local_index[1]);
   
   // Test hash ID conversions
   int hash_id = elevation_map.localIndexToHashId(local_index);
   std::array<int, 2> local_from_hash = elevation_map.hashIdToLocalIndex(hash_id);
   EXPECT_EQ(local_from_hash[0], local_index[0]);
   EXPECT_EQ(local_from_hash[1], local_index[1]);
}

// Test "inside" boundary checking
TEST_F(TerrainElevationTest, BoundaryChecking) {
   elevation_map.recenter({0.0f, 0.0f});
   
   // Max map range with half_map_dim = 512 and resolution = 0.05 is ±25.6m
   const float max_range = half_map_dim * resolution;
   
   // Points inside the map
   EXPECT_TRUE(elevation_map.inside(std::array<float, 2>{0.0f, 0.0f}));
   EXPECT_TRUE(elevation_map.inside(std::array<float, 2>{max_range - 0.1f, max_range - 0.1f}));
   EXPECT_TRUE(elevation_map.inside(std::array<float, 2>{-max_range + 0.1f, -max_range + 0.1f}));
   
   // Points outside the map
   EXPECT_FALSE(elevation_map.inside(std::array<float, 2>{max_range + 0.1f, 0.0f}));
   EXPECT_FALSE(elevation_map.inside(std::array<float, 2>{0.0f, -max_range - 0.1f}));
   EXPECT_FALSE(elevation_map.inside(std::array<float, 2>{30.0f, 30.0f}));
}

// Test map recentering
TEST_F(TerrainElevationTest, Recentering) {
   // Initialize with specific values
   elevation_map.initializeLocalMap(0.0, 1.0);
   
   // Set a specific elevation value at the origin
   std::array<float, 2> origin = {0.0f, 0.0f};
   elevation_map.update(origin, 10.0f, 0.5f);
   
   // Verify the value was set
   auto elevation_before = elevation_map.getElevation(origin);
   ASSERT_TRUE(elevation_before.has_value());
   
   // Move the map center by a small amount (still keeping original point in map)
   std::array<float, 2> new_center = {5.0f, 5.0f};
   elevation_map.recenter(new_center);
   
   // Check that the original point is still in the map
   auto elevation_after = elevation_map.getElevation(origin);
   ASSERT_TRUE(elevation_after.has_value());
   EXPECT_FLOAT_EQ(elevation_after->height, elevation_before->height);
   
   // Move the map center by a large amount (forcing original point out of map)
   std::array<float, 2> far_center = {60.0f, 60.0f};
   elevation_map.recenter(far_center);
   
   // Original point should be reset to default values or be outside map
   auto elevation_reset = elevation_map.getElevation(origin);
   if (elevation_reset.has_value()) {
       EXPECT_FLOAT_EQ(elevation_reset->height, 0.0f);  // Default height after reset
   } else {
       // If out of bounds, that's also acceptable
       EXPECT_FALSE(elevation_map.inside(origin));
   }
}

// Test single point update and Kalman filter
TEST_F(TerrainElevationTest, SinglePointUpdate) {
   elevation_map.initializeLocalMap(0.0, 1.0);
   
   // Update a single point
   std::array<float, 2> test_loc = {1.0f, 1.0f};
   float new_height = 5.0f;
   float new_stdev = 0.2f;
   
   // Check return value
   EXPECT_TRUE(elevation_map.update(test_loc, new_height, new_stdev));
   
   // Verify the update worked
   auto elevation = elevation_map.getElevation(test_loc);
   ASSERT_TRUE(elevation.has_value());
   
   // Calculate expected Kalman filter results
   float expected_height = (0.2f * 0.0f + 1.0f * 5.0f) / (0.2f + 1.0f);
   float expected_stdev = (0.2f * 1.0f) / (0.2f + 1.0f);
   
   // Verify against calculated values
   EXPECT_NEAR(elevation->height, expected_height, 1e-5);
   EXPECT_NEAR(elevation->variance, expected_stdev, 1e-5);
}

// Test circle propagation in update function
TEST_F(TerrainElevationTest, CirclePropagation) {
   elevation_map.initializeLocalMap(0.0, 1.0);
   
   // Update a point
   std::array<float, 2> center = {10.0f, 10.0f};
   float new_height = 10.0f;
   float new_stdev = 0.1f;
   
   elevation_map.update(center, new_height, new_stdev);
   
   // Check points at different distances from center
   // With radius = 0.5m, we'll test points within and just beyond this radius
   std::vector<std::array<float, 2>> test_points = {
       {10.0f, 10.0f},    // Center point
       {10.05f, 10.0f},   // 0.05m away (1 cell)
       {10.25f, 10.0f},   // 0.25m away (5 cells)
       {10.45f, 10.0f},   // 0.45m away (9 cells) - just within radius
       {10.55f, 10.0f},   // 0.55m away (11 cells) - just outside radius
       {10.7f, 10.0f}     // 0.7m away (14 cells) - clearly outside radius
   };
   
   // Get elevations at test points
   std::vector<std::optional<ElevationCell>> elevations;
   for (const auto& point : test_points) {
       elevations.push_back(elevation_map.getElevation(point));
   }
   
   // All points should have valid elevations (they're within map bounds)
   for (size_t i = 0; i < elevations.size(); ++i) {
       ASSERT_TRUE(elevations[i].has_value()) << "Point " << i << " has no elevation";
   }
   
   // Center point should have the highest elevation due to Kalman update
   float center_value = elevations[0]->height;
   EXPECT_GT(center_value, 0.0f);
   
   // Elevation should decrease with distance within radius
   // (if using Gaussian weight model: exp(-d²/(2*r²)))
   EXPECT_GT(elevations[0]->height, elevations[1]->height);
   EXPECT_GT(elevations[1]->height, elevations[2]->height);
   if (elevations[3]->height > 0) {  // If propagation reaches this far
       EXPECT_GT(elevations[2]->height, elevations[3]->height);
   }
   
   // Verify Gaussian falloff (approximately)
   if (elevations[1]->height > 0) {
       float d1 = 0.05f;
       float expected_weight1 = std::exp(-d1*d1/(2.0f*radius*radius));
       EXPECT_NEAR(elevations[1]->height / center_value, expected_weight1, 0.1f);
   }
   
   // Points outside radius should have lower or zero values
   if (elevations[4]->height > 0 && elevations[3]->height > 0) {
       EXPECT_LT(elevations[4]->height, elevations[3]->height);
   }
}

// Test elevation retrieval for out-of-bounds points
TEST_F(TerrainElevationTest, OutOfBoundsRetrieval) {
   elevation_map.recenter({0.0f, 0.0f});
   
   // Try to get elevation for points outside the map
   // With half_map_dim = 512 and resolution = 0.05, map extends to ±25.6m
   std::array<float, 2> out_of_bounds = {30.0f, 30.0f};
   auto elevation = elevation_map.getElevation(out_of_bounds);
   
   // Should return nullopt
   EXPECT_FALSE(elevation.has_value());
}

// Test performance of the update function with known parameters
TEST_F(TerrainElevationTest, UpdatePerformance) {
   elevation_map.initializeLocalMap(0.0, 1.0);
   
   // With radius = 0.5m and resolution = 0.05m, each update affects approximately:
   // π * (0.5/0.05)² = π * 100 ≈ 314 cells
   
   // Number of updates to perform
   const int num_updates = 1000;
   
   // Prepare random positions (to avoid overhead during timing)
   std::vector<std::array<float, 2>> positions;
   std::vector<float> heights;
   std::vector<float> stdevs;
   
   for (int i = 0; i < num_updates; ++i) {
       // Generate random positions within the map (±20m from center)
       float x = (rand() % 800 - 400) * 0.05f;  // -20 to 20
       float y = (rand() % 800 - 400) * 0.05f;  // -20 to 20
       float height = rand() % 10;              // 0 to 9
       float stdev = 0.1f + (rand() % 10) / 10.0f;  // 0.1 to 1.0
       
       positions.push_back({x, y});
       heights.push_back(height);
       stdevs.push_back(stdev);
   }
   
   // Measure time for updates
   auto start_time = std::chrono::high_resolution_clock::now();
   
   for (int i = 0; i < num_updates; ++i) {
       elevation_map.update(positions[i], heights[i], stdevs[i]);
   }
   
   auto end_time = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
   
   // Calculate and output performance metrics
   double updates_per_second = (duration > 0) ? (num_updates * 1000.0 / duration) : 0;
   double cells_per_second = updates_per_second * 314;  // Approximate cell updates per second
   
   std::cout << "Performance: " << num_updates << " updates in " << duration << " ms" << std::endl;
   std::cout << "Updates per second: " << updates_per_second << std::endl;
   std::cout << "Approximate cell updates per second: " << cells_per_second << std::endl;
   
   // Fail if performance is below threshold
   const double minimum_acceptable_updates_per_second = 400000;
   EXPECT_GT(updates_per_second, minimum_acceptable_updates_per_second);
}

}  // namespace serow
