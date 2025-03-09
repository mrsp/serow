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
    TerrainElevation terrain;
};


TEST_F(TerrainElevationTest, FastModulo) {
   for (int i=-10000; i <=10000; i++) {
      EXPECT_EQ(fast_mod<half_map_dim>(i), i % half_map_dim);
      EXPECT_EQ(fast_mod<map_dim>(i), i % map_dim);
   }
}

}
// TEST_F(TerrainElevationTest, LocationToGlobalIndex) {
//    // Test positive locations
//    EXPECT_EQ(terrain.locationToGlobalIndex(0.55f), 11);  // 0.55 / 0.05 = 11
//    EXPECT_EQ(terrain.locationToGlobalIndex(1.0f), 20);   // 1.0 / 0.05 = 20
   
//    // Test negative locations
//    EXPECT_EQ(terrain.locationToGlobalIndex(-0.55f), -11); // -0.55 / 0.05 = -11
//    EXPECT_EQ(terrain.locationToGlobalIndex(-1.0f), -20);  // -1.0 / 0.05 = -20
   
//    // Test 2D locations
//    std::array<float, 2> loc2d_pos = {0.55f, 1.0f};
//    std::array<int, 2> expected_pos = {11, 20};
//    EXPECT_EQ(terrain.locationToGlobalIndex(loc2d_pos), expected_pos);
   
//    std::array<float, 2> loc2d_neg = {-0.55f, -1.0f};
//    std::array<int, 2> expected_neg = {-11, -20};
//    EXPECT_EQ(terrain.locationToGlobalIndex(loc2d_neg), expected_neg);
// }

// TEST_F(TerrainElevationTest, Inside) {
//    // Test points inside the map (half_map_dim = {500, 500})
//    std::array<int, 2> point_inside = {100, 100};
//    EXPECT_TRUE(terrain.inside(point_inside));
   
//    // Test points at the edge of the map
//    std::array<int, 2> point_edge = {500, 500};
//    EXPECT_TRUE(terrain.inside(point_edge));
   
//    // Test points outside the map
//    std::array<int, 2> point_outside = {501, 0};
//    EXPECT_FALSE(terrain.inside(point_outside));
   
//    std::array<int, 2> point_far_outside = {1000, 1000};
//    EXPECT_FALSE(terrain.inside(point_far_outside));
   
//    // Test with locations instead of indices
//    std::array<float, 2> loc_inside = {5.0f, 5.0f};  // 5.0 / 0.05 = 100
//    EXPECT_TRUE(terrain.inside(loc_inside));
   
//    std::array<float, 2> loc_outside = {30.0f, 5.0f}; // 30.0 / 0.05 = 600 > 500
//    EXPECT_FALSE(terrain.inside(loc_outside));
// }

// TEST_F(TerrainElevationTest, ResetAndInitialize) {
//    // Test resetCell
//    int hash_id = terrain.localIndexToHashId({100, 100});
//    terrain.elevation_[hash_id] = ElevationCell(5.0f, 1.0f);
//    EXPECT_EQ(terrain.elevation_[hash_id].height, 5.0f);
   
//    terrain.resetCell(hash_id);
//    EXPECT_EQ(terrain.elevation_[hash_id].height, 0.0f);
   
//    // Test resetLocalMap
//    // Set several points to non-zero values
//    for (int i = 0; i < 10; i++) {
//        terrain.elevation_[i] = ElevationCell(3.0f, 1.0f);
//    }
   
//    terrain.resetLocalMap();
//    // Check that all points are reset
//    for (int i = 0; i < 10; i++) {
//        EXPECT_EQ(terrain.elevation_[i].height, 0.0f);
//        EXPECT_EQ(terrain.elevation_[i].stdev, 1e2f);
//    }
   
//    // Test initializeLocalMap
//    terrain.initializeLocalMap(2.5f, 1.0f);
//    // Check a sample of points
//    for (int i = 0; i < 100; i += 10) {
//        EXPECT_EQ(terrain.elevation_[i].height, 2.5f);
//        EXPECT_EQ(terrain.elevation_[i].stdev, 1.0f);
//    }
// }

// TEST_F(TerrainElevationTest, Recenter) {
//    // Initialize map with non-zero heights
//    terrain.initializeLocalMap(1.0f, 1.0f);
   
//    // Set a specific height at a known location
//    std::array<float, 2> loc = {1.0f, 1.0f};
//    terrain.update(loc, 5.0f, 1.0f);
   
//    // Get the hash ID before recentering
//    int hash_id_before = terrain.locationToHashId(loc);
//    EXPECT_EQ(terrain.elevation_[hash_id_before].height, 5.0f);
//    EXPECT_EQ(terrain.elevation_[hash_id_before].stdev, 1.0f);

//    // Recenter the map slightly (within map boundaries)
//    std::array<float, 2> new_center = {2.0f, 2.0f};
//    terrain.recenter(new_center);
   
//    // The point should still be accessible after minor recentering
//    int hash_id_after = terrain.locationToHashId(loc);
//    EXPECT_EQ(terrain.elevation_[hash_id_after].height, 5.0f);
   
//    // Test recentering that crosses boundaries but doesn't reset the map
//    std::array<float, 2> boundary_center = {10.0f, 10.0f};
//    terrain.recenter(boundary_center);
   
//    // Test major recentering that should reset the map
//    // With map_dim = {1000, 1000}, we need to move more than 1000 cells to trigger a reset
//    std::array<float, 2> far_center = {100.0f, 100.0f}; // 100 / 0.05 = 2000 > 1000
//    terrain.recenter(far_center);
   
//    // After major recentering, the entire map should be reset
//    // Check a sample of points
//    for (int i = 0; i < 100; i += 10) {
//        EXPECT_EQ(terrain.elevation_[i].height, 0.0f);
//    }
// }

// TEST_F(TerrainElevationTest, IndexConversions) {
//    // Test globalIndexToLocation
//    std::array<int, 2> global_idx = {20, -10};
//    std::array<float, 2> expected_loc = {1.0f, -0.5f}; // 20 * 0.05 = 1.0, -10 * 0.05 = -0.5
//    std::array<float, 2> actual_loc = terrain.globalIndexToLocation(global_idx);
//    EXPECT_FLOAT_EQ(actual_loc[0], expected_loc[0]);
//    EXPECT_FLOAT_EQ(actual_loc[1], expected_loc[1]);
   
//    // Test globalIndexToLocalIndex and localIndexToGlobalIndex (round trip)
//    std::array<int, 2> local_idx = terrain.globalIndexToLocalIndex(global_idx);
//    std::array<int, 2> round_trip = terrain.localIndexToGlobalIndex(local_idx);
//    EXPECT_EQ(round_trip, global_idx);
   
//    // Test with values near the map boundaries
//    std::array<int, 2> edge_global = {500, 500};
//    std::array<int, 2> edge_local = terrain.globalIndexToLocalIndex(edge_global);
//    std::array<int, 2> edge_round_trip = terrain.localIndexToGlobalIndex(edge_local);
//    EXPECT_EQ(edge_round_trip, edge_global);
   
//    // Test hashIdToLocalIndex and localIndexToHashId (round trip)
//    std::array<int, 2> local_pos = {100, 150};
//    int hash_id = terrain.localIndexToHashId(local_pos);
//    std::array<int, 2> local_round_trip = terrain.hashIdToLocalIndex(hash_id);
//    EXPECT_EQ(local_round_trip, local_pos);
// }

// TEST_F(TerrainElevationTest, UpdateHeight) {
//    // Test updating height at a valid location
//    std::array<float, 2> valid_loc = {5.0f, 5.0f}; // 5.0 / 0.05 = 100, which is inside the map
//    float new_height = 3.75f;
//    float new_stdev = 1.0f;

//    bool update_result = terrain.update(valid_loc, new_height, new_stdev);
//    EXPECT_TRUE(update_result);
   
//    int hash_id = terrain.locationToHashId(valid_loc);
//    EXPECT_FLOAT_EQ(terrain.elevation_[hash_id].height, new_height);
//    EXPECT_FLOAT_EQ(terrain.elevation_[hash_id].stdev, new_stdev);

//    // Test updating height at an invalid location (outside map bounds)
//    std::array<float, 2> invalid_loc = {50.0f, 50.0f}; // 50.0 / 0.05 = 1000, which is outside
//    update_result = terrain.update(invalid_loc, new_height, new_stdev);
//    EXPECT_FALSE(update_result);
// }

// // Additional test for high-resolution operations
// TEST_F(TerrainElevationTest, HighResolutionOperations) {
//    // Test operations with the high-resolution setting (0.05)
   
//    // Fine-grained location to index conversion
//    EXPECT_EQ(terrain.locationToGlobalIndex(0.05f), 1);  // Exactly one grid cell
//    EXPECT_EQ(terrain.locationToGlobalIndex(0.025f), 1); // Half a grid cell, rounded up
//    EXPECT_EQ(terrain.locationToGlobalIndex(-0.025f), -1); // Negative half cell, rounded down
   
//    // Test small movements in recenter
//    std::array<float, 2> start_loc = {0.0f, 0.0f};
//    terrain.recenter(start_loc);
   
//    // Set a specific pattern of heights
//    std::array<float, 2> loc1 = {0.05f, 0.0f};
//    std::array<float, 2> loc2 = {0.1f, 0.0f};
//    std::array<float, 2> loc3 = {0.15f, 0.0f};
   
//    terrain.update(loc1, 1.0f, 5.0f);
//    terrain.update(loc2, 2.0f, 10.0f);
//    terrain.update(loc3, 3.0f, 20.0f);
   
//    // Recenter by one cell
//    std::array<float, 2> new_center = {0.05f, 0.0f};
//    terrain.recenter(new_center);
   
//    // Check if the heights are still accessible at their correct locations
//    EXPECT_FLOAT_EQ(terrain.elevation_[terrain.locationToHashId(loc1)].height, 1.0f);
//    EXPECT_FLOAT_EQ(terrain.elevation_[terrain.locationToHashId(loc2)].height, 2.0f);
//    EXPECT_FLOAT_EQ(terrain.elevation_[terrain.locationToHashId(loc3)].height, 3.0f);
//    EXPECT_FLOAT_EQ(terrain.elevation_[terrain.locationToHashId(loc1)].stdev, 5.0f);
//    EXPECT_FLOAT_EQ(terrain.elevation_[terrain.locationToHashId(loc2)].stdev, 10.0f);
//    EXPECT_FLOAT_EQ(terrain.elevation_[terrain.locationToHashId(loc3)].stdev, 20.0f);
// }

// }
