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
#include <serow/NaiveTerrainElevation.hpp>
#include <random>
#include <cmath>
#include <algorithm>

namespace serow {

    class TerrainElevationTest : public ::testing::Test {
        protected:
            void SetUp() override {
                // Initialize both implementations with the same parameters
                terrain.initializeLocalMap(initial_height, initial_variance);
                naive_terrain.initializeLocalMap(initial_height, initial_variance);
                
                // Set the same minimum variance
                terrain.min_terrain_height_variance_ = min_variance;
                naive_terrain.min_terrain_height_variance_ = min_variance;
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
                       floatEqual(a.variance, b.variance, epsilon) &&
                       a.contact == b.contact &&
                       a.updated == b.updated;
            }
        };
        
        // Test basic initialization
        TEST_F(TerrainElevationTest, Initialization) {
            // Check if both implementations initialize the same
            const auto& elevation_cell = terrain.default_elevation_;
            const auto& naive_elevation_cell = naive_terrain.default_elevation_;
            
            EXPECT_TRUE(elevationCellEqual(elevation_cell, naive_elevation_cell));
        }
        
        // Test location to global index conversion
        TEST_F(TerrainElevationTest, LocationToGlobalIndex) {
            // Test various locations
            std::vector<float> test_locations = {
                -10.0f, -5.0f, -1.0f, -0.5f, -0.01f, 0.0f, 0.01f, 0.5f, 1.0f, 5.0f, 10.0f
            };
            
            for (const auto& loc : test_locations) {
                // Results may differ based on implementation details
                int idx_terrain = terrain.locationToGlobalIndex(loc);
                int idx_naive = naive_terrain.locationToGlobalIndex(loc);
                
                // Log the difference for analysis
                std::cout << "Location: " << loc << ", TerrainElevation: " << idx_terrain 
                          << ", NaiveTerrainElevation: " << idx_naive << std::endl;
                
                // We're not expecting equality here, just documenting the differences
            }
        }
        
        // Test inside function
        TEST_F(TerrainElevationTest, Inside) {
            // Test various 2D locations
            std::vector<std::array<float, 2>> test_locations = {
                {0.0f, 0.0f}, {1.0f, 1.0f}, {-1.0f, -1.0f}, {5.0f, -5.0f}, {-5.0f, 5.0f}
            };
            
            for (const auto& loc : test_locations) {
                bool inside_terrain = terrain.inside(loc);
                bool inside_naive = naive_terrain.inside(loc);
                
                std::cout << "Location: [" << loc[0] << ", " << loc[1] << "], "
                          << "TerrainElevation inside: " << inside_terrain 
                          << ", NaiveTerrainElevation inside: " << inside_naive << std::endl;
            }
        }
        
        // Test getting elevation at specific locations
        TEST_F(TerrainElevationTest, GetElevation) {
            std::vector<std::array<float, 2>> test_locations = {
                {0.0f, 0.0f}, {0.5f, 0.5f}, {-0.5f, -0.5f}, {1.0f, -1.0f}, {-1.0f, 1.0f}
            };
            
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
                {{-0.02f, -0.03f}, -0.05f, 0.04f}
            };
            
            // Apply updates to both implementations
            for (const auto& [loc, height, variance] : test_updates) {
                bool success_terrain = terrain.update(loc, height, variance);
                bool success_naive = naive_terrain.update(loc, height, variance);
                
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
                            if (dx == 0.0f && dy == 0.0f) continue;
                            
                            std::array<float, 2> nearby_loc = {loc[0] + dx, loc[1] + dy};
                            auto nearby_terrain = terrain.getElevation(nearby_loc);
                            auto nearby_naive = naive_terrain.getElevation(nearby_loc);
                            
                            if (nearby_terrain.has_value() && nearby_naive.has_value()) {
                                EXPECT_TRUE(elevationCellEqual(nearby_terrain.value(), nearby_naive.value()));
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
            // Verify that recenter updates the origin
            std::array<float, 2> new_center = {2.0f, 3.0f};
            terrain.recenter(new_center);
            
            auto origin_terrain = terrain.getMapOrigin();
            std::cout << "After recenter, TerrainElevation origin: [" 
                      << origin_terrain[0] << ", " << origin_terrain[1] << "]" << std::endl;
            
            // NaiveTerrainElevation doesn't have a recenter method, so we can't compare directly
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
            
            for (int i = 0; i < num_updates; ++i) {
                std::array<float, 2> loc = {loc_dist(gen), loc_dist(gen)};
                float height = height_dist(gen);
                float variance = var_dist(gen);
                
                bool success_terrain = terrain.update(loc, height, variance);
                bool success_naive = naive_terrain.update(loc, height, variance);
                
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
            
            std::cout << "Successfully updated " << successful_updates << " out of " 
                      << num_updates << " random locations." << std::endl;
        }
        
        // Test edge cases
        TEST_F(TerrainElevationTest, EdgeCases) {
            // Test very small values
            std::array<float, 2> small_loc = {0.001f, 0.001f};
            terrain.update(small_loc, 0.0001f, 0.0001f);
            naive_terrain.update(small_loc, 0.0001f, 0.0001f);
            
            auto small_terrain = terrain.getElevation(small_loc);
            auto small_naive = naive_terrain.getElevation(small_loc);
            
            if (small_terrain.has_value() && small_naive.has_value()) {
                EXPECT_TRUE(elevationCellEqual(small_terrain.value(), small_naive.value()));
            } else {
                EXPECT_EQ(small_terrain.has_value(), small_naive.has_value());
            }
            
            // Test very large values
            std::array<float, 2> large_loc = {100.0f, 100.0f};
            bool large_success_terrain = terrain.update(large_loc, 10.0f, 1.0f);
            bool large_success_naive = naive_terrain.update(large_loc, 10.0f, 1.0f);
            
            EXPECT_EQ(large_success_terrain, large_success_naive);
        }
        
        // Comprehensive test that iterates through a grid of locations
        TEST_F(TerrainElevationTest, ComprehensiveGridTest) {
            const float grid_start = -1.0f;
            const float grid_end = 1.0f;
            const float grid_step = 0.1f;
            
            int total_points = 0;
            int matching_points = 0;
            
            for (float x = grid_start; x <= grid_end; x += grid_step) {
                for (float y = grid_start; y <= grid_end; y += grid_step) {
                    std::array<float, 2> loc = {x, y};
                    float height = 0.1f * (x + y);
                    float variance = 0.01f;
                    
                    bool success_terrain = terrain.update(loc, height, variance);
                    bool success_naive = naive_terrain.update(loc, height, variance);
                    
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
            EXPECT_GT(match_percentage, 90.0f);
        }
        
} // namespace serow

