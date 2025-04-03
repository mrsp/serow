#include <gtest/gtest.h>
#include <cmath>
#include <array>
#include <serow/NaiveTerrainElevation.hpp>
#include <serow/TerrainElevation.hpp>
#include <serow/common.hpp>

namespace serow {

class RecenterComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize both implementations with the same parameters.
        terrain.initializeLocalMap(initial_height, initial_variance, min_variance);
        naive_terrain.initializeLocalMap(initial_height, initial_variance, min_variance);
    }

    // Map initialization parameters.
    const float initial_height = 0.0f;
    const float initial_variance = 1.0f;
    const float min_variance = 0.001f;

    // The two implementations.
    TerrainElevation terrain;
    NaiveTerrainElevation naive_terrain;

    // Helper: Compare floating point numbers.
    bool floatEqual(float a, float b, float epsilon = 1e-5f) {
        return std::fabs(a - b) < epsilon;
    }

    // Helper: Compare two elevation cells.
    bool cellEqual(const ElevationCell &a, const ElevationCell &b, float epsilon = 1e-5f) {
        return floatEqual(a.height, b.height, epsilon) &&
               floatEqual(a.variance, b.variance, epsilon) &&
               a.contact == b.contact &&
               a.updated == b.updated;
    }
};

//
// Test: Compare the recenter operations of both implementations
//
// This test populates the local maps with non-default values, then recenter both
// implementations with the same new center. Finally, it compares every cell in the
// local map to verify that both recenter operations produced the same result.
//
TEST_F(RecenterComparisonTest, RecenterOutputConsistency) {
    // Populate the map with some non-default measurements.
    // We use a grid of global positions that cover a reasonable range.
    // (Assume update() accepts a position as std::array<float, 2> and a measurement.)
    for (int i = -half_map_dim + 1; i < half_map_dim; ++i) {
        for (int j = -half_map_dim + 1; j < half_map_dim; ++j) {
            // Create a test measurement (nontrivial) for each global coordinate.
            float measurement = std::sin(static_cast<float>(i)) + std::cos(static_cast<float>(j));
            std::array<float, 2> pos = {static_cast<float>(i), static_cast<float>(j)};
            terrain.update(pos, measurement,1e-3);
            naive_terrain.update(pos, measurement,1e-3);
        }
    }

    // Choose a new center that shifts the map, but not so far as to force a complete reset
    // in both implementations. Adjust the values if needed to satisfy the thresholds.
    std::array<float, 2> new_center = {50.0f, 50.0f};

    // Recenter both implementations.
    terrain.recenter(new_center);
    naive_terrain.recenter(new_center);

    // Compare every cell in the local map.
    // We assume the map is a square of size 'map_dim' x 'map_dim'.
    ElevationCell cellTerrain;// = terrain.getElevation(pos).value();
    ElevationCell cellNaive;// = terrain.getElevation(pos).value();

    for (int i = 0; i < map_dim; ++i) {
        for (int j = 0; j < map_dim; ++j) {
            std::array<float,2> pos{static_cast<float>(i),static_cast<float>(j)};
            if (terrain.getElevation(pos).has_value()){
               cellTerrain = terrain.getElevation(pos).value();
            }else{
              std::cout << "Terrain Estimator has no value at -> x:" << pos[0] << "  y: " << pos[1] << '\n';
            }
            if (naive_terrain.getElevation(pos).has_value()){
               cellNaive = naive_terrain.getElevation(pos).value();
            }else{
              std::cout << "Naive Estimator has no value at -> x:" << pos[0] << "  y: " << pos[1] << '\n';
            }
            EXPECT_TRUE(cellEqual(cellTerrain, cellNaive))
                << "Mismatch at cell (" << i << ", " << j << ") after recentering to ("
                << new_center[0] << ", " << new_center[1] << ").";
        }
    }
}

}  // namespace serow
