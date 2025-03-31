#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

#include "TerrainElevation.hpp"

namespace serow {

class NaiveTerrainElevation {
public:
    void printMapInformation() const;

    void initializeLocalMap(const float height, const float variance,
                            const float min_variance = 1e-6);

    const LocalMapState& getLocalMap();

    bool update(const std::array<float, 2>& loc, float height, float variance, double timestamp);

    std::optional<ElevationCell> getElevation(const std::array<float, 2>& loc) const;

    const std::array<float, 2>& getMapOrigin() const;

private:
    void resetLocalMap();

    void updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                      const std::array<int, 2>& new_origin_i);

    bool inside(const std::array<int, 2>& id_g) const;

    bool inside(const std::array<float, 2>& location) const;

    void resetCell(const int i, const int j);

    int locationToGlobalIndex(const float loc) const;

    std::array<int, 2> locationToGlobalIndex(const std::array<float, 2>& loc) const;

    std::array<float, 2> globalIndexToLocation(const std::array<int, 2>& id_g) const;

    ElevationCell elevation_[map_dim][map_dim];

    ElevationCell default_elevation_;
    ElevationCell empty_elevation_{0.0, 1e4};

    std::array<int, 2> local_map_origin_i_{0, 0};
    std::array<int, 2> local_map_bound_max_i_{};
    std::array<int, 2> local_map_bound_min_i_{};
    std::array<float, 2> local_map_origin_d_{0.0, 0.0};
    std::array<float, 2> local_map_bound_max_d_{};
    std::array<float, 2> local_map_bound_min_d_{};

    float min_terrain_height_variance_{};
    double timestamp_{};

    friend class TerrainElevationTest;  // Allow full access
};

}  // namespace serow
