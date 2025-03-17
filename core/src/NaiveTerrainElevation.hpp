#pragma once

#include "TerrainElevation.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <optional>


namespace serow {


class NaiveTerrainElevation {
   public:

    void printMapInformation() const;
    
    bool inside(const std::array<int, 2>& id_g) const;

    bool inside(const std::array<float, 2>& location) const;

    void updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                      const std::array<int, 2>& new_origin_i);

    void resetCell(const int i, const int j);

    int locationToGlobalIndex(const float loc) const;

    std::array<int, 2> locationToGlobalIndex(const std::array<float, 2>& loc) const;

    std::array<float, 2> globalIndexToLocation(const std::array<int, 2>& id_g) const;

    void initializeLocalMap(const float height, const float variance);

    void resetLocalMap();

    bool update(const std::array<float, 2>& loc, float height, float variance);

    std::optional<ElevationCell> getElevation(const std::array<float, 2>& loc) const;
    
    const std::array<float, 2>& getMapOrigin() const;

    ElevationCell elevation_[map_dim][map_dim];
    std::vector<int64_t> contact_cells;
    
    ElevationCell default_elevation_;
    ElevationCell empty_elevation_{0.0, 1e4};
    
    std::array<int, 2> local_map_origin_i_{0, 0};
    std::array<int, 2> local_map_bound_max_i_{};
    std::array<int, 2> local_map_bound_min_i_{};
    std::array<float, 2> local_map_origin_d_{0.0, 0.0};
    std::array<float, 2> local_map_bound_max_d_{};
    std::array<float, 2> local_map_bound_min_d_{};

    float min_terrain_height_variance_{};

    friend class TerrainElevationTest; // Allow full access
};


}  // namespace serow
