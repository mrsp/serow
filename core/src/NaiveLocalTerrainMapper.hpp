#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <mutex>
#include <optional>
#include <vector>

#include "common.hpp"

namespace serow {

class NaiveLocalTerrainMapper : public TerrainElevation {
public:
    void recenter(const std::array<float, 2>& loc) override;

    void initializeLocalMap(const float height, const float variance,
                            const float min_variance = 1e-6) override;

    bool update(const std::array<float, 2>& loc, float height, float variance) override;

    bool setElevation(const std::array<float, 2>& loc, const ElevationCell& elevation) override;

    std::optional<ElevationCell> getElevation(const std::array<float, 2>& loc) override;

    bool inside(const std::array<int, 2>& id_g) const override;

    bool inside(const std::array<float, 2>& loc) const override;

    std::array<ElevationCell, map_size> getElevationMap() override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::array<ElevationCell, map_size> result;
        for (int i = 0; i < map_dim; ++i) {
            for (int j = 0; j < map_dim; ++j) {
                result[i * map_dim + j] = elevation_[i][j];
            }
        }
        return result;
    }

    int locationToHashId(const std::array<float, 2>& loc) const override;

private:
    int locationToGlobalIndex(const float loc) const;

    std::array<int, 2> locationToGlobalIndex(const std::array<float, 2>& loc) const;

    std::array<float, 2> globalIndexToLocation(const std::array<int, 2>& id_g) const;

    std::array<int, 2> globalIndexToLocalIndex(const std::array<int, 2>& id_g) const;

    std::array<int, 2> localIndexToGlobalIndex(const std::array<int, 2>& id_l) const;

    std::array<int, 2> locationToLocalIndex(const std::array<float, 2>& loc) const;

    std::array<float, 2> localIndexToLocation(const std::array<int, 2>& id_l) const;

    void resetLocalMap();

    void updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                      const std::array<int, 2>& new_origin_i) override;

    void resetCell(const int i, const int j);

    // Map data
    ElevationCell elevation_[map_dim][map_dim];

    std::mutex mutex_;

    friend class TerrainElevationTest;  // Allow full access
};

}  // namespace serow
