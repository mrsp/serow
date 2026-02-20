/**
 * Copyright (C) Stylianos Piperakis, Ownage Dynamics L.P.
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
#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <mutex>
#include <optional>
#include <vector>

#include "common.hpp"

namespace serow {

// Forward declaration of test class
class TerrainElevationTest;

class NaiveLocalTerrainMapper : public TerrainElevation {
public:
    void recenter(const std::array<float, 2>& loc) override;

    void initializeLocalMap(const float height, const float variance,
                            const float min_variance = 1e-6,
                            const float max_recenter_distance = 0.35,
                            const size_t max_contact_points = 4,
                            const float min_contact_probability = 0.15) override;

    bool update(const std::array<float, 2>& loc, float height, float variance, 
                std::optional<std::array<float, 3>> normal = std::nullopt) override;

    bool setElevation(const std::array<float, 2>& loc, const ElevationCell& elevation) override;

    std::optional<ElevationCell> getElevation(const std::array<float, 2>& loc) override;

    bool inside(const std::array<int, 2>& id_g) const override;

    bool inside(const std::array<float, 2>& loc) const override;

    std::array<ElevationCell, map_size> getElevationMap() override {
        std::lock_guard<std::mutex> lock(mutex_);
        return elevation_;
    }

    std::tuple<std::array<float, 2>, std::array<float, 2>, std::array<float, 2>> getLocalMapInfo()
        override {
        std::lock_guard<std::mutex> lock(mutex_);
        return {local_map_origin_d_, local_map_bound_max_d_, local_map_bound_min_d_};
    }

    int locationToHashId(const std::array<float, 2>& loc) const override;

    std::array<float, 2> hashIdToLocation(const int hash_id) const override;

private:
    int locationToGlobalIndex(const float loc) const;

    std::array<int, 2> locationToGlobalIndex(const std::array<float, 2>& loc) const;

    std::array<float, 2> globalIndexToLocation(const std::array<int, 2>& id_g) const;

    std::array<int, 2> globalIndexToLocalIndex(const std::array<int, 2>& id_g) const;

    std::array<int, 2> localIndexToGlobalIndex(const std::array<int, 2>& id_l) const;

    std::array<int, 2> locationToLocalIndex(const std::array<float, 2>& loc) const;

    std::array<float, 2> localIndexToLocation(const std::array<int, 2>& id_l) const;

    std::array<int, 2> hashIdToLocalIndex(const int hash_id) const;

    void resetLocalMap();

    void updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                      const std::array<int, 2>& new_origin_i) override;

    int localIndexToHashId(const std::array<int, 2>& id_l) const;

    int globalIndexToHashId(const std::array<int, 2>& id_g) const;

    std::mutex mutex_;

    friend class serow::TerrainElevationTest;  // Allow full access
};

}  // namespace serow
