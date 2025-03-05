#pragma once

#include <iostream>
#include <cmath>

namespace {
static constexpr uint16_t size_x = 1000;
static constexpr uint16_t size_y = 1000;
static constexpr float resolution = 0.05; 
static constexpr float resolution_inv = 1.0 / resolution; 
static constexpr uint16_t map_size = size_x * size_y;
static constexpr uint16_t half_map_size = map_size / 2;
static constexpr std::array<int16_t, 2> map_dim = {size_x, size_y};
static constexpr std::array<int16_t, 2> half_map_dim = {size_x / 2, size_y / 2};
}

namespace serow {


class TerrainElevation {
public:

    bool inside(const std::array<int16_t, 2>& id_g) {
        if ((abs(id_g[0] - local_map_origin_i[0]) - half_map_dim[0]) > 0 || 
             (abs(id_g[1] - local_map_origin_i[1]) - half_map_dim[1]) > 0) {
            return false;
        }
        return true;
    }

    bool inside(const std::array<double, 2>& location) {
        return inside(locationToGlobalIndex(location));
    }

    void clearOutOfMapCells(const std::vector<int>& clear_id, const int& i) {
        std::vector<int> ids{i, (i + 1) % 2};
        for (const auto& x: clear_id) {
            for (int y = -sc_.half_map_size_i(ids[1]); y <= sc_.half_map_size_i(ids[1]); y++) {
                    const std::array<int16_t, 2> temp_clear_id = {x, y};
                    resetCell(getLocalIndexHash(temp_clear_id));
                }
            }
        }
    }

    void recenter() {

    }

    int16_t toGlobalIndex(const double pos) {
        return static_cast<int16_t>(resolution_inv * pos + pos > 0.0 ? 0.5 : -0.5);
    } 


    std::array<int16_t, 2> globalIndexToLocalIndex(const std::array<int16_t, 2> &id_g) const {
        std::array<int16_t, 2> id_l = {};

        for (int i = 0; i < 2; ++i) {
            // [eq. (7) in paper] Compute the i_k
            id_l[i] = id_g[i] % map_dim[i];
            // [eq. (8) in paper] Normalize the local index
            if (id_l[i] > half_map_dim[i]) {
                id_l[i] -= map_dim[i];
            } else if (id_l[i] < -half_map_dim[i]) {
                id_l[i] += map_dim[i];
            } 
        }
        return id_l;
    }


    double toLocation(const int16_t idx) {
        return  idx * resolution;
    }
private:
    std::array<float, map_size> height_{};
};


} 