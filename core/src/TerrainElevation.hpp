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

    int16_t locationToGlobalIndex(const float loc) {
        return static_cast<int16_t>(resolution_inv * loc + loc > 0.0 ? 0.5 : -0.5);
    } 

    std::array<int16_t, 2> locationToGlobalIndex(const std::array<float, 2>& loc) {
        return {resolution_inv * loc[0] + loc[0] > 0.0 ? 0.5 : -0.5, 
               {resolution_inv * loc[1] + loc[1] > 0.0 ? 0.5 : -0.5};
    } 

    std::array<float, 2> globalIndexToLocation(const std::array<int16_t, 2>& id_g) const {
        return {id_g[0] * resolution, id_g[1] * resolution};
    }

    std::array<int16_t, 2> globalIndexToLocalIndex(const std::array<int16_t, 2>& id_g) const {
        std::array<int16_t, 2> id_l = {};

        for (size_t i = 0; i < 2; i++) {
            id_l[i] = id_g[i] % map_dim[i];
            if (id_l[i] > half_map_dim[i]) {
                id_l[i] -= map_dim[i];
            } else if (id_l[i] < -half_map_dim[i]) {
                id_l[i] += map_dim[i];
            } 
        }
        return id_l;
    }

    std::array<int16_t, 2> localIndexToGlobalIndex(const std::array<int16_t, 2>& id_l) const {
        std::array<int16_t, 2> id_g = {};
        for (size_t i = 0; i < 2; i++) {
            int16_t min_id_g = -half_map_dim[i] + local_map_origin_i_[i];
            int16_t min_id_l = min_id_g % map_dim[i];
            min_id_l -= min_id_l > half_map_dim[i] ? map_dim[i] : 0;
            min_id_l += min_id_l < -half_map_dim[i] ? map_dim[i] : 0;

            int16_t cur_dis_to_min_id = id_l[i] - min_id_l;
            cur_dis_to_min_id = cur_dis_to_min_id < 0 ? map_dim[i] + cur_dis_to_min_id : cur_dis_to_min_id;
            int16_t cur_id = cur_dis_to_min_id + min_id_g;
            id_g[i] = cur_id;
        }
        return id_g;
    }
    
    std::array<float, 2> localIndexToLocation(const std::array<int16_t, 2>& id_l) const {
        std::array<float, 2> loc = {};
        for (size_t i = 0; i < 2; i++) {
            int16_t min_id_g = -half_map_dim[i] + local_map_origin_i_[i];
            int16_t min_id_l = min_id_g % map_dim[i];
            min_id_l -= min_id_l > half_map_dim[i] ? map_dim[i] : 0;
            min_id_l += min_id_l < -half_map_dim[i] ? map_dim[i] : 0;

            int16_t cur_dis_to_min_id = id_l[i] - min_id_l;
            cur_dis_to_min_id = cur_dis_to_min_id < 0 ? map_dim[i] + cur_dis_to_min_id : cur_dis_to_min_id;
            int16_t cur_id = cur_dis_to_min_id + min_id_g;
            loc[i] = cur_id * resolution;
        }
        return loc;
    }

    std::array<int16_t, 2> hashIdToLocalIndex(const int hash_id) const {
        int16_t id0 = hash_id / map_dim[1];
        int16_t id1 = hash_id - id0 * map_dim[1];
        return {id0 - half_map_dim[0], id1 - half_map_dim[1]};
    }

    std::array<int16_t, 2> hashIdToGlobalIndex(const int hash_id) const {
        return localIndexToGlobalIndex(hashIdToLocalIndex(hash_id));
    }

    std::array<float, 2> hashIdToLocation(const int hash_id) const {
        return localIndexToLocation(hashIdToLocalIndex(hash_id));
    }

    int localIndexToHashId(const std::array<int16_t, 2>& id_in) const {
        const std::array<int16_t, 2> id = {id_in[0] + half_map_dim[0], id_in[1] + half_map_dim[1]};
        return id[0] * map_dim[1] + id[1];
    }

    int locationToHashId(const std::array<float, 2>& loc) const {
        return localIndexToHashId(globalIndexToLocalIndex(locationToGlobalIndex(loc)));
    }


    int globalIndexToHashId(const std::array<int16_t, 2>& id_g) const {
        return localIndexToHashId(globalIndexToLocalIndex(id_g));
    }
private:
    std::array<float, map_size> height_{};
};

} 