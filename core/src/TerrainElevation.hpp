#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

static constexpr int size_x = 1000;
static constexpr int size_y = 1000;
static constexpr float resolution = 0.05;
static constexpr float resolution_inv = 1.0 / resolution;
static constexpr int map_size = size_x * size_y;
static constexpr int half_map_size = map_size / 2;
static constexpr std::array<int, 2> map_dim = {size_x, size_y};
static constexpr std::array<int, 2> half_map_dim = {size_x / 2, size_y / 2};

}  // namespace

namespace serow {

class TerrainElevation {
   public:
    bool inside(const std::array<int, 2>& id_g) const;

    bool inside(const std::array<float, 2>& location) const;

    void resetCell(const int& hash_id);

    void updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                      const std::array<int, 2>& new_origin_i);

    void clearOutOfMapCells(const std::vector<int>& clear_id, const int& i);

    void recenter(const std::array<float, 2>& location);

    int locationToGlobalIndex(const float loc) const;

    std::array<int, 2> locationToGlobalIndex(const std::array<float, 2>& loc) const;

    std::array<float, 2> globalIndexToLocation(const std::array<int, 2>& id_g) const;

    std::array<int, 2> globalIndexToLocalIndex(const std::array<int, 2>& id_g) const;

    std::array<int, 2> localIndexToGlobalIndex(const std::array<int, 2>& id_l) const;

    std::array<float, 2> localIndexToLocation(const std::array<int, 2>& id_l) const;

    std::array<int, 2> hashIdToLocalIndex(const int hash_id) const;

    std::array<int, 2> hashIdToGlobalIndex(const int hash_id) const;

    std::array<float, 2> hashIdToLocation(const int hash_id) const;

    int localIndexToHashId(const std::array<int, 2>& id_in) const;

    int locationToHashId(const std::array<float, 2>& loc) const;

    int globalIndexToHashId(const std::array<int, 2>& id_g) const;

    void initializeLocalMap(const float height);

    void resetLocalMap();

   private:
    std::array<float, map_size> height_{};

    std::array<int, 2> local_map_origin_i_{};
    std::array<int, 2> local_map_bound_max_i_{};
    std::array<int, 2> local_map_bound_min_i_{};
    std::array<float, 2> local_map_origin_d_{};
    std::array<float, 2> local_map_bound_max_d_{};
    std::array<float, 2> local_map_bound_min_d_{};

    int normalize(int x, int a, int b);
};

}  // namespace serow
