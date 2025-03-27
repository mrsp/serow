#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

namespace {

static constexpr float resolution = 0.01;
static constexpr float resolution_inv = 1.0 / resolution;
static constexpr float radius = 0.05;
static constexpr int radius_cells = static_cast<int>(radius * resolution_inv) + 1;
static constexpr int map_dim = 1024;                // 2^7
static constexpr int half_map_dim = map_dim / 2;    // 2^6
static constexpr int map_size = map_dim * map_dim;  // 2^14 = 16.384
static constexpr int half_map_size = map_size / 2;  // 2^13 = 8.192

template <int N>
inline int fast_mod(const int x) {
    static_assert((N & (N - 1)) == 0, "N must be a power of 2");
    constexpr int mask = N - 1;

    // For positive numbers, the bitwise AND works perfectly
    if (x >= 0) {
        return x & mask;
    }

    // For negative numbers, we need special handling
    int remainder = x & mask;

    // If remainder is 0, the result is 0
    if (remainder == 0) {
        return 0;
    }

    // Otherwise, we need to return a negative result
    return remainder - N;
}

inline int normalize(const int x) {
    // Since a = -half_map_dim and b = half_map_dim, the range is 2*half_map_dim + 1
    constexpr int a = -half_map_dim;

    int range = 2 * half_map_dim + 1;
    int y = (x - a) % range;
    return (y < 0 ? y + range : y) + a;
}

}  // namespace

namespace serow {

struct ElevationCell {
    float height{};
    float variance{};
    bool contact{};
    bool updated{};
    ElevationCell() = default;
    ElevationCell(float height, float variance) {
        this->height = height;
        this->variance = variance;
    }
};

class TerrainElevation {
public:
    void printMapInformation() const;

    bool inside(const std::array<int, 2>& id_g) const;

    bool inside(const std::array<float, 2>& location) const;

    void resetCell(const int& hash_id);

    void updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                      const std::array<int, 2>& new_origin_i);

    void clearOutOfMapCells(const std::vector<int>& clear_id);

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

    bool isHashIdValid(const int id) const;

    void initializeLocalMap(const float height, const float variance);

    void resetLocalMap();

    bool update(const std::array<float, 2>& loc, float height, float variance);

    std::optional<ElevationCell> getElevation(const std::array<float, 2>& loc) const;

    const std::array<float, 2>& getMapOrigin() const;

    std::array<ElevationCell, map_size> elevation_;

    ElevationCell default_elevation_;
    ElevationCell empty_elevation_{0.0, 1e2};

    std::array<int, 2> local_map_origin_i_{0, 0};
    std::array<int, 2> local_map_bound_max_i_{};
    std::array<int, 2> local_map_bound_min_i_{};
    std::array<float, 2> local_map_origin_d_{0.0, 0.0};
    std::array<float, 2> local_map_bound_max_d_{};
    std::array<float, 2> local_map_bound_min_d_{};

    float min_terrain_height_variance_{};

    friend class TerrainElevationTest;  // Allow full access
};

}  // namespace serow
