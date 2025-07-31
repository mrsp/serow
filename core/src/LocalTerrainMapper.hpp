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
    constexpr int a = -half_map_dim;
    constexpr int b = half_map_dim;
    constexpr int range = b - a + 1;
    int y = (x - a) % range;
    return (y < 0 ? y + range : y) + a;
}

class LocalTerrainMapper : public TerrainElevation {
public:
    virtual void recenter(const std::array<float, 2>& loc) override;

    void initializeLocalMap(const float height, const float variance,
                            const float min_variance = 1e-6,
                            const float max_recenter_distance = 0.35,
                            const size_t max_contact_points = 4) override;

    bool update(const std::array<float, 2>& loc, float height, float variance) override;

    bool setElevation(const std::array<float, 2>& loc, const ElevationCell& elevation) override;

    std::optional<ElevationCell> getElevation(const std::array<float, 2>& loc) override;

    bool inside(const std::array<int, 2>& id_g) const override;

    bool inside(const std::array<float, 2>& location) const override;

    std::array<ElevationCell, map_size> getElevationMap() override {
        std::lock_guard<std::mutex> lock(mutex_);
        return elevation_;
    }

    std::tuple<std::array<float, 2>, std::array<float, 2>, std::array<float, 2>> getLocalMapInfo()
        override {
        std::lock_guard<std::mutex> lock(mutex_);
        return {local_map_origin_d_, local_map_bound_max_d_, local_map_bound_min_d_};
    }

    // Coordinate conversion functions made public for testing
    std::array<int, 2> globalIndexToLocalIndex(const std::array<int, 2>& id_g) const;
    std::array<int, 2> localIndexToGlobalIndex(const std::array<int, 2>& id_l) const;
    std::array<float, 2> globalIndexToLocation(const std::array<int, 2>& id_g) const;
    std::array<float, 2> localIndexToLocation(const std::array<int, 2>& id_l) const;
    std::array<int, 2> locationToGlobalIndex(const std::array<float, 2>& loc) const;

    // Hash ID related functions made public for testing
    bool isHashIdValid(const int id) const;
    int locationToHashId(const std::array<float, 2>& loc) const override;
    int localIndexToHashId(const std::array<int, 2>& id_in) const;
    std::array<int, 2> hashIdToLocalIndex(const int hash_id) const;
    std::array<int, 2> hashIdToGlobalIndex(const int hash_id) const;
    std::array<float, 2> hashIdToLocation(const int hash_id) const override;
    int globalIndexToHashId(const std::array<int, 2>& id_g) const;

private:
    void updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                      const std::array<int, 2>& new_origin_i) override;

    void resetLocalMap();

    void resetCell(const int& hash_id);

    void clearOutOfMapCells(const std::vector<int>& clear_id, const int i);

    int locationToGlobalIndex(const float loc) const;

    std::mutex mutex_;

    friend class TerrainElevationTest;  // Allow full access
};

}  // namespace serow
