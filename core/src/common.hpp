#pragma once

namespace serow {

 constexpr float resolution = 0.01;
 constexpr float resolution_inv = 1.0 / resolution;
 constexpr float radius = 0.05;
 constexpr int radius_cells = static_cast<int>(radius * resolution_inv) + 1;
 constexpr int map_dim = 512;                // 2^7
 constexpr int half_map_dim = map_dim / 2;    // 2^6
 constexpr int map_size = map_dim * map_dim;  // 2^14 = 16.384
 constexpr int half_map_size = map_size / 2;  // 2^13 = 8.192

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

struct LocalMapState {
    double timestamp{};
    std::array<std::array<float, 3>, map_size> data{};
};

}  // namespace serow
