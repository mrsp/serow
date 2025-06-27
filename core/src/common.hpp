#pragma once
#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <optional>

namespace serow {

inline std::string findFilepath(const std::string& filename) {
    const char* serow_path_env = std::getenv("SEROW_PATH");
    if (serow_path_env == nullptr) {
        throw std::runtime_error("Environmental variable SEROW_PATH is not set.");
    }

    std::filesystem::path serow_path(serow_path_env);
    for (const auto& entry : std::filesystem::recursive_directory_iterator(serow_path)) {
        if (std::filesystem::is_regular_file(entry) && entry.path().filename() == filename) {
            return entry.path().string();
        }
    }

    throw std::runtime_error("File '" + filename + "' not found.");
}


constexpr float resolution = 0.01;
constexpr float resolution_inv = 1.0 / resolution;
constexpr float radius = 0.05;
constexpr int radius_cells = static_cast<int>(radius * resolution_inv) + 1;
constexpr int map_dim = 512;                 // 2^7
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
    std::vector<std::array<float, 3>> data{};
};

class TerrainElevation {
public:
    virtual ~TerrainElevation() = default;

    void printMapInformation() {
        const std::string GREEN = "\033[1;32m";
        const std::string WHITE = "\033[1;37m";
        std::cout << GREEN << "\tresolution: " << resolution << std::endl;
        std::cout << GREEN << "\tinverse resolution: " << resolution_inv << std::endl;
        std::cout << GREEN << "\tlocal map size: " << map_size << std::endl;
        std::cout << GREEN << "\tlocal map half size: " << half_map_size << std::endl;
        std::cout << GREEN << "\tlocal map dim: " << map_dim << std::endl;
        std::cout << GREEN << "\tlocal map half dim: " << half_map_dim << WHITE << std::endl;
    };

    const std::array<float, 2>& getMapOrigin() const {
        return local_map_origin_d_;
    }

    virtual void recenter(const std::array<float, 2>& location) = 0;

    virtual void initializeLocalMap(const float height, const float variance,
                                    const float min_variance = 1e-6) = 0;

    virtual bool update(const std::array<float, 2>& loc, float height, float variance) = 0;

    virtual bool setElevation(const std::array<float, 2>& loc, const ElevationCell& elevation) = 0;

    virtual std::optional<ElevationCell> getElevation(const std::array<float, 2>& loc) = 0;

    virtual bool inside(const std::array<int, 2>& id_g) const = 0;

    virtual bool inside(const std::array<float, 2>& location) const = 0;

    virtual int locationToHashId(const std::array<float, 2>& loc) const = 0;

    virtual std::array<float, 2> hashIdToLocation(const int hash_id) const = 0;

    virtual std::array<ElevationCell, map_size> getElevationMap() = 0;

protected:
    virtual void updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                              const std::array<int, 2>& new_origin_i) = 0;

    std::array<ElevationCell, map_size> elevation_;

    ElevationCell default_elevation_;
    float min_terrain_height_variance_{};

    std::array<int, 2> local_map_origin_i_{0, 0};
    std::array<int, 2> local_map_bound_max_i_{};
    std::array<int, 2> local_map_bound_min_i_{};
    std::array<float, 2> local_map_origin_d_{0.0, 0.0};
    std::array<float, 2> local_map_bound_max_d_{};
    std::array<float, 2> local_map_bound_min_d_{};
};

}  // namespace serow
