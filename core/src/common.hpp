#pragma once
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace serow {

inline std::string findFilepath(const std::string& filename) {
    const char* serow_path_env = std::getenv("SEROW_PATH");
    if (serow_path_env == nullptr) {
        throw std::runtime_error("Environmental variable SEROW_PATH is not set.");
    }

    std::function<std::string(const std::filesystem::path&)> searchRecursive =
        [&](const std::filesystem::path& dir) -> std::string {
        std::error_code ec;

        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (ec) {
                continue;
            }

            if (std::filesystem::is_regular_file(entry, ec) && !ec) {
                if (entry.path().filename() == filename) {
                    return entry.path().string();
                }
            } else if (std::filesystem::is_directory(entry, ec) && !ec) {
                try {
                    std::string result = searchRecursive(entry.path());
                    if (!result.empty()) {
                        return result;
                    }
                } catch (const std::exception& e) {
                    std::cout << "Skipping subdirectory: " << entry.path()
                              << " due to: " << e.what() << std::endl;
                }
            }
        }

        return "";
    };

    std::string result = searchRecursive(serow_path_env);
    if (result.empty()) {
        throw std::runtime_error("File '" + filename + "' not found.");
    }

    return result;
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
                                    const float min_variance = 1e-6,
                                    const float max_recenter_distance = 0.35,
                                    const size_t max_contact_points = 4) = 0;

    virtual bool update(const std::array<float, 2>& loc, float height, float variance) = 0;

    virtual bool setElevation(const std::array<float, 2>& loc, const ElevationCell& elevation) = 0;

    virtual std::optional<ElevationCell> getElevation(const std::array<float, 2>& loc) = 0;

    virtual bool inside(const std::array<int, 2>& id_g) const = 0;

    virtual bool inside(const std::array<float, 2>& location) const = 0;

    virtual int locationToHashId(const std::array<float, 2>& loc) const = 0;

    virtual std::array<float, 2> hashIdToLocation(const int hash_id) const = 0;

    virtual std::array<ElevationCell, map_size> getElevationMap() = 0;

    virtual std::tuple<std::array<float, 2>, std::array<float, 2>, std::array<float, 2>>
    getLocalMapInfo() = 0;

    void addContactPoint(const std::array<float, 2>& point) {
        // Check if the point is inside the local map
        if (!inside(point)) {
            return;
        }

        // Check if the point is already in the contact points
        constexpr float eps = 1e-6f;
        for (const auto& contact_point : contact_points_) {
            if (std::abs(contact_point[0] - point[0]) < eps &&
                std::abs(contact_point[1] - point[1]) < eps) {
                return;
            }
        }

        contact_points_.push_front(point);
        while (contact_points_.size() > max_contact_points_) {
            contact_points_.pop_back();
        }
    }

    float getMaxRecenterDistance() const {
        return max_recenter_distance_;
    }

    void clearContactPoints() {
        contact_points_.clear();
    }

    void interpolateContactPoints() {
        // Minimum number of contact points to compute a valid BBox else nothing to do here
        if (contact_points_.size() < 4) {
            return;
        }

        // Check that all contact points are inside the local map
        for (const auto& point : contact_points_) {
            if (!inside(point)) {
                // Remove the point from the contact points
                contact_points_.erase(
                    std::remove(contact_points_.begin(), contact_points_.end(), point),
                    contact_points_.end());
            }
        }

        // If we removed some contact points and we can't compute a valid BBox, nothing to do
        // here. This is faster to first check that all the points are inside the local map
        if (contact_points_.size() < 4) {
            return;
        }

        // Compute the bounding box of the contact points
        float min_x = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();

        for (const auto& point : contact_points_) {
            min_x = std::min(min_x, point[0]);
            max_x = std::max(max_x, point[0]);
            min_y = std::min(min_y, point[1]);
            max_y = std::max(max_y, point[1]);
        }

        // Check if the bounding box is a valid one
        if (min_x > max_x || min_y > max_y) {
            std::cout << "Invalid bounding box, clearing contact points" << std::endl;
            clearContactPoints();
            return;
        }

        // Interpolate using inverse distance weighting
        const float step = resolution;
        const float power = 2.0f;  // Power parameter for IDW

        for (float x = min_x; x <= max_x; x += step) {
            for (float y = min_y; y <= max_y; y += step) {
                std::array<float, 2> point{x, y};
                auto cell = getElevation(point);

                // Skip if cell doesn't exist or already has contact
                if (!cell || cell->contact) {
                    continue;
                }

                float sum_weights = 0.0f;
                float weighted_height = 0.0f;
                float weighted_variance = 0.0f;

                // Calculate weighted sum from all contact points
                for (const auto& contact_point : contact_points_) {
                    auto contact_cell = getElevation(contact_point);
                    if (!contact_cell) {
                        continue;
                    }

                    // Calculate distance
                    float dx = point[0] - contact_point[0];
                    float dy = point[1] - contact_point[1];
                    float distance = std::sqrt(dx * dx + dy * dy);

                    // Avoid division by zero
                    if (distance < resolution) {
                        weighted_height = contact_cell->height;
                        weighted_variance = contact_cell->variance;
                        sum_weights = 1.0f;
                        break;
                    }

                    // Calculate weight using inverse distance
                    float weight = 1.0f / std::pow(distance, power);
                    sum_weights += weight;
                    weighted_height += weight * contact_cell->height;
                    weighted_variance += weight * contact_cell->variance;
                }

                if (sum_weights > 0.0f) {
                    // Normalize the weighted sums
                    weighted_height /= sum_weights;
                    weighted_variance /= sum_weights;

                    // Update the cell
                    ElevationCell new_cell;
                    new_cell.height = weighted_height;
                    new_cell.variance = weighted_variance;
                    new_cell.contact = false;
                    new_cell.updated = true;
                    setElevation(point, new_cell);
                }
            }
        }

        // Remove the oldest two contact points
        contact_points_.pop_back();
        contact_points_.pop_back();
    }

protected:
    virtual void updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                              const std::array<int, 2>& new_origin_i) = 0;

    std::array<ElevationCell, map_size> elevation_;

    ElevationCell default_elevation_;
    float min_terrain_height_variance_{};
    size_t max_contact_points_{4};
    float max_recenter_distance_{0.35};
    std::deque<std::array<float, 2>> contact_points_{};

    std::array<int, 2> local_map_origin_i_{0, 0};
    std::array<int, 2> local_map_bound_max_i_{};
    std::array<int, 2> local_map_bound_min_i_{};
    std::array<float, 2> local_map_origin_d_{0.0, 0.0};
    std::array<float, 2> local_map_bound_max_d_{};
    std::array<float, 2> local_map_bound_min_d_{};
};

}  // namespace serow
