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
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace serow {

inline std::string findFilepath(const std::string& filename) {
    const char* serow_path_env = std::getenv("SEROW_PATH");
    if (serow_path_env == nullptr) {
        throw std::runtime_error("Environment variable SEROW_PATH is not set.");
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
                              << " due to: " << e.what() << '\n';
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

constexpr int map_dim = 512;                 // 2^7
constexpr int half_map_dim = map_dim / 2;    // 2^6
constexpr int map_size = map_dim * map_dim;  // 2^14 = 16.384
constexpr int half_map_size = map_size / 2;  // 2^13 = 8.192

class TerrainElevation {
public:
    struct Params {
        float resolution;
        float resolution_inv;
        float radius;
        int radius_cells;
        float dist_variance_gain;
        float power;
        float min_variance;
        float max_recenter_distance;
        size_t max_contact_points;
        float min_contact_probability;
        float min_stable_contact_probability;
        float min_stable_foot_angular_velocity;
        float min_stable_foot_linear_velocity;
        Params()
            : resolution(0.02f),
              resolution_inv(1.0f / 0.02f),
              radius(0.20f),
              radius_cells(static_cast<int>(0.20f * 1.0f / 0.02f) + 1),
              dist_variance_gain(100.0f),
              power(5.0f),
              min_variance(1e-6f),
              max_recenter_distance(0.35f),
              max_contact_points(4),
              min_contact_probability(0.15f),
              min_stable_contact_probability(0.95f),
              min_stable_foot_angular_velocity(0.03f),
              min_stable_foot_linear_velocity(0.03f) {}
        Params(const float resolution, const float radius, const float dist_variance_gain,
               const float power, const float min_variance, const float max_recenter_distance,
               const size_t max_contact_points, const float min_contact_probability,
               const float min_stable_contact_probability = 0.95f,
               const float min_stable_foot_angular_velocity = 0.03f,
               const float min_stable_foot_linear_velocity = 0.03f) {
            this->resolution = resolution;
            this->resolution_inv = 1.0f / resolution;
            this->radius = radius;
            this->radius_cells = static_cast<int>(radius * resolution_inv) + 1;
            this->dist_variance_gain = dist_variance_gain;
            this->power = power;
            this->min_variance = min_variance;
            this->max_recenter_distance = max_recenter_distance;
            this->max_contact_points = max_contact_points;
            this->min_contact_probability = min_contact_probability;
            this->min_stable_contact_probability = min_stable_contact_probability;
            this->min_stable_foot_angular_velocity = min_stable_foot_angular_velocity;
            this->min_stable_foot_linear_velocity = min_stable_foot_linear_velocity;
        }
    };
    TerrainElevation(bool point_feet = false) : point_feet_(point_feet) {}
    virtual ~TerrainElevation() = default;

    void printMapInformation() {
        const std::string GREEN = "\033[1;32m";
        const std::string WHITE = "\033[1;37m";
        std::cout << GREEN << "\tresolution: " << params_.resolution << '\n';
        std::cout << GREEN << "\tinverse resolution: " << params_.resolution_inv << '\n';
        std::cout << GREEN << "\tlocal map size: " << map_size << '\n';
        std::cout << GREEN << "\tlocal map half size: " << half_map_size << '\n';
        std::cout << GREEN << "\tlocal map dim: " << map_dim << '\n';
        std::cout << GREEN << "\tlocal map half dim: " << half_map_dim << WHITE << '\n';
    };

    const std::array<float, 2>& getMapOrigin() const {
        return local_map_origin_d_;
    }

    virtual void recenter(const std::array<float, 2>& location) = 0;

    virtual void initializeLocalMap(const float height, const float variance,
                                    const Params& params = Params()) = 0;

    virtual bool update(const std::array<float, 2>& loc, float height, float variance,
                        std::optional<std::array<float, 3>> normal = std::nullopt) = 0;

    bool setElevation(const std::array<float, 2>& loc, const ElevationCell& elevation) {
        std::lock_guard<std::mutex> lock(mutex_);
        return setElevationUnlocked(loc, elevation);
    }

    std::optional<ElevationCell> getElevation(const std::array<float, 2>& loc) {
        std::lock_guard<std::mutex> lock(mutex_);
        return getElevationUnlocked(loc);
    }

    virtual bool inside(const std::array<int, 2>& id_g) const = 0;

    virtual bool inside(const std::array<float, 2>& location) const = 0;

    virtual int locationToHashId(const std::array<float, 2>& loc) const = 0;

    virtual std::array<float, 2> hashIdToLocation(const int hash_id) const = 0;

    virtual std::array<ElevationCell, map_size> getElevationMap() = 0;

    virtual std::tuple<std::array<float, 2>, std::array<float, 2>, std::array<float, 2>>
    getLocalMapInfo() = 0;

    // Downsampled elevation/variance grid built under one lock (consistent snapshot).
    struct DownsampledElevationGrid {
        std::vector<float> elevation;
        std::vector<float> variance;
        std::array<float, 2> origin{};
        double resolution{};
        uint32_t width{};
        uint32_t height{};
    };

    std::optional<DownsampledElevationGrid> copyDownsampledElevationGrid(size_t downsample_factor) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (downsample_factor == 0) {
            return std::nullopt;
        }

        const double res =
            static_cast<double>(params_.resolution) * static_cast<double>(downsample_factor);
        if (!(res > 0.0) || !std::isfinite(res)) {
            return std::nullopt;
        }

        const auto& origin = local_map_origin_d_;
        const auto& bound_max = local_map_bound_max_d_;
        const auto& bound_min = local_map_bound_min_d_;

        const double dx = static_cast<double>(bound_max[0]) - static_cast<double>(bound_min[0]);
        const double dy = static_cast<double>(bound_max[1]) - static_cast<double>(bound_min[1]);
        if (!std::isfinite(dx) || !std::isfinite(dy) || dx <= 0.0 || dy <= 0.0) {
            return std::nullopt;
        }

        const uint32_t width = static_cast<uint32_t>(std::ceil(dx / res));
        const uint32_t height = static_cast<uint32_t>(std::ceil(dy / res));
        if (width == 0 || height == 0) {
            return std::nullopt;
        }

        const size_t grid_size = static_cast<size_t>(width) * static_cast<size_t>(height);
        constexpr size_t max_grid_size =
            static_cast<size_t>(map_dim) * static_cast<size_t>(map_dim);
        if (grid_size == 0 || grid_size > max_grid_size) {
            return std::nullopt;
        }

        DownsampledElevationGrid grid;
        grid.origin = origin;
        grid.resolution = res;
        grid.width = width;
        grid.height = height;
        grid.elevation.assign(grid_size, std::numeric_limits<float>::quiet_NaN());
        grid.variance.assign(grid_size, std::numeric_limits<float>::quiet_NaN());

        for (uint32_t row = 0; row < height; ++row) {
            for (uint32_t col = 0; col < width; ++col) {
                const float x = bound_min[0] + static_cast<float>(col * res);
                const float y = bound_min[1] + static_cast<float>(row * res);
                const auto cell = getElevationUnlocked({x, y});
                if (!cell.has_value()) {
                    continue;
                }
                const size_t idx = static_cast<size_t>(row) * width + col;
                grid.elevation[idx] = cell->height;
                grid.variance[idx] = cell->variance;
            }
        }

        return grid;
    }

    void addContactPoint(const std::array<float, 2>& point) {
        std::lock_guard<std::mutex> lock(mutex_);

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
        while (contact_points_.size() > params_.max_contact_points) {
            contact_points_.pop_back();
        }
    }

    float getMaxRecenterDistance() const {
        return params_.max_recenter_distance;
    }

    float getResolution() const {
        return params_.resolution;
    }

    float getMinContactProbability() const {
        return params_.min_contact_probability;
    }

    float getMinVariance() const {
        return params_.min_variance;
    }

    float getMinStableContactProbability() const {
        return params_.min_stable_contact_probability;
    }

    float getMinStableFootAngularVelocity() const {
        return params_.min_stable_foot_angular_velocity;
    }

    float getMinStableFootLinearVelocity() const {
        return params_.min_stable_foot_linear_velocity;
    }

    void clearContactPoints() {
        contact_points_.clear();
    }

    void interpolateContactPoints() {
        std::lock_guard<std::mutex> lock(mutex_);

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
            std::cout << "Invalid bounding box, clearing contact points" << '\n';
            clearContactPoints();
            return;
        }

        // Interpolate using inverse distance weighting
        const float step = params_.resolution;
        const float power = params_.power;  // Power parameter for IDW

        for (float x = min_x; x <= max_x; x += step) {
            for (float y = min_y; y <= max_y; y += step) {
                std::array<float, 2> point{x, y};
                auto cell = getElevationUnlocked(point);

                // Skip if cell doesn't exist or already has contact
                if (!cell || cell->contact) {
                    continue;
                }

                float sum_weights = 0.0f;
                float weighted_height = 0.0f;
                float weighted_variance = 0.0f;

                // Calculate weighted sum from all contact points
                for (const auto& contact_point : contact_points_) {
                    auto contact_cell = getElevationUnlocked(contact_point);
                    if (!contact_cell) {
                        continue;
                    }

                    // Calculate distance
                    float dx = point[0] - contact_point[0];
                    float dy = point[1] - contact_point[1];
                    float distance = std::sqrt(dx * dx + dy * dy);

                    // Avoid division by zero
                    if (distance < params_.resolution) {
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
                    setElevationUnlocked(point, new_cell);
                }
            }
        }

        if (!contact_points_.empty()) {
            contact_points_.pop_back();
        }
    }

protected:
    virtual void updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                              const std::array<int, 2>& new_origin_i) = 0;

    // Unlocked accessors for callers that already hold mutex_.
    virtual bool setElevationUnlocked(const std::array<float, 2>& loc,
                                      const ElevationCell& elevation) = 0;
    virtual std::optional<ElevationCell> getElevationUnlocked(const std::array<float, 2>& loc) = 0;

    mutable std::mutex mutex_;

    std::array<ElevationCell, map_size> elevation_;
    Params params_;
    ElevationCell default_elevation_;
    std::deque<std::array<float, 2>> contact_points_{};

    std::array<int, 2> local_map_origin_i_{0, 0};
    std::array<int, 2> local_map_bound_max_i_{};
    std::array<int, 2> local_map_bound_min_i_{};
    std::array<float, 2> local_map_origin_d_{0.0, 0.0};
    std::array<float, 2> local_map_bound_max_d_{};
    std::array<float, 2> local_map_bound_min_d_{};

    bool point_feet_{false};
};

// Terrain-only contact debouncer.  This does not change the contact flags used by
// leg odometry; it only protects the terrain map and terrain EKF correction from
// one-sample dropouts such as 1110111 and from the first samples after touchdown.
struct TerrainContactFilterState {
    bool stable{false};
    int on_count{0};
    int off_count{0};
    int age{0};
};

class TerrainContactFilter {
public:
    struct Params {
        int min_on_samples{3};
        int max_dropout_samples{2};
        int min_off_samples{3};
        int skip_after_touchdown_samples{5};
        double stable_contact_threshold{0.5};
        Params() = default;
        Params(const int min_on_samples, const int max_dropout_samples, const int min_off_samples,
               const int skip_after_touchdown_samples, const double stable_contact_threshold)
            : min_on_samples(min_on_samples),
              max_dropout_samples(max_dropout_samples),
              min_off_samples(min_off_samples),
              skip_after_touchdown_samples(skip_after_touchdown_samples),
              stable_contact_threshold(stable_contact_threshold) {}
    };

    TerrainContactFilter() = default;
    explicit TerrainContactFilter(const Params& params) : params_(params) {}
    std::map<std::string, double> filter(
        const std::map<std::string, double>& contacts_probability) {
        became_stable_.clear();

        std::map<std::string, double> filtered;
        for (const auto& [cf, cp] : contacts_probability) {
            const bool stable_contact_candidate = cp > params_.stable_contact_threshold;
            TerrainContactFilterState& s = state_[cf];
            const bool was_stable = s.stable;
            if (stable_contact_candidate) {
                ++s.on_count;
                s.off_count = 0;

                if (!s.stable && s.on_count >= params_.min_on_samples) {
                    s.stable = true;
                    s.age = 0;
                }
            } else {
                ++s.off_count;
                s.on_count = 0;

                // Fill short holes: 1110111 remains stable contact for terrain.
                if (s.stable && s.off_count <= params_.max_dropout_samples) {
                    // keep stable=true
                } else if (s.off_count >= params_.min_off_samples) {
                    s.stable = false;
                    s.age = 0;
                }
            }

            if (s.stable) {
                ++s.age;
            }

            became_stable_[cf] = (!was_stable && s.stable);
            filtered[cf] = (s.stable && s.age > params_.skip_after_touchdown_samples) ? 1.0 : 0.0;
        }

        return filtered;
    }

    bool becameStable(const std::string& cf) {
        const auto jt = became_stable_.find(cf);
        return jt != became_stable_.end() && jt->second;
    }

    void reset() {
        state_.clear();
        became_stable_.clear();
    }

private:
    Params params_;
    std::map<std::string, TerrainContactFilterState> state_;
    std::map<std::string, bool> became_stable_;
};

}  // namespace serow
