#include "NaiveTerrainElevation.hpp"

#include <algorithm>
#include <unordered_set>

namespace serow {


void NaiveTerrainElevation::printMapInformation() const {
    const std::string GREEN = "\033[1;32m";
    const std::string WHITE = "\033[1;37m";
    std::cout << GREEN << "\tresolution: " << resolution << std::endl;
    std::cout << GREEN << "\tinverse resolution: " << resolution_inv << std::endl;
    std::cout << GREEN << "\tlocal map size: " << map_size << std::endl;
    std::cout << GREEN << "\tlocal map half size: " << half_map_size << std::endl;
    std::cout << GREEN << "\tlocal map dim: " << map_dim << std::endl;
    std::cout << GREEN << "\tlocal map half dim: " << half_map_dim << WHITE << std::endl;
}

int NaiveTerrainElevation::locationToGlobalIndex(const float loc) const {
    return static_cast<int>(resolution_inv * loc) + half_map_dim;
}

std::array<int, 2> NaiveTerrainElevation::locationToGlobalIndex(
    const std::array<float, 2>& loc) const {
    return {locationToGlobalIndex(loc[0]), locationToGlobalIndex(loc[1])};
}

std::array<float, 2> NaiveTerrainElevation::globalIndexToLocation(
    const std::array<int, 2>& id_g) const {
    return {(id_g[0] - half_map_dim) * resolution, (id_g[1] - half_map_dim) * resolution};
}

bool NaiveTerrainElevation::inside(const std::array<int, 2>& id_g) const {
    return (id_g[0] >= 0 && id_g[0] < map_dim && id_g[1] >= 0 && id_g[1] < map_dim);
}

bool NaiveTerrainElevation::inside(const std::array<float, 2>& location) const {
    return inside(locationToGlobalIndex(location));
}

void NaiveTerrainElevation::resetCell(const int i, const int j) {
    elevation_[i][j] = default_elevation_;
}

void NaiveTerrainElevation::resetLocalMap() {
    for (int i = 0; i < map_dim; ++i) {
        for (int j = 0; j < map_dim; ++j) {
            elevation_[i][j] = empty_elevation_;
        }
    }
}

void NaiveTerrainElevation::initializeLocalMap(const float height, const float variance) {
    default_elevation_ = ElevationCell(height, variance);
    for (int i = 0; i < map_dim; ++i) {
        for (int j = 0; j < map_dim; ++j) {
            elevation_[i][j] = default_elevation_;
        }
    }
}

void NaiveTerrainElevation::updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                                         const std::array<int, 2>& new_origin_i) {
    local_map_origin_i_ = new_origin_i;
    local_map_origin_d_ = new_origin_d;
    local_map_bound_max_i_ = {map_dim - 1, map_dim - 1};
    local_map_bound_min_i_ = {0, 0};
    local_map_bound_min_d_ = globalIndexToLocation(local_map_bound_min_i_);
    local_map_bound_max_d_ = globalIndexToLocation(local_map_bound_max_i_);
}

bool NaiveTerrainElevation::update(const std::array<float, 2>& loc, float height, float variance) {
    if (!inside(loc)) {
        return false;
    }

    const std::array<int, 2> center_idx = locationToGlobalIndex(loc);
    const float height_prev = elevation_[center_idx[0]][center_idx[1]].height;
    const float variance_prev = elevation_[center_idx[0]][center_idx[1]].variance;

    const float kalman_gain = variance_prev / (variance_prev + variance);
    ElevationCell new_elevation;
    new_elevation.height = height_prev + kalman_gain * (height - height_prev);
    new_elevation.variance = (1.0f - kalman_gain) * variance_prev;
    elevation_[center_idx[0]][center_idx[1]] = new_elevation;

    std::cout << "Cell to be updated " << center_idx[0] << " " << center_idx[1] << " for loc " << loc[0] << " " << loc[1] << std::endl;
    std::cout << "Height prev " << height_prev << " variance prev " << variance_prev << std::endl;
    std::cout << "Height mes " << height << " variance mes " << variance << std::endl;
    std::cout << "Height upd " << new_elevation.height << " variance upd " << new_elevation.variance << std::endl;
    std::cout << "-------------------------------" << std::endl;

    const float radius_squared = radius * radius;
    const float weight_factor = -1.0f / (2.0f * radius_squared);
    const int radius_cells = static_cast<int>(std::ceil(radius / resolution));

    for (int i = -radius_cells; i <= radius_cells; ++i) {
        for (int j = -radius_cells; j <= radius_cells; ++j) {
            if (i == 0 && j == 0) {
                continue;
            }

            const float dx = i * resolution;
            const float dy = j * resolution;
            const float dist_squared = dx * dx + dy * dy;

            if (dist_squared > radius_squared) {
                continue;
            }

            const std::array<int, 2> global_idx = {center_idx[0] + i, center_idx[1] + j};
            if (!inside(global_idx)) {
                continue;
            }

            const float weight = std::exp(dist_squared * weight_factor);
            float cell_height = elevation_[global_idx[0]][global_idx[1]].height;
            float cell_variance = elevation_[global_idx[0]][global_idx[1]].variance;

            const float effective_variance = new_elevation.variance / weight;
            const float neighbor_kalman_gain = cell_variance / (cell_variance + effective_variance);

            elevation_[global_idx[0]][global_idx[1]].height =
                cell_height + neighbor_kalman_gain * weight * (new_elevation.height - cell_height);
            elevation_[global_idx[0]][global_idx[1]].variance =
                (1.0f - neighbor_kalman_gain * weight) * cell_variance;
        }
    }

    return true;
}

std::optional<ElevationCell> NaiveTerrainElevation::getElevation(
    const std::array<float, 2>& loc) const {
    if (!inside(loc)) {
        return std::nullopt;
    }
    const std::array<int, 2> idx = locationToGlobalIndex(loc);
    return elevation_[idx[0]][idx[1]];
}

const std::array<float, 2>& NaiveTerrainElevation::getMapOrigin() const {
    return local_map_origin_d_;
}

bool NaiveTerrainElevation::interpolate(const std::vector<std::array<float, 2>>& locs) {
    if (locs.size() < 2) {
        return false; // Need at least 2 points for meaningful interpolation
    }

    // Step 1: Determine the bounding box of locations
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    // Store locations and their indices for fast access
    std::vector<std::array<int, 2>> known_indices;
    known_indices.reserve(locs.size());

    for (const auto& loc : locs) {
        if (!inside(loc)) {
            return false; // All points must be within the map
        }

        min_x = std::min(min_x, loc[0]);
        min_y = std::min(min_y, loc[1]);
        max_x = std::max(max_x, loc[0]);
        max_y = std::max(max_y, loc[1]);

        known_indices.push_back(locationToGlobalIndex(loc));
    }

    // Step 2: Add a buffer around the bounding box
    float buffer = radius * 2.0f; // Buffer distance beyond the points
    min_x -= buffer;
    min_y -= buffer;
    max_x += buffer;
    max_y += buffer;

    // Step 3: Convert bounds to grid indices
    std::array<int, 2> min_idx = locationToGlobalIndex({min_x, min_y});
    std::array<int, 2> max_idx = locationToGlobalIndex({max_x, max_y});

    // Ensure we stay within the map bounds
    min_idx[0] = std::max(0, std::min(map_dim - 1, min_idx[0]));
    min_idx[1] = std::max(0, std::min(map_dim - 1, min_idx[1]));
    max_idx[0] = std::max(0, std::min(map_dim - 1, max_idx[0]));
    max_idx[1] = std::max(0, std::min(map_dim - 1, max_idx[1]));

    // Step 4: Pre-compute location data structure for quicker lookup
    struct LocationData {
        std::array<float, 2> loc;
        std::array<int, 2> idx;
        float height;
        float variance;
    };
    std::vector<LocationData> location_data;
    location_data.reserve(locs.size());

    for (size_t i = 0; i < locs.size(); ++i) {
        const auto& loc = locs[i];
        const auto& idx = known_indices[i];
        location_data.push_back({
            loc,
            idx,
            elevation_[idx[0]][idx[1]].height,
            elevation_[idx[0]][idx[1]].variance
        });
    }

    // Step 5: Iterate through cells in the bounded region and interpolate
    const int k_nearest = std::min(5, static_cast<int>(locs.size())); // Use at most 5 nearest neighbors

    for (int i = min_idx[0]; i <= max_idx[0]; ++i) {
        for (int j = min_idx[1]; j <= max_idx[1]; ++j) {
            const std::array<int, 2> curr_idx = {i, j};

            // Skip if this is one of our known points
            bool is_known = false;
            for (const auto& idx : known_indices) {
                if (idx[0] == i && idx[1] == j) {
                    is_known = true;
                    break;
                }
            }
            if (is_known) {
                continue; // Preserve known points
            }

            // Get the current cell's location
            const std::array<float, 2> curr_loc = globalIndexToLocation(curr_idx);

            // Find k nearest neighbors
            std::vector<std::pair<float, size_t>> distances; // (distance squared, index in location_data)
            distances.reserve(location_data.size());

            for (size_t k = 0; k < location_data.size(); ++k) {
                const auto& loc_data = location_data[k];
                const float dx = curr_loc[0] - loc_data.loc[0];
                const float dy = curr_loc[1] - loc_data.loc[1];
                const float dist_sq = dx * dx + dy * dy;
                distances.push_back({dist_sq, k});
            }

            // Sort by distance (partial sort for efficiency)
            std::partial_sort(distances.begin(),
                              distances.begin() + std::min(k_nearest, static_cast<int>(distances.size())),
                              distances.end());

            // Limit to k nearest
            distances.resize(std::min(k_nearest, static_cast<int>(distances.size())));

            // Check for near-exact match
            if (!distances.empty() && distances[0].first < 1e-6f) {
                const auto& loc_data = location_data[distances[0].second];
                elevation_[i][j].height = loc_data.height;
                elevation_[i][j].variance = loc_data.variance;
                continue;
            }

            // Perform inverse distance weighted interpolation
            float sum_weights = 0.0f;
            float weighted_sum_heights = 0.0f;
            float sum_squared_weights = 0.0f;
            float weight_variance_product = 0.0f;

            const float min_dist_sq = 1e-6f; // Avoid division by zero

            for (const auto& [dist_sq, idx] : distances) {
                const float clamped_dist_sq = std::max(dist_sq, min_dist_sq);
                // Use Gaussian weighting based on radius, similar to update()
                const float weight = std::exp(-clamped_dist_sq / (2.0f * radius * radius));
                const auto& loc_data = location_data[idx];

                sum_weights += weight;
                weighted_sum_heights += weight * loc_data.height;
                sum_squared_weights += weight * weight;
                weight_variance_product += weight * weight * loc_data.variance;
            }

            if (sum_weights > 0.0f) {
                // Interpolate height
                float interpolated_height = weighted_sum_heights / sum_weights;

                // Interpolate variance with distance-based uncertainty
                float interpolated_variance = weight_variance_product / sum_squared_weights;
                float min_dist = std::sqrt(distances[0].first);
                float distance_uncertainty = min_dist * 0.1f; // 10% of min distance as uncertainty
                interpolated_variance += distance_uncertainty * distance_uncertainty;

                // Update the cell
                elevation_[i][j].height = interpolated_height;
                elevation_[i][j].variance = interpolated_variance;
            }
        }
    }

    return true;
}


}  // namespace serow
