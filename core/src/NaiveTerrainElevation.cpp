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

    const std::array<int, 2> idx = locationToGlobalIndex(loc);
    ElevationCell& cell = elevation_[idx[0]][idx[1]];

    // If the cell is uninitialized (empty), set it directly
    if (cell.variance > 10.0) {
        cell.height = height;
        cell.variance = variance;
        return true;
    }

    // Kalman filter update for the target cell
    const float prior_variance = cell.variance;
    const float prior_height = cell.height;

    // Ensure variances are positive to avoid division issues
    const float effective_variance = std::max(variance, 1e-6f);
    const float effective_prior_variance = std::max(prior_variance, 1e-6f);

    // Compute Kalman gain
    const float kalman_gain = effective_prior_variance / (effective_prior_variance + effective_variance);

    // Update height and variance
    cell.height = prior_height + kalman_gain * (height - prior_height);
    cell.variance = (1.0f - kalman_gain) * effective_prior_variance;
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
        return false; // Need at least 2 points for interpolation
    }

    // Step 1: Collect known points and map them to grid cells
    struct LocationData {
        float height;
        float variance;
        float weight_sum; // Accumulated weight for averaging
    };
    std::unordered_map<int64_t, LocationData> influence_map; // Key: packed (i,j), Value: aggregated data

    const float scale_sq = resolution * resolution; // Intrinsic scale based on resolution
    const int influence_range = 3; // Fixed range in cells (e.g., 3x resolution)

    for (const auto& loc : locs) {
        if (!inside(loc)) {
            return false;
        }
        const std::array<int, 2> center_idx = locationToGlobalIndex(loc);
        const float height = elevation_[center_idx[0]][center_idx[1]].height;
        const float variance = elevation_[center_idx[0]][center_idx[1]].variance;
        const std::array<float, 2> center_loc = globalIndexToLocation(center_idx);

        // Step 2: Propagate influence to nearby cells
        for (int di = -influence_range; di <= influence_range; ++di) {
            for (int dj = -influence_range; dj <= influence_range; ++dj) {
                const std::array<int, 2> idx = {center_idx[0] + di, center_idx[1] + dj};
                if (!inside(idx)) {
                    continue;
                }

                // Pack indices into a single key (assuming map_dim < 2^31)
                const int64_t key = static_cast<int64_t>(idx[0]) << 32 | idx[1];

                // Compute weight based on distance
                const std::array<float, 2> cell_loc = globalIndexToLocation(idx);
                const float dx = cell_loc[0] - center_loc[0];
                const float dy = cell_loc[1] - center_loc[1];
                const float dist_sq = std::max(dx * dx + dy * dy, 1e-6f);
                const float weight = 1.0f / (dist_sq + scale_sq);

                // Aggregate influence
                auto& data = influence_map[key];
                data.height += weight * height;
                data.variance += weight * variance;
                data.weight_sum += weight;
            }
        }
    }

    // Step 3: Apply interpolated values to the map
    for (const auto& [key, data] : influence_map) {
        const int i = static_cast<int>(key >> 32);
        const int j = static_cast<int>(key & 0xFFFFFFFF);
        
        // Skip if this is a known point (preserve original data)
        bool is_known = false;
        for (const auto& loc : locs) {
            const std::array<int, 2> idx = locationToGlobalIndex(loc);
            if (idx[0] == i && idx[1] == j) {
                is_known = true;
                break;
            }
        }
        if (is_known) {
            continue;
        }

        // Normalize and update
        if (data.weight_sum > 0.0f) {
            elevation_[i][j].height = data.height / data.weight_sum;
            elevation_[i][j].variance = data.variance /data.weight_sum;
        }
    }

    return true;
}


}  // namespace serow
