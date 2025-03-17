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

    variance += min_terrain_height_variance_;
    const std::array<int, 2> center_idx = locationToGlobalIndex(loc);
    ElevationCell& cell = elevation_[center_idx[0]][center_idx[1]];
    cell.contact = true;
    cell.updated = true;
    const int64_t key = static_cast<int64_t>(center_idx[0]) << 32 | center_idx[1];
    contact_cells.push_back(key);

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
    
    const int radius_cells = static_cast<int>(radius * resolution_inv) + 1;
    // Process a square region centered on the robot
    for (int di = -radius_cells; di <= radius_cells; ++di) {
        for (int dj = -radius_cells; dj <= radius_cells; ++dj) {
            if (di == 0 && dj == 0) {
                continue;
            }
            const std::array<int, 2> idx = {center_idx[0] + di, center_idx[1] + dj};
            if (!inside(idx)) {
                continue; 
            }
            elevation_[idx[0]][idx[1]] = cell;
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

}  // namespace serow
