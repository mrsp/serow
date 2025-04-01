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

std::array<int, 2> NaiveTerrainElevation::globalLocationToMapLocation(const std::array<float, 2>& global_location, const float yaw) const {

    const std::array<float, 2> origin_offset = {
        global_location[0] - local_map_origin_d_[0],
        global_location[1] - local_map_origin_d_[1]
    };

    return {origin_offset[0] * std::cos(yaw) + origin_offset[1] * std::sin(yaw), 
            -origin_offset[1] * std::sin(yaw) + origin_offset[0] * std::cos(yaw)};
}

int NaiveTerrainElevation::mapLocationToGlobalIndex(const float loc) const {
    if (loc > 0.0) {
        return static_cast<int>(resolution_inv * loc + 0.5);
    } else {
        return static_cast<int>(resolution_inv * loc - 0.5);
    }
}

std::array<int, 2> NaiveTerrainElevation::mapLocationToGlobalIndex(const std::array<float, 2>& loc) const {
    return {mapLocationToGlobalIndex(loc[0]), mapLocationToGlobalIndex(loc[1])};
}

std::array<float, 2> NaiveTerrainElevation::globalIndexToMapLocation(
    const std::array<int, 2>& id_g) const {
    return {(id_g[0] * resolution) + local_map_origin_d_[0], 
            (id_g[1] * resolution) + local_map_origin_d_[1]};
}

bool NaiveTerrainElevation::inside(const std::array<int, 2>& id_g) const {
    if ((abs(id_g[0] - local_map_origin_i_[0]) - half_map_dim) > 0 ||
        (abs(id_g[1] - local_map_origin_i_[1]) - half_map_dim) > 0) {
        return false;
    }
    return true;
}

bool NaiveTerrainElevation::inside(const std::array<float, 2>& location) const {
    return inside(locationToGlobalIndex(location));
}


std::array<int, 2> NaiveTerrainElevation::globalIndexToLocalIndex(const std::array<int, 2>& id_g) const {
    return {id_g[0] - local_map_origin_i_[0], id_g[1] - local_map_origin_i_[1]};
}


void NaiveTerrainElevation::resetCell(const int i, const int j) {
    elevation_[i][j] = default_elevation_;
}

void NaiveTerrainElevation::resetLocalMap() {
    for (int i = 0; i < map_dim; ++i) {
        for (int j = 0; j < map_dim; ++j) {
            elevation_[i][j] = default_elevation_;
        }
    }
}

void NaiveTerrainElevation::initializeLocalMap(const float height, const float variance,
                                               const float min_variance) {
    default_elevation_ = ElevationCell(height, variance);
    min_terrain_height_variance_ = min_variance;
    for (int i = 0; i < map_dim; ++i) {
        for (int j = 0; j < map_dim; ++j) {
            elevation_[i][j] = default_elevation_;
        }
    }
    updateLocalMapOriginAndBound({0.0f, 0.0f});
}

void NaiveTerrainElevation::updateLocalMapPose(const std::array<float, 2>& new_origin, float new_yaw) {
    origin_ = new_origin;
    yaw_ = new_yaw;
}

bool NaiveTerrainElevation::update(const std::array<float, 2>& loc, float height, float variance,
                                   double timestamp) {
    if (!inside(loc)) {
        return false;
    }

    variance = std::max(min_terrain_height_variance_, variance);
    const std::array<int, 2> center_idx = mapLocationToGlobalIndex(loc);
    ElevationCell& cell = elevation_[center_idx[0]][center_idx[1]];
    cell.contact = true;
    cell.updated = true;

    // Kalman filter update for the target cell
    const float prior_variance = cell.variance;
    const float prior_height = cell.height;

    // Ensure variances are positive to avoid division issues
    const float effective_variance = std::max(variance, 1e-6f);
    const float effective_prior_variance = std::max(prior_variance, 1e-6f);

    // Compute Kalman gain
    const float kalman_gain =
        effective_prior_variance / (effective_prior_variance + effective_variance);

    // Update height and variance
    cell.height = prior_height + kalman_gain * (height - prior_height);
    cell.variance = (1.0f - kalman_gain) * effective_prior_variance;

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
    timestamp_ = timestamp;
    return true;
}

void NaiveTerrainElevation::recenter(const std::array<float, 2>& loc) {
    const std::array<int, 2> new_origin_i = globalLocationToMapLocation(loc) * resolution_inv;
    const std::array<int, 2> shift = {
        new_origin_i[0] - local_map_origin_i_[0],
        new_origin_i[1] - local_map_origin_i_[1]
    };
    
    // If shift is too large, reset the entire map
    if (std::abs(shift[0]) >= map_dim || std::abs(shift[1]) >= map_dim) {
        resetLocalMap();
        updateLocalMapOriginAndBound(loc, new_origin_i);
        return;
    }
    
    // Create a temporary copy of the current data
    ElevationCell temp_map[map_dim][map_dim];
    for (int i = 0; i < map_dim; ++i) {
        for (int j = 0; j < map_dim; ++j) {
            temp_map[i][j] = elevation_[i][j];
        }
    }
    
    // Clear the current map first
    resetLocalMap();
    
    // Shift the data to new positions
    for (int i = 0; i < map_dim; ++i) {
        for (int j = 0; j < map_dim; ++j) {
            // Calculate source position in the temp map
            const int src_i = i + shift[0];
            const int src_j = j + shift[1];
            
            // If source position is valid, copy data from temp map to the new position
            if (src_i >= 0 && src_i < map_dim && src_j >= 0 && src_j < map_dim) {
                elevation_[i][j] = temp_map[src_i][src_j];
            }
        }
    }
    
    // Update the map origin and bounds
    updateLocalMapOriginAndBound(loc, new_origin_i);
}

std::optional<ElevationCell> NaiveTerrainElevation::getElevation(
    const std::array<float, 2>& loc) const {
    if (!inside(loc)) {
        return std::nullopt;
    }
    const std::array<int, 2> idx = locationToGlobalIndex(loc);
    const std::array<int, 2> array_idx = globalIndexToArrayIndex(idx);
    return elevation_[array_idx[0]][array_idx[1]];
}

const std::array<float, 2>& NaiveTerrainElevation::getMapOrigin() const {
    return local_map_origin_d_;
}

bool NaiveTerrainElevation::setElevation(const std::array<float, 2>& loc, const ElevationCell& elevation) {
    if (!inside(loc)) {
        return false;
    }   
    const std::array<int, 2> idx = locationToGlobalIndex(loc);
    const std::array<int, 2> array_idx = globalIndexToArrayIndex(idx);
    elevation_[array_idx[0]][array_idx[1]] = elevation;
    return true;
}

}  // namespace serow
