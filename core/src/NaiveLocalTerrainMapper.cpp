#include "NaiveLocalTerrainMapper.hpp"

#include <algorithm>

namespace serow {

// Coordinate conversion functions
int NaiveLocalTerrainMapper::locationToGlobalIndex(const float loc) const {
    if (loc > 0.0) {
        return static_cast<int>(resolution_inv * loc + 0.5);
    } else {
        return static_cast<int>(resolution_inv * loc - 0.5);
    }
}

std::array<int, 2> NaiveLocalTerrainMapper::locationToGlobalIndex(
    const std::array<float, 2>& loc) const {
    return {locationToGlobalIndex(loc[0]), locationToGlobalIndex(loc[1])};
}

std::array<float, 2> NaiveLocalTerrainMapper::globalIndexToLocation(
    const std::array<int, 2>& id_g) const {
    return {(id_g[0] * resolution), (id_g[1] * resolution)};
}

std::array<int, 2> NaiveLocalTerrainMapper::globalIndexToLocalIndex(
    const std::array<int, 2>& id_g) const {
    return {id_g[0] + half_map_dim - local_map_origin_i_[0], id_g[1] + half_map_dim - local_map_origin_i_[1]};
}

std::array<int, 2> NaiveLocalTerrainMapper::localIndexToGlobalIndex(
    const std::array<int, 2>& id_l) const {
    return {id_l[0] + local_map_origin_i_[0] - half_map_dim, id_l[1] + local_map_origin_i_[1] - half_map_dim};
}

std::array<int, 2> NaiveLocalTerrainMapper::locationToLocalIndex(
    const std::array<float, 2>& loc) const {
    return globalIndexToLocalIndex(locationToGlobalIndex(loc));
}

std::array<float, 2> NaiveLocalTerrainMapper::localIndexToLocation(
    const std::array<int, 2>& id_l) const {
    return globalIndexToLocation(localIndexToGlobalIndex(id_l));
}

bool NaiveLocalTerrainMapper::inside(const std::array<int, 2>& id_g) const {
    int x = abs(id_g[0] - local_map_origin_i_[0]);
    int y = abs(id_g[1] - local_map_origin_i_[1]);
    if ((x - half_map_dim) > 0 || (y - half_map_dim) > 0) {
        return false;
    }
    return true;
}

bool NaiveLocalTerrainMapper::inside(const std::array<float, 2>& loc) const {
    const std::array<int, 2> id_g = locationToGlobalIndex(loc);
    return inside(id_g);
}

void NaiveLocalTerrainMapper::resetLocalMap() {
    for (int i = 0; i < map_size; ++i) {
        elevation_[i] = default_elevation_;
    }
    clearContactPoints();
}

void NaiveLocalTerrainMapper::initializeLocalMap(const float height, const float variance,
                                                 const float min_variance) {
    default_elevation_ = ElevationCell(height, variance);
    min_terrain_height_variance_ = min_variance;
    for (int i = 0; i < map_size; ++i) {
        elevation_[i] = default_elevation_;
    }
    updateLocalMapOriginAndBound({0.0f, 0.0f}, {0, 0});
}

bool NaiveLocalTerrainMapper::update(const std::array<float, 2>& loc, float height,
                                     float variance) {
    if (!inside(loc)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);

    variance = std::max(min_terrain_height_variance_, variance);
    const std::array<int, 2> center_idx = locationToGlobalIndex(loc);
    const std::array<int, 2> center_local_idx = globalIndexToLocalIndex(center_idx);
    const int center_hash_id = localIndexToHashId(center_local_idx);
    ElevationCell& cell = elevation_[center_hash_id];
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
            const std::array<int, 2> local_idx = globalIndexToLocalIndex(idx);
            const int hash_id = localIndexToHashId(local_idx);
            elevation_[hash_id] = cell;
        }
    }
    return true;
}

void NaiveLocalTerrainMapper::recenter(const std::array<float, 2>& loc) {
    const std::array<int, 2> new_origin_i = locationToGlobalIndex(loc);
    const std::array<int, 2> shift = {new_origin_i[0] - local_map_origin_i_[0],
                                      new_origin_i[1] - local_map_origin_i_[1]};

    std::lock_guard<std::mutex> lock(mutex_);
    // If shift is too large, reset the entire map
    if (std::abs(shift[0]) >= map_dim || std::abs(shift[1]) >= map_dim) {
        resetLocalMap();
        updateLocalMapOriginAndBound(loc, new_origin_i);
        return;
    }

    // Create a temporary copy of the current data
    std::array<ElevationCell, map_size> temp_map;
    for (int i = 0; i < map_size; ++i) {
        temp_map[i] = elevation_[i];
    }

    // Reset all cells to default
    resetLocalMap();

    // Move the data according to the shift
    for (int old_i = 0; old_i < map_dim; ++old_i) {
        for (int old_j = 0; old_j < map_dim; ++old_j) {
            // Calculate the new position after shifting
            int new_i = old_i - shift[0];
            int new_j = old_j - shift[1];

            // Check if the new position is within the map bounds
            if (new_i >= 0 && new_i < map_dim && new_j >= 0 && new_j < map_dim) {
                // Move the data from the old position to the new position
                const int new_hash_id = localIndexToHashId({new_i, new_j});
                const int old_hash_id = localIndexToHashId({old_i, old_j});
                elevation_[new_hash_id] = temp_map[old_hash_id];
            }
        }
    }

    // Update the map origin and bounds
    updateLocalMapOriginAndBound(loc, new_origin_i);
}

std::optional<ElevationCell> NaiveLocalTerrainMapper::getElevation(
    const std::array<float, 2>& loc) {
    if (!inside(loc)) {
        return std::nullopt;
    }
    const int hash_id = locationToHashId(loc);
    std::lock_guard<std::mutex> lock(mutex_);
    return elevation_[hash_id];
}

bool NaiveLocalTerrainMapper::setElevation(const std::array<float, 2>& loc,
                                           const ElevationCell& elevation) {
    if (!inside(loc)) {
        return false;
    }
    const int hash_id = locationToHashId(loc);
    std::lock_guard<std::mutex> lock(mutex_);
    elevation_[hash_id] = elevation;
    return true;
}

void NaiveLocalTerrainMapper::updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                                           const std::array<int, 2>& new_origin_i) {
    // update local map origin and local map bound
    local_map_origin_i_ = new_origin_i;
    local_map_origin_d_ = new_origin_d;

    local_map_bound_max_i_ = {local_map_origin_i_[0] + half_map_dim,
                              local_map_origin_i_[1] + half_map_dim};
    local_map_bound_min_i_ = {local_map_origin_i_[0] - half_map_dim,
                              local_map_origin_i_[1] - half_map_dim};

    // the float map bound only consider the closed cell center
    local_map_bound_min_d_ = globalIndexToLocation(local_map_bound_min_i_);
    local_map_bound_max_d_ = globalIndexToLocation(local_map_bound_max_i_);
}

int NaiveLocalTerrainMapper::globalIndexToHashId(const std::array<int, 2>& id_g) const {
    const std::array<int, 2> id_l = globalIndexToLocalIndex(id_g);
    return localIndexToHashId(id_l);
}

int NaiveLocalTerrainMapper::localIndexToHashId(const std::array<int, 2>& id_l) const {
    return id_l[0] * map_dim + id_l[1];
}

int NaiveLocalTerrainMapper::locationToHashId(const std::array<float, 2>& loc) const {
    const std::array<int, 2> id_g = locationToGlobalIndex(loc);
    const std::array<int, 2> id_l = globalIndexToLocalIndex(id_g);
    return localIndexToHashId(id_l);
}

std::array<int, 2> NaiveLocalTerrainMapper::hashIdToLocalIndex(const int hash_id) const {
    return {hash_id / map_dim, hash_id % map_dim};
}

std::array<float, 2> NaiveLocalTerrainMapper::hashIdToLocation(const int hash_id) const {
    const std::array<int, 2> id_l = hashIdToLocalIndex(hash_id);
    return localIndexToLocation(id_l);
}

}  // namespace serow
