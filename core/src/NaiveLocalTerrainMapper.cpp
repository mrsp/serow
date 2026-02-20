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
    return {id_g[0] + half_map_dim - local_map_origin_i_[0],
            id_g[1] + half_map_dim - local_map_origin_i_[1]};
}

std::array<int, 2> NaiveLocalTerrainMapper::localIndexToGlobalIndex(
    const std::array<int, 2>& id_l) const {
    return {id_l[0] + local_map_origin_i_[0] - half_map_dim,
            id_l[1] + local_map_origin_i_[1] - half_map_dim};
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
                                                 const float min_variance,
                                                 const float max_recenter_distance,
                                                 const size_t max_contact_points,
                                                 const float min_contact_probability) {
    default_elevation_ = ElevationCell(height, variance);
    min_terrain_height_variance_ = min_variance;
    max_contact_points_ = max_contact_points;
    max_recenter_distance_ = max_recenter_distance;
    min_contact_probability_ = min_contact_probability;

    // Make sure the max recenter distance is within the map bounds
    const float max_recenter_distance_bound = 0.5f * half_map_dim * resolution;
    if (max_recenter_distance_ > max_recenter_distance_bound) {
        max_recenter_distance_ = max_recenter_distance_bound;
        std::cout << "Max recenter distance is too large, setting to " << max_recenter_distance_
                  << std::endl;
    }

    // Make sure the min contact probability is within the range [0, 1]
    if (min_contact_probability_ < 0.0f || min_contact_probability_ > 1.0f) {
        min_contact_probability_ = 0.15f;
        std::cout << "Min contact probability is out of range, setting to "
                  << min_contact_probability_ << std::endl;
    }

    for (int i = 0; i < map_size; ++i) {
        elevation_[i] = default_elevation_;
    }
    updateLocalMapOriginAndBound({0.0f, 0.0f}, {0, 0});
}

bool NaiveLocalTerrainMapper::update(const std::array<float, 2>& loc, float height,
                                     float variance, std::optional<std::array<float, 3>> normal) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!inside(loc)) {
        return false;
    }

    variance = std::max(min_terrain_height_variance_, variance);
    const std::array<int, 2> center_idx = locationToGlobalIndex(loc);
    const std::array<int, 2> center_local_idx = globalIndexToLocalIndex(center_idx);
    const int center_hash_id = localIndexToHashId(center_local_idx);
    if (center_hash_id < 0 || center_hash_id >= map_size) {
        return false;
    }
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

    // Process a region around the contact point 
    int rc = radius_cells;
    float nx_over_nz = 0.0f;
    float ny_over_nz = 0.0f;
    if (normal.has_value()) {
        rc = radius_cells * 2;
        nx_over_nz = normal.value()[0] / normal.value()[2];
        ny_over_nz = normal.value()[1] / normal.value()[2];
    }

    const float d_max = rc * resolution;
    const float dist_variance_gain_ = 100.0f / (d_max * d_max);
    for (int di = -rc; di <= rc; ++di) {
        for (int dj = -rc; dj <= rc; ++dj) {
            if (di == 0 && dj == 0) continue;
    
            const std::array<int, 2> idx = {center_idx[0] + di, center_idx[1] + dj};
            if (!inside(idx)) continue;
    
            const std::array<int, 2> local_idx = globalIndexToLocalIndex(idx);
            const int hash_id = localIndexToHashId(local_idx);
            if (hash_id < 0 || hash_id >= map_size) continue;
    
            const std::array<float, 2> cell_xy = globalIndexToLocation(idx);
            const float dx = cell_xy[0] - loc[0];
            const float dy = cell_xy[1] - loc[1];
            const float dist2 = dx * dx + dy * dy;
    
            // Inflate measurement variance with distance (e.g. exponential or linear)
            const float sigma_scale = 1.0f + dist_variance_gain_ * dist2;
            const float neighbor_meas_variance = variance * sigma_scale;
    
            // Compute predicted height 
            const float predicted_height = cell.height - nx_over_nz * dx - ny_over_nz * dy;
    
            // Run a proper Kalman update on the neighbor cell
            ElevationCell& neighbor = elevation_[hash_id];
            const float S = neighbor.variance + neighbor_meas_variance;
            const float K = neighbor.variance / S;
            neighbor.height   = neighbor.height + K * (predicted_height - neighbor.height);
            neighbor.variance = (1.0f - K) * neighbor.variance;
            neighbor.updated  = true;
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
    std::lock_guard<std::mutex> lock(mutex_);
    if (!inside(loc)) {
        return std::nullopt;
    }
    const int hash_id = locationToHashId(loc);
    if (hash_id < 0 || hash_id >= map_size) {
        return std::nullopt;
    }
    return elevation_[hash_id];
}

bool NaiveLocalTerrainMapper::setElevation(const std::array<float, 2>& loc,
                                           const ElevationCell& elevation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!inside(loc)) {
        return false;
    }
    const int hash_id = locationToHashId(loc);
    if (hash_id < 0 || hash_id >= map_size) {
        return false;
    }
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
