#include "LocalTerrainMapper.hpp"

namespace serow {

bool LocalTerrainMapper::inside(const std::array<float, 2>& location) const {
    return inside(locationToGlobalIndex(location));
}

bool LocalTerrainMapper::inside(const std::array<int, 2>& id_g) const {
    int x = abs(id_g[0] - local_map_origin_i_[0]);
    int y = abs(id_g[1] - local_map_origin_i_[1]);
    if ((x - half_map_dim) > 0 || (y - half_map_dim) > 0) {
        return false;
    }
    return true;
}

void LocalTerrainMapper::updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
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

void LocalTerrainMapper::clearOutOfMapCells(const std::vector<int>& clear_id, const int i) {
    std::array<int, 2> ids{i, (i + 1) % 2};
    for (const int& x : clear_id) {
        for (int y = -half_map_dim; y <= half_map_dim; y++) {
            std::array<int, 2> temp_clear_id{};
            temp_clear_id[ids[0]] = x;
            temp_clear_id[ids[1]] = y;
            const int hash_id = localIndexToHashId(temp_clear_id);
            resetCell(hash_id);
        }
    }
}

void LocalTerrainMapper::recenter(const std::array<float, 2>& loc) {
    // Compute the shifting index
    const std::array<int, 2> new_origin_i = locationToGlobalIndex(loc);
    const std::array<float, 2> new_origin_d = globalIndexToLocation(new_origin_i);

    // Compute the delta shift
    const std::array<int, 2> shift_num = {new_origin_i[0] - local_map_origin_i_[0],
                                          new_origin_i[1] - local_map_origin_i_[1]};

    std::lock_guard<std::mutex> lock(mutex_);
    // If shift is too large, reset the entire map
    if (std::abs(shift_num[0]) >= map_dim || std::abs(shift_num[1]) >= map_dim) {
        resetLocalMap();
        updateLocalMapOriginAndBound(new_origin_d, new_origin_i);
        return;
    }

    // Clear the memory out of the map size
    for (size_t i = 0; i < 2; i++) {
        if (shift_num[i] == 0) {
            continue;
        }
        const int min_id_g = local_map_bound_min_i_[i];
        const int min_id_l = fast_mod<map_dim>(min_id_g);
        std::vector<int> clear_id;
        if (shift_num[i] > 0) {
            // forward shift, the min id should be cut
            for (int k = 0; k < shift_num[i]; k++) {
                int temp_id = min_id_l + k;
                temp_id = normalize(temp_id);
                clear_id.push_back(temp_id);
            }
        } else {
            // backward shift, the max should be shifted
            for (int k = -1; k >= shift_num[i]; k--) {
                int temp_id = min_id_l + k;
                temp_id = normalize(temp_id);
                clear_id.push_back(temp_id);
            }
        }

        if (clear_id.empty()) {
            continue;
        }
        clearOutOfMapCells(clear_id, i);
    }

    // Update the map origin and bounds
    updateLocalMapOriginAndBound(new_origin_d, new_origin_i);
}

// Conversion functions
int LocalTerrainMapper::localIndexToHashId(const std::array<int, 2>& id_in) const {
    const std::array<int, 2> id = {id_in[0] + half_map_dim, id_in[1] + half_map_dim};
    return id[0] * map_dim + id[1];
}

int LocalTerrainMapper::locationToGlobalIndex(const float loc) const {
    if (loc >= 0.0) {
        return static_cast<int>(resolution_inv * loc + 0.5);
    } else {
        return static_cast<int>(resolution_inv * loc - 0.5);
    }
}

std::array<int, 2> LocalTerrainMapper::locationToGlobalIndex(
    const std::array<float, 2>& loc) const {
    return {locationToGlobalIndex(loc[0]), locationToGlobalIndex(loc[1])};
}

std::array<float, 2> LocalTerrainMapper::globalIndexToLocation(
    const std::array<int, 2>& id_g) const {
    return {static_cast<float>(id_g[0]) * resolution, static_cast<float>(id_g[1]) * resolution};
}

std::array<int, 2> LocalTerrainMapper::globalIndexToLocalIndex(
    const std::array<int, 2>& id_g) const {
    std::array<int, 2> id_l = {};
    for (size_t i = 0; i < 2; i++) {
        // Apply modulo to keep within bounds
        id_l[i] = fast_mod<map_dim>(id_g[i]);
        // Adjust to keep within [-half_map_dim, half_map_dim]
        if (id_l[i] > half_map_dim) {
            id_l[i] -= map_dim;
        } else if (id_l[i] < -half_map_dim) {
            id_l[i] += map_dim;
        }
    }
    return id_l;
}

std::array<int, 2> LocalTerrainMapper::localIndexToGlobalIndex(
    const std::array<int, 2>& id_l) const {
    std::array<int, 2> id_g = {};
    for (size_t i = 0; i < 2; i++) {
        int min_id_g = local_map_bound_min_i_[i];
        int min_id_l = fast_mod<map_dim>(min_id_g);
        if (min_id_l > half_map_dim) {
            min_id_l -= map_dim;
        }
        if (min_id_l < -half_map_dim) {
            min_id_l += map_dim;
        }
        int cur_dis_to_min_id = id_l[i] - min_id_l;
        if (cur_dis_to_min_id < 0) {
            cur_dis_to_min_id += map_dim;
        }
        const int cur_id = cur_dis_to_min_id + min_id_g;
        id_g[i] = cur_id;
    }
    return id_g;
}

std::array<float, 2> LocalTerrainMapper::localIndexToLocation(
    const std::array<int, 2>& id_l) const {
    return globalIndexToLocation(localIndexToGlobalIndex(id_l));
}

std::array<int, 2> LocalTerrainMapper::hashIdToLocalIndex(const int hash_id) const {
    const int id0 = hash_id / map_dim;
    const int id1 = hash_id - id0 * map_dim;
    return {id0 - half_map_dim, id1 - half_map_dim};
}

std::array<int, 2> LocalTerrainMapper::hashIdToGlobalIndex(const int hash_id) const {
    return localIndexToGlobalIndex(hashIdToLocalIndex(hash_id));
}

std::array<float, 2> LocalTerrainMapper::hashIdToLocation(const int hash_id) const {
    return localIndexToLocation(hashIdToLocalIndex(hash_id));
}

int LocalTerrainMapper::locationToHashId(const std::array<float, 2>& loc) const {
    const std::array<int, 2> idx = locationToGlobalIndex(loc);
    return globalIndexToHashId(idx);
}

int LocalTerrainMapper::globalIndexToHashId(const std::array<int, 2>& id_g) const {
    return localIndexToHashId(globalIndexToLocalIndex(id_g));
}

bool LocalTerrainMapper::update(const std::array<float, 2>& loc, float height, float variance) {
    if (!inside(loc)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);

    variance = std::max(variance, min_terrain_height_variance_);
    const std::array<int, 2> center_idx = locationToGlobalIndex(loc);
    const int center_hash_id = globalIndexToHashId(center_idx);
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
            const auto hash_id = globalIndexToHashId(idx);
            elevation_[hash_id] = cell;
        }
    }
    return true;
}

std::optional<ElevationCell> LocalTerrainMapper::getElevation(const std::array<float, 2>& loc) {
    std::lock_guard lock(mutex_);

    if (!inside(loc)) {
        return std::nullopt;
    }

    const int hash_id = locationToHashId(loc);

    // Add bounds check!
    if (hash_id < 0 || hash_id >= static_cast<int>(elevation_.size())) {
        return std::nullopt;
    }

    return elevation_[hash_id];
}

void LocalTerrainMapper::initializeLocalMap(const float height, const float variance,
                                            const float min_variance,
                                            const float max_recenter_distance,
                                            const size_t max_contact_points,
                                            const float min_contact_probability) {
    default_elevation_ = std::move(ElevationCell(height, variance));
    min_terrain_height_variance_ = min_variance;
    max_contact_points_ = max_contact_points;
    max_recenter_distance_ = max_recenter_distance;

    // Make sure the max recenter distance is within the map bounds
    const float max_recenter_distance_bound = 0.5f * half_map_dim * resolution;
    if (max_recenter_distance_ > max_recenter_distance_bound) {
        max_recenter_distance_ = max_recenter_distance_bound;
        std::cout << "Max recenter distance is too large, setting to " << max_recenter_distance_
                  << std::endl;
    }

    // Make sure the min contact probability is within the range [0, 1]
    if (min_contact_probability < 0.0f || min_contact_probability > 1.0f) {
        min_contact_probability_ = 0.15f;
        std::cout << "Min contact probability is out of range, setting to "
                  << min_contact_probability_ << std::endl;
    }

    resetLocalMap();
}

void LocalTerrainMapper::resetCell(const int& hash_id) {
    if (!isHashIdValid(hash_id)) {
        return;
    }
    elevation_[hash_id] = default_elevation_;
}

void LocalTerrainMapper::resetLocalMap() {
    for (size_t i = 0; i < map_size; i++) {
        elevation_[i] = default_elevation_;
    }
    clearContactPoints();
}

bool LocalTerrainMapper::isHashIdValid(const int id) const {
    if (id >= map_size) {
        return false;
    }
    if (id < 0) {
        return false;
    }
    return true;
}

bool LocalTerrainMapper::setElevation(const std::array<float, 2>& loc,
                                      const ElevationCell& elevation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!inside(loc)) {
        return false;
    }
    const int idx = locationToHashId(loc);

    // Add bounds check!
    if (idx < 0 || idx >= static_cast<int>(elevation_.size())) {
        return false;
    }

    elevation_[idx] = elevation;
    return true;
}

}  // namespace serow
