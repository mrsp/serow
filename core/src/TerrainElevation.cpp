#include "TerrainElevation.hpp"

#include <algorithm>
#include <unordered_set>

namespace serow {

void TerrainElevation::printMapInformation() const {
    const std::string GREEN = "\033[1;32m";
    const std::string WHITE = "\033[1;37m";
    std::cout << GREEN << "\tresolution: " << resolution << std::endl;
    std::cout << GREEN << "\tinverse resolution: " << resolution_inv << std::endl;
    std::cout << GREEN << "\tlocal map size: " << map_size << std::endl;
    std::cout << GREEN << "\tlocal map half size: " << half_map_size << std::endl;
    std::cout << GREEN << "\tlocal map dim: " << map_dim << std::endl;
    std::cout << GREEN << "\tlocal map half dim: " << half_map_dim << WHITE << std::endl;
}

int TerrainElevation::locationToGlobalIndex(const float loc) const {
    if (loc > 0.0) {
        return static_cast<int>(resolution_inv * loc + 0.5);
    } else {
        return static_cast<int>(resolution_inv * loc - 0.5);
    }
}

std::array<int, 2> TerrainElevation::locationToGlobalIndex(const std::array<float, 2>& loc) const {
    return {locationToGlobalIndex(loc[0]), locationToGlobalIndex(loc[1])};
}

bool TerrainElevation::inside(const std::array<int, 2>& id_g) const {
    if ((abs(id_g[0] - local_map_origin_i_[0]) - half_map_dim) > 0 ||
        (abs(id_g[1] - local_map_origin_i_[1]) - half_map_dim) > 0) {
        return false;
    }
    return true;
}

bool TerrainElevation::inside(const std::array<float, 2>& location) const {
    return inside(locationToGlobalIndex(location));
}

void TerrainElevation::resetCell(const int& hash_id) {
    elevation_[hash_id] = default_elevation_;
}

void TerrainElevation::resetLocalMap() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fill(elevation_.begin(), elevation_.end(), empty_elevation_);
}

void TerrainElevation::updateLocalMap(double timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);  // Single lock for the entire function
    try {
        // Create a temporary state to avoid modifying the shared state until we're done
        LocalMapState local_map_state{};
        local_map_state.timestamp = timestamp;

        // Process the map in chunks to avoid stack overflow
        constexpr size_t chunk_size = 1000;
        for (size_t chunk_start = 0; chunk_start < map_size; chunk_start += chunk_size) {
            size_t chunk_end = std::min(chunk_start + chunk_size, static_cast<size_t>(map_size));

            for (size_t i = chunk_start; i < chunk_end; i++) {
                try {
                    // Validate the index before accessing
                    if (i >= map_size) {
                        std::cerr << "Index " << i << " out of bounds for map_size " << map_size
                                  << std::endl;
                        continue;
                    }

                    // Get the world location and height atomically
                    const auto world_loc = globalIndexToWorldLocation(hashIdToGlobalIndex(i));
                    const auto height = elevation_[i].height;

                    // Store the data
                    local_map_state.data[i] = {world_loc[0], world_loc[1], height};
                } catch (const std::exception& e) {
                    std::cerr << "Error processing cell " << i << ": " << e.what() << std::endl;
                    // Continue with next cell instead of crashing
                    continue;
                }
            }
        }

        // Only update the shared state if we successfully created the new state
        local_map_state_ = std::move(local_map_state);
    } catch (const std::exception& e) {
        std::cerr << "Error in updateLocalMap: " << e.what() << std::endl;
        // Don't update the state if there was an error
    }
}

const LocalMapState& TerrainElevation::getLocalMap() {
    std::lock_guard<std::mutex> lock(mutex_);
    return local_map_state_;
}

void TerrainElevation::initializeLocalMap(const float height, const float variance) {
    std::lock_guard<std::mutex> lock(mutex_);
    default_elevation_ = std::move(ElevationCell(height, variance));
    std::fill(elevation_.begin(), elevation_.end(), default_elevation_);
}

void TerrainElevation::updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
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

void TerrainElevation::clearOutOfMapCells(const std::vector<int>& clear_id) {
    for (const int& x : clear_id) {
        for (int y = -half_map_dim; y <= half_map_dim; y++) {
            const std::array<int, 2> temp_clear_id = {x, y};
            const int hash_id = localIndexToHashId(temp_clear_id);
            if (isHashIdValid(hash_id)) {
                resetCell(hash_id);
            } else {
                continue;
            }
        }
    }
}

void TerrainElevation::recenter(const std::array<float, 2>& location) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Compute the shifting index
    const std::array<int, 2> new_origin_i = locationToGlobalIndex(location);
    const std::array<float, 2> new_origin_d = {new_origin_i[0] * resolution,
                                               new_origin_i[1] * resolution};

    // Compute the delta shift
    const std::array<int, 2> shift_num = {new_origin_i[0] - local_map_origin_i_[0],
                                          new_origin_i[1] - local_map_origin_i_[1]};
    for (size_t i = 0; i < 2; i++) {
        if (fabs(shift_num[i]) > map_dim) {
            // Clear all map
            resetLocalMap();
            updateLocalMapOriginAndBound(new_origin_d, new_origin_i);
            return;
        }
    }

    // Clear the memory out of the map size
    for (size_t i = 0; i < 2; i++) {
        if (shift_num[i] == 0) {
            continue;
        }
        const int min_id_g = -half_map_dim + local_map_origin_i_[i];
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
        clearOutOfMapCells(clear_id);
    }
    updateLocalMapOriginAndBound(new_origin_d, new_origin_i);
}

std::array<float, 2> TerrainElevation::globalIndexToLocation(const std::array<int, 2>& id_g) const {
    return {id_g[0] * resolution, id_g[1] * resolution};
}

std::array<float, 2> TerrainElevation::globalIndexToWorldLocation(
    const std::array<int, 2>& id_g) const {
    return {id_g[0] * resolution + local_map_origin_d_[0],
            id_g[1] * resolution + local_map_origin_d_[1]};
}

std::array<int, 2> TerrainElevation::globalIndexToLocalIndex(const std::array<int, 2>& id_g) const {
    std::array<int, 2> id_l = {};
    for (size_t i = 0; i < 2; i++) {
        id_l[i] = fast_mod<map_dim>(id_g[i]);
        if (id_l[i] > half_map_dim) {
            id_l[i] -= map_dim;
        } else if (id_l[i] < -half_map_dim) {
            id_l[i] += map_dim;
        }
    }
    return id_l;
}

std::array<int, 2> TerrainElevation::localIndexToGlobalIndex(const std::array<int, 2>& id_l) const {
    std::array<int, 2> id_g = {};
    for (size_t i = 0; i < 2; i++) {
        const int min_id_g = -half_map_dim + local_map_origin_i_[i];
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
        int cur_id = cur_dis_to_min_id + min_id_g;
        id_g[i] = cur_id;
    }
    return id_g;
}

std::array<float, 2> TerrainElevation::localIndexToLocation(const std::array<int, 2>& id_l) const {
    std::array<float, 2> loc = {};
    for (size_t i = 0; i < 2; i++) {
        const int min_id_g = -half_map_dim + local_map_origin_i_[i];
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
        loc[i] = cur_id * resolution;
    }
    return loc;
}

std::array<int, 2> TerrainElevation::hashIdToLocalIndex(const int hash_id) const {
    const int id0 = hash_id / map_dim;
    const int id1 = hash_id - id0 * map_dim;
    return {id0 - half_map_dim, id1 - half_map_dim};
}

std::array<int, 2> TerrainElevation::hashIdToGlobalIndex(const int hash_id) const {
    return localIndexToGlobalIndex(hashIdToLocalIndex(hash_id));
}

std::array<float, 2> TerrainElevation::hashIdToLocation(const int hash_id) const {
    return localIndexToLocation(hashIdToLocalIndex(hash_id));
}

int TerrainElevation::localIndexToHashId(const std::array<int, 2>& id_in) const {
    const std::array<int, 2> id = {id_in[0] + half_map_dim, id_in[1] + half_map_dim};
    return id[0] * map_dim + id[1];
}

int TerrainElevation::locationToHashId(const std::array<float, 2>& loc) const {
    return localIndexToHashId(globalIndexToLocalIndex(locationToGlobalIndex(loc)));
}

int TerrainElevation::globalIndexToHashId(const std::array<int, 2>& id_g) const {
    return localIndexToHashId(globalIndexToLocalIndex(id_g));
}

bool TerrainElevation::isHashIdValid(const int id) const {
    if (id > map_size) {
        return false;
    }
    if (id < 0) {
        return false;
    }
    return true;
}

bool TerrainElevation::update(const std::array<float, 2>& loc, float height, float variance) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!inside(loc)) {
        return false;
    }

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
            elevation_[globalIndexToHashId(idx)] = cell;
        }
    }

    return true;
}

std::optional<ElevationCell> TerrainElevation::getElevation(const std::array<float, 2>& loc) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!inside(loc)) {
        return std::nullopt;
    }
    const int hash_id = locationToHashId(loc);
    return elevation_[hash_id];
}

const std::array<float, 2>& TerrainElevation::getMapOrigin() {
    std::lock_guard<std::mutex> lock(mutex_);
    return local_map_origin_d_;
}

}  // namespace serow
