#include "TerrainElevation.hpp"
#include <algorithm>  // For std::find

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

void TerrainElevation::resetCell(const int& hash_id) { elevation_[hash_id] = default_elevation_; }

void TerrainElevation::resetLocalMap() {
    std::fill(elevation_.begin(), elevation_.end(), empty_elevation_);
}

void TerrainElevation::initializeLocalMap(const float height, const float variance) {
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

int TerrainElevation::isHashIdValid(const int id) const {
    if (id > map_size) {
        return false;
    }
    if (id < 0) {
        return false;
    }
    return true;
}

bool TerrainElevation::update(const std::array<float, 2>& loc, float height, float variance) {
    if (!inside(loc)) {
        return false;
    }

    // Update the center cell using Kalman filter
    const int center_hash_id = locationToHashId(loc);
    const float height_prev = elevation_[center_hash_id].height;
    const float variance_prev = elevation_[center_hash_id].variance;

    // Kalman update
    const float kalman_gain = variance_prev / (variance_prev + variance);
    ElevationCell new_elevation;
    new_elevation.height = height_prev + kalman_gain * (height - height_prev);
    new_elevation.variance = (1.0f - kalman_gain) * variance_prev;
    elevation_[center_hash_id] = new_elevation;

    // Pre-compute constants for the weight function
    const float radius_squared = radius * radius;
    const float weight_factor = -1.0f / (2.0f * radius_squared);

    // Calculate the radius in grid cells (rounded up)
    const int radius_cells = static_cast<int>(std::ceil(radius / resolution));

    // Get the center cell's indices
    const std::array<int, 2> center_idx = locationToGlobalIndex(loc);

    // Iterate only over the cells within the radius
    for (int i = -radius_cells; i <= radius_cells; ++i) {
        for (int j = -radius_cells; j <= radius_cells; ++j) {
            // Skip the center cell (already updated)
            if (i == 0 && j == 0) {
                continue;
            }

            // Calculate squared distance in grid coordinates
            const float dx = i * resolution;
            const float dy = j * resolution;
            const float dist_squared = dx * dx + dy * dy;

            // Skip cells outside the circle
            if (dist_squared > radius_squared) {
                continue;
            }

            // Calculate global index and check if inside map
            const std::array<int, 2> global_idx = {center_idx[0] + i, center_idx[1] + j};
            if (!inside(global_idx)) {
                continue;
            }

            // Calculate hash ID directly from global index (avoid extra conversions)
            const int hash_id = globalIndexToHashId(global_idx);

            // Calculate weight (avoid sqrt when possible)
            const float weight = std::exp(dist_squared * weight_factor);

            // Get the current cell values
            float cell_height = elevation_[hash_id].height;
            float cell_variance = elevation_[hash_id].variance;
            
            // Apply Kalman filter update to neighboring cells with diminishing influence
            const float effective_variance = new_elevation.variance / weight;
            const float neighbor_kalman_gain = cell_variance / (cell_variance + effective_variance);
            
            // Update the neighboring cell
            elevation_[hash_id].height = cell_height + 
                neighbor_kalman_gain * weight * (new_elevation.height - cell_height);
            elevation_[hash_id].variance = (1.0f - neighbor_kalman_gain * weight) * cell_variance;
        }
    }

    return true;
}

bool TerrainElevation::interpolate(const std::vector<std::array<float, 2>>& locs) {
    
    if (locs.empty()) {
        return false;
    }

    std::vector<int> hash_ids;
    float x = 0;
    float y = 0;
    float max_x = -1e4;
    float max_y = -1e4;

    for (const auto& loc : locs) {
        if (!inside(loc)) {
            return false;
        }

        hash_ids.push_back(locationToHashId(loc));
        x += loc[0];
        y += loc[1];
        
        if (loc[0] > max_x) {
            max_x = loc[0];
        } 
        if (loc[1] > max_y) {
            max_y = loc[1];
        }
    }
    x /= locs.size();
    y /= locs.size();

    const float region_radius = std::max(abs(x - max_x), abs(y - max_y)) + radius;
    
    // Calculate the radius in grid cells (rounded up)
    const int radius_cells = static_cast<int>(std::ceil(region_radius / resolution));
   
    // Get the center cell's indices
    const std::array<int, 2> center_idx = locationToGlobalIndex({x, y});

    // Iterate only over the cells within the radius
    std::vector<float> distances;
    distances.reserve(locs.size());
    std::vector<double> weights;
    weights.reserve(locs.size());

    for (int i = -radius_cells; i <= radius_cells; ++i) {
        for (int j = -radius_cells; j <= radius_cells; ++j) {
            const std::array<int, 2> id_g = {center_idx[0] + i, center_idx[1] + j};

            const int hash_id = globalIndexToHashId(id_g);
            if (std::find(hash_ids.begin(), hash_ids.end(), hash_id) != hash_ids.end()) {
                continue;
            }
            distances.clear();
            weights.clear();
            float min_distance = 1e4;
            std::array<float, 2> min_loc = {};
            for (const auto& loc : locs) {
                const std::array<float, 2> curr_loc = globalIndexToLocation(id_g);
                const float dx = curr_loc[0] - loc[0];
                const float dy = curr_loc[1] - loc[1];
                const float distance = std::sqrt(dx * dx + dy * dy);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_loc = loc;
                }
                distances.push_back(distance);
            }

            if (min_distance < 1e-4 && inside(min_loc)) {
                const int min_hash_id = locationToHashId(min_loc); 
                elevation_[hash_id] = elevation_[min_hash_id];
                continue;
            }

            for (const auto& distance : distances) {
                weights.push_back(1.0 / (distance * distance));
            }
            
            float sum_weights = 0.0;
            for (const auto& weight : weights) {
                sum_weights += weight;
            }
            
            for (auto& weight : weights) {
                weight /= sum_weights;
            }

            // Calculate weighted average for values
            float weighted_sum_values = 0.0;
            for (size_t k = 0; k < weights.size(); ++k) {
                weighted_sum_values += weights[k] * elevation_[hash_ids[k]].height;
            }
            
            // Variance propagation - using squared weights and variances
            float propagated_variance = 0.0;
            for (size_t k = 0; k < weights.size(); ++k) {
                propagated_variance += weights[k] * weights[k] * elevation_[hash_ids[k]].variance;  // Squared weights
            }
            elevation_[hash_id] = {weighted_sum_values, propagated_variance};
        }
    }
    return true;
}

std::optional<ElevationCell> TerrainElevation::getElevation(const std::array<float, 2>& loc) const {
    if (!inside(loc)) {
        return std::nullopt;
    }
    const int hash_id = locationToHashId(loc);
    return elevation_[hash_id];
}

const std::array<float, 2>& TerrainElevation::getMapOrigin() const {
    return local_map_origin_d_;
}

}  // namespace serow
