#include "TerrainElevation.hpp"

namespace serow {

void TerrainElevation::printMapInformation() const {
    const std::string GREEN = "\033[1;32m";
    std::cout << GREEN << "\tresolution: " << resolution << std::endl;
    std::cout << GREEN << "\tinverse resolution: " << resolution_inv << std::endl;
    std::cout << GREEN << "\tlocal map size : " << map_size << std::endl;
    std::cout << GREEN << "\thalf local map size: " << half_map_size << std::endl;
    std::cout << GREEN << "\tlocal map dim: " << map_dim[0] << " " << map_dim[1] << std::endl;
    std::cout << GREEN << "\tlocal map half dim: " << half_map_dim[0] << " " << half_map_dim[1]
              << std::endl;
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
    if ((abs(id_g[0] - local_map_origin_i_[0]) - half_map_dim[0]) > 0 ||
        (abs(id_g[1] - local_map_origin_i_[1]) - half_map_dim[1]) > 0) {
        return false;
    }
    return true;
}

bool TerrainElevation::inside(const std::array<float, 2>& location) const {
    return inside(locationToGlobalIndex(location));
}

void TerrainElevation::resetCell(const int& hash_id) { height_[hash_id] = default_height_; }

void TerrainElevation::resetLocalMap() { std::fill(height_.begin(), height_.end(), 0.0); }

void TerrainElevation::initializeLocalMap(const float height) {
    std::fill(height_.begin(), height_.end(), height);
    default_height_ = height;
}

void TerrainElevation::updateLocalMapOriginAndBound(const std::array<float, 2>& new_origin_d,
                                                    const std::array<int, 2>& new_origin_i) {
    // update local map origin and local map bound
    local_map_origin_i_ = new_origin_i;
    local_map_origin_d_ = new_origin_d;

    local_map_bound_max_i_ = {local_map_origin_i_[0] + half_map_dim[0],
                              local_map_origin_i_[1] + half_map_dim[1]};
    local_map_bound_min_i_ = {local_map_origin_i_[0] - half_map_dim[0],
                              local_map_origin_i_[1] - half_map_dim[1]};

    // the float map bound only consider the closed cell center
    local_map_bound_min_d_ = globalIndexToLocation(local_map_bound_min_i_);
    local_map_bound_max_d_ = globalIndexToLocation(local_map_bound_max_i_);
}

void TerrainElevation::clearOutOfMapCells(const std::vector<int>& clear_id, const int& i) {
    std::vector<int> ids{i, (i + 1) % 2};
    for (const auto& x : clear_id) {
        for (int y = -half_map_dim[ids[1]]; y <= half_map_dim[ids[1]]; y++) {
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
        if (fabs(shift_num[i]) > map_dim[i]) {
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
        const int min_id_g = -half_map_dim[i] + local_map_origin_i_[i];
        const int min_id_l = min_id_g % map_dim[i];
        std::vector<int> clear_id;
        if (shift_num[i] > 0) {
            // forward shift, the min id should be cut
            for (int k = 0; k < shift_num[i]; k++) {
                int temp_id = min_id_l + k;
                temp_id = normalize(temp_id, -half_map_dim[i], half_map_dim[i]);
                clear_id.push_back(temp_id);
            }
        } else {
            // backward shift, the max should be shifted
            for (int k = -1; k >= shift_num[i]; k--) {
                int temp_id = min_id_l + k;
                temp_id = normalize(temp_id, -half_map_dim[i], half_map_dim[i]);
                clear_id.push_back(temp_id);
            }
        }

        if (clear_id.empty()) {
            continue;
        }
        clearOutOfMapCells(clear_id, i);
    }
    updateLocalMapOriginAndBound(new_origin_d, new_origin_i);
}

std::array<float, 2> TerrainElevation::globalIndexToLocation(const std::array<int, 2>& id_g) const {
    return {id_g[0] * resolution, id_g[1] * resolution};
}

std::array<int, 2> TerrainElevation::globalIndexToLocalIndex(const std::array<int, 2>& id_g) const {
    std::array<int, 2> id_l = {};

    for (size_t i = 0; i < 2; i++) {
        id_l[i] = id_g[i] % map_dim[i];
        if (id_l[i] > half_map_dim[i]) {
            id_l[i] -= map_dim[i];
        } else if (id_l[i] < -half_map_dim[i]) {
            id_l[i] += map_dim[i];
        }
    }
    return id_l;
}

std::array<int, 2> TerrainElevation::localIndexToGlobalIndex(const std::array<int, 2>& id_l) const {
    std::array<int, 2> id_g = {};
    for (size_t i = 0; i < 2; i++) {
        int min_id_g = -half_map_dim[i] + local_map_origin_i_[i];
        int min_id_l = min_id_g % map_dim[i];

        if (min_id_l > half_map_dim[i]) {
            min_id_l -= map_dim[i];
        }
        if (min_id_l < -half_map_dim[i]) {
            min_id_l += map_dim[i];
        }

        int cur_dis_to_min_id = id_l[i] - min_id_l;
        if (cur_dis_to_min_id < 0) {
            cur_dis_to_min_id += map_dim[i];
        }
        int cur_id = cur_dis_to_min_id + min_id_g;
        id_g[i] = cur_id;
    }
    return id_g;
}

std::array<float, 2> TerrainElevation::localIndexToLocation(const std::array<int, 2>& id_l) const {
    std::array<float, 2> loc = {};
    for (size_t i = 0; i < 2; i++) {
        int min_id_g = -half_map_dim[i] + local_map_origin_i_[i];
        int min_id_l = min_id_g % map_dim[i];
        if (min_id_l > half_map_dim[i]) {
            min_id_l -= map_dim[i];
        }
        if (min_id_l < -half_map_dim[i]) {
            min_id_l += map_dim[i];
        }

        int cur_dis_to_min_id = id_l[i] - min_id_l;
        if (cur_dis_to_min_id < 0) {
            cur_dis_to_min_id += map_dim[i];
        }
        int cur_id = cur_dis_to_min_id + min_id_g;
        loc[i] = cur_id * resolution;
    }
    return loc;
}

std::array<int, 2> TerrainElevation::hashIdToLocalIndex(const int hash_id) const {
    int id0 = hash_id / map_dim[1];
    int id1 = hash_id - id0 * map_dim[1];
    return {id0 - half_map_dim[0], id1 - half_map_dim[1]};
}

std::array<int, 2> TerrainElevation::hashIdToGlobalIndex(const int hash_id) const {
    return localIndexToGlobalIndex(hashIdToLocalIndex(hash_id));
}

std::array<float, 2> TerrainElevation::hashIdToLocation(const int hash_id) const {
    return localIndexToLocation(hashIdToLocalIndex(hash_id));
}

int TerrainElevation::localIndexToHashId(const std::array<int, 2>& id_in) const {
    const std::array<int, 2> id = {id_in[0] + half_map_dim[0], id_in[1] + half_map_dim[1]};
    return id[0] * map_dim[1] + id[1];
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

int TerrainElevation::normalize(int x, int a, int b) const {
    const int range = b - a + 1;
    const int y = (x - a) % range;
    if (y < 0) {
        return y + range + a;
    } else {
        return y + a;
    }
}

bool TerrainElevation::update(const std::array<float, 2>& loc, float height) {
    if (!inside(loc)) {
        return false;
    }
    const int hash_id = locationToHashId(loc);
    height_[hash_id] = height;
    return true;
}

std::optional<float> TerrainElevation::getHeight(const std::array<float, 2>& loc) const {
    if (!inside(loc)) {
        return std::nullopt;
    }
    
    const int hash_id = locationToHashId(loc);
    return height_[hash_id];
}


}  // namespace serow
