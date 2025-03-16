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

bool NaiveTerrainElevation::interpolate(const std::vector<std::array<float, 2>>& locs,
                                        const std::array<float, 2>& robot_heading,
                                        float look_ahead_distance) {
    if (locs.size() < 3) {
        return false;  // Need at least 3 points to estimate a plane
    }

    // Step 1: Collect known points with their elevations (3D coordinates)
    std::vector<std::array<float, 3>> points3D;
    points3D.reserve(locs.size());

    for (const auto& loc : locs) {
        if (!inside(loc)) {
            return false;
        }
        const std::array<int, 2> idx = locationToGlobalIndex(loc);
        points3D.push_back({loc[0], loc[1], elevation_[idx[0]][idx[1]].height});
    }

    // Step 2: Fit a plane to these points using least squares
    // For a plane equation: ax + by + c = z
    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
    float sum_xx = 0.0f, sum_xy = 0.0f, sum_xz = 0.0f;
    float sum_yy = 0.0f, sum_yz = 0.0f;

    for (const auto& p : points3D) {
        sum_x += p[0];
        sum_y += p[1];
        sum_z += p[2];
        sum_xx += p[0] * p[0];
        sum_xy += p[0] * p[1];
        sum_xz += p[0] * p[2];
        sum_yy += p[1] * p[1];
        sum_yz += p[1] * p[2];
    }

    const float n = static_cast<float>(points3D.size());

    // Solve the system of equations to find plane coefficients
    const float det = sum_xx * sum_yy - sum_xy * sum_xy;
    if (std::abs(det) < 1e-6f) {
        // Determinant too small, points might be collinear
        return false;
    }

    const float a = (sum_xz * sum_yy - sum_yz * sum_xy) / det;
    const float b = (sum_xx * sum_yz - sum_xy * sum_xz) / det;
    const float c = (sum_z - a * sum_x - b * sum_y) / n;

    // Step 3: Calculate the error of the fit
    float total_error = 0.0f;
    for (const auto& p : points3D) {
        const float predicted_z = a * p[0] + b * p[1] + c;
        const float error = p[2] - predicted_z;
        total_error += error * error;
    }
    const float avg_error = std::sqrt(total_error / n);

    // Step 4: Compute influence region (distance from the robot)
    // Find the centroid of foot positions
    float centroid_x = sum_x / n;
    float centroid_y = sum_y / n;

    // Find the maximum distance from centroid to any foot
    float max_foot_distance = 0.0f;
    for (const auto& p : points3D) {
        const float dx = p[0] - centroid_x;
        const float dy = p[1] - centroid_y;
        const float dist = std::sqrt(dx * dx + dy * dy);
        max_foot_distance = std::max(max_foot_distance, dist);
    }

    // Define the influence radius as max distance plus some margin
    const float influence_radius = max_foot_distance * 2.0f;

    // Step 5: Forward-looking prediction based on robot heading
    // Normalize heading vector
    const float heading_norm =
        std::sqrt(robot_heading[0] * robot_heading[0] + robot_heading[1] * robot_heading[1]);
    if (heading_norm < 1e-6f) {
        return false;  // Invalid heading
    }

    const float heading_x = robot_heading[0] / heading_norm;
    const float heading_y = robot_heading[1] / heading_norm;

    // Calculate the slope in the direction of heading
    const float directional_slope = a * heading_x + b * heading_y;

    // Calculate perpendicular direction to heading
    const float perp_x = -heading_y;
    const float perp_y = heading_x;

    // Calculate how much the terrain slopes perpendicular to the heading
    const float perpendicular_slope = a * perp_x + b * perp_y;

    // Define look-ahead region based on robot heading and predicted slope
    const int look_ahead_cells = static_cast<int>(look_ahead_distance * resolution_inv);
    const int look_ahead_width = static_cast<int>(influence_radius * resolution_inv);

    // Calculate extended radius for normal mapping around the robot
    const float max_radius = influence_radius * 1.5f;  // Regular mapping radius
    const int radius_cells = static_cast<int>(max_radius * resolution_inv) + 1;

    // Store points for visualization/debugging (optional)
    std::vector<std::array<float, 3>> prediction_points;

    // Step 6: Apply the mapping in two phases:
    // 6a. Regular circular region around the robot
    const std::array<float, 2> centroid = {centroid_x, centroid_y};
    const std::array<int, 2> centroid_idx = locationToGlobalIndex(centroid);

    // Process a square region centered on the robot
    for (int di = -radius_cells; di <= radius_cells; ++di) {
        for (int dj = -radius_cells; dj <= radius_cells; ++dj) {
            const std::array<int, 2> idx = {centroid_idx[0] + di, centroid_idx[1] + dj};
            if (!inside(idx) || elevation_[idx[0]][idx[1]].contact) {
                continue;  // Skip if outside map or if it's a contact point
            }

            const std::array<float, 2> loc = globalIndexToLocation(idx);
            const float dx = loc[0] - centroid_x;
            const float dy = loc[1] - centroid_y;
            const float dist = std::sqrt(dx * dx + dy * dy);

            // Skip forward area (will be handled by forward prediction)
            const float forward_proj = dx * heading_x + dy * heading_y;
            if (forward_proj > influence_radius * 0.5f && dist > influence_radius) {
                continue;
            }

            if (dist <= max_radius) {
                // Calculate plane height at this location
                const float plane_height = a * loc[0] + b * loc[1] + c;

                // Determine confidence based on distance
                float confidence;
                if (dist <= influence_radius) {
                    // Full confidence within influence radius
                    confidence = 1.0f;
                } else {
                    // Linear blend between influence_radius and max_radius
                    confidence = 1.0f - (dist - influence_radius) / (max_radius - influence_radius);
                }

                // Blend with existing height based on confidence
                const float existing_height = elevation_[idx[0]][idx[1]].height;
                const float new_height =
                    confidence * plane_height + (1.0f - confidence) * existing_height;

                // Update the cell
                elevation_[idx[0]][idx[1]].height = new_height;
                // Variance increases with distance from robot
                elevation_[idx[0]][idx[1]].variance =
                    std::max(avg_error, 0.05f) * (1.0f + dist / influence_radius);
                elevation_[idx[0]][idx[1]].updated = true;
            }
        }
    }

    // 6b. Forward-looking prediction in the direction of travel
    // Create a wedge-shaped prediction in front of the robot
    for (int d = 1; d <= look_ahead_cells; ++d) {
        // Width of the wedge increases with distance
        const int width = static_cast<int>(look_ahead_width * (float(d) / look_ahead_cells));

        for (int w = -width; w <= width; ++w) {
            // Calculate position along the wedge
            const float forward_x = centroid_x + d * resolution * heading_x;
            const float forward_y = centroid_y + d * resolution * heading_y;

            // Position perpendicular to heading
            const float side_x = w * resolution * perp_x;
            const float side_y = w * resolution * perp_y;

            // Combined position
            const float pos_x = forward_x + side_x;
            const float pos_y = forward_y + side_y;

            const std::array<float, 2> pred_loc = {pos_x, pos_y};
            if (!inside(pred_loc)) {
                continue;
            }

            const std::array<int, 2> pred_idx = locationToGlobalIndex(pred_loc);
            if (elevation_[pred_idx[0]][pred_idx[1]].contact) {
                continue;  // Skip contact points
            }

            // Distance from robot centroid
            const float dx = pos_x - centroid_x;
            const float dy = pos_y - centroid_y;
            const float dist = std::sqrt(dx * dx + dy * dy);

            // Calculate the base height using the fitted plane
            const float plane_height = a * pos_x + b * pos_y + c;

            // Adjust for uncertainty in prediction with distance
            // Lateral uncertainty increases with distance
            const float lateral_factor = 1.0f + std::abs(float(w) / width) * 0.5f;

            // Forward uncertainty increases with distance
            const float forward_factor = 1.0f + float(d) / look_ahead_cells;

            // Combined uncertainty factor
            const float uncertainty_factor = lateral_factor * forward_factor;

            // Calculate confidence (decreases with distance)
            const float confidence = 1.0f - float(d) / (look_ahead_cells * 1.2f);

            // Blend predicted height with existing height
            const float existing_height = elevation_[pred_idx[0]][pred_idx[1]].height;
            const float new_height =
                confidence * plane_height + (1.0f - confidence) * existing_height;

            // Update the cell
            elevation_[pred_idx[0]][pred_idx[1]].height = new_height;

            // Variance increases with distance from robot and is higher for predictions
            elevation_[pred_idx[0]][pred_idx[1]].variance =
                std::max(avg_error, 0.05f) * uncertainty_factor * (1.0f + dist / influence_radius);
            elevation_[pred_idx[0]][pred_idx[1]].updated = true;
            // Store prediction point for visualization (optional)
            prediction_points.push_back({pos_x, pos_y, new_height});
        }
    }

    return true;
}

}  // namespace serow
