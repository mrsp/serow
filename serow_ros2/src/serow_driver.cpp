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
#include "SerowRos2.hpp"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    try {
        auto serow_ros2_driver = std::make_shared<SerowRos2>();
        rclcpp::spin(serow_ros2_driver);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("serow_ros2"), "Error: %s", e.what());
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
