/**
 * Copyright (C) 2024 Stylianos Piperakis, Ownage Dynamics L.P.
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
#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <serow/TerrainElevation.hpp>

namespace serow {

class TerrainElevationTest : public ::testing::Test {
   protected:
    TerrainElevation te;
};

TEST_F(TerrainElevationTest, CheckTerrainElevation) {
   te.printMapInformation();
   te.recenter({0.0, 0.0});

   std::array<float, 2> location = {1.0, 2.0};
   auto id_g = te.locationToGlobalIndex(location);
   std::cout << "Global index " << id_g[0] << " " << id_g[1] << std::endl;
   auto id_l = te.globalIndexToLocalIndex(id_g);
   std::cout << "Local index " << id_l[0] << " " << id_l[1] << std::endl;
   auto hash_id = te.localIndexToHashId(id_l);
   std::cout << "Hash index " << hash_id << std::endl;

   location = {0.3, -2.0};
   id_g = te.locationToGlobalIndex(location);
   std::cout << "Global index " << id_g[0] << " " << id_g[1] << std::endl;
   id_l = te.globalIndexToLocalIndex(id_g);
   std::cout << "Local index " << id_l[0] << " " << id_l[1] << std::endl;
   hash_id = te.localIndexToHashId(id_l);
   std::cout << "Hash index " << hash_id << std::endl;
}

}
