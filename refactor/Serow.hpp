/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#pragma once
#include "JointEstimator.hpp"
#include "State.hpp"
#include "ContactEKF.hpp"
#include "Mahony.hpp"
#include "LegOdometry.hpp"
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <string>

namespace serow {

struct JointMeasurement {
    double timestamp{};
    double position{};
}



class Serow {
    public:
    Serow(std::string config);
    private:
        std::unordered_map<std::string, JointEstimator> joint_estimators_;
        State state_;
        ContactEKF base_estimator_;
        CoMEKF com_estimator_;
        std::unique_ptr<Mahony> attitude_estimator_;
        std::unique_ptr<RobotKinematics> kinematic_estimator_;
        std::unique_ptr<LegOdometry> leg_odometry_;
};

}  // namespace serow