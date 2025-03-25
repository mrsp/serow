#include "DebugLogger.hpp"

namespace serow {

DebugLogger::DebugLogger(const std::string& log_file_path) {
    // Create the log file
    log_file_.open(log_file_path);
}

void DebugLogger::log(const BaseState& base_state) {}

void DebugLogger::log(const CentroidalState& centroidal_state) {}

void DebugLogger::log(const ContactState& contact_state) {}

void DebugLogger::log(const ImuMeasurement& imu_measurement) {}

void DebugLogger::log(const std::map<std::string, JointMeasurement>& joints_measurement) {}

void DebugLogger::log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurement) {}

DebugLogger::~DebugLogger() { log_file_.close(); }

}  // namespace serow
