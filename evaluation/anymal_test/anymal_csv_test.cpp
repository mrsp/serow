#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include "serow/Serow.hpp"

using json = nlohmann::json;

namespace {

// -----------------------------------------------------------------------------
// Default paths. You can override all of these from the command line:
//   ./anymal_csv_test [serow_config] [sensor_csv] [output_csv] [tum_output] [joint_suffix]
// Example:
//   ./anymal_csv_test anymal.json \
//     /home/michael/github/serow/evaluation/anymal_test/anymal_data/test/cyn-1/sensor_data.csv \
//     /home/michael/github/serow/evaluation/anymal_test/anymal_data/test/cyn-1/serow/fused_state.csv \
//     /home/michael/github/serow/evaluation/anymal_test/anymal_data/test/cyn-1/serow/serow_traj_tum.csv
//
// joint_suffix is useful only if your Pinocchio model joint names are e.g.
// "LF_HAA_joint" instead of "LF_HAA". For the columns you gave, the default
// assumes the model joint names are LF_HAA, LF_HFE, LF_KFE, ...
// -----------------------------------------------------------------------------
const std::string DEFAULT_CONFIG = "anymal.json";
const std::string DEFAULT_SENSOR_CSV =
    "/home/michael/github/serow/evaluation/anymal_test/anymal_data/test/cyn-1/anymal_data.csv";
const std::string DEFAULT_OUTPUT_CSV =
    "/home/michael/github/serow/evaluation/anymal_test/anymal_data/test/cyn-1/serow/fused_state.csv";
const std::string DEFAULT_OUTPUT_TUM =
    "/home/michael/github/serow/evaluation/anymal_test/anymal_data/test/cyn-1/serow/serow_traj_tum.csv";

// false = use SEROW/momentum-observer contact estimator
// true  = use binary contact_LF/RF/LH/RH flags from anymal_data.csv
const bool USE_CONTACT_FLAGS = true;
const std::vector<std::string> JOINTS = {
    "LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"};

const std::vector<std::string> LEGS = {"LF", "RF", "LH", "RH"};

std::string trim(const std::string& s) {
    auto start = std::find_if_not(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
    auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) { return std::isspace(c); }).base();
    if (start >= end) return "";
    return std::string(start, end);
}

std::vector<std::string> split(const std::string& line, char delimiter) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, delimiter)) out.push_back(trim(item));
    return out;
}

std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
    return s;
}

bool containsToken(const std::string& frame, const std::string& token) {
    const std::string f = upper(frame);
    const std::string t = upper(token);
    return f.find(t) != std::string::npos;
}

std::optional<std::ifstream> openConfigForFootFrames(const std::string& config_path) {
    // Try exactly as given.
    {
        std::ifstream f(config_path);
        if (f.is_open()) return std::move(f);
    }

    // Try relative to SEROW_PATH, if available.
    const char* env_p = std::getenv("SEROW_PATH");
    if (env_p != nullptr) {
        const std::filesystem::path root(env_p);
        const std::vector<std::filesystem::path> candidates = {
            root / config_path,
            root / "config" / config_path,
            root / "configs" / config_path,
            root / "resources" / config_path};
        for (const auto& p : candidates) {
            std::ifstream f(p.string());
            if (f.is_open()) return std::move(f);
        }
    }

    return std::nullopt;
}

std::map<std::string, std::string> loadLegToFootFrame(const std::string& config_path) {
    // Defaults for common ANYmal URDF/config naming.
    std::map<std::string, std::string> leg_to_frame = {
        {"LF", "LF_FOOT"}, {"RF", "RF_FOOT"}, {"LH", "LH_FOOT"}, {"RH", "RH_FOOT"}};

    auto cfg_stream_opt = openConfigForFootFrames(config_path);
    if (!cfg_stream_opt.has_value()) {
        std::cerr << "Warning: could not open config to read foot_frames. "
                  << "Using defaults LF_FOOT/RF_FOOT/LH_FOOT/RH_FOOT.\n";
        return leg_to_frame;
    }

    json cfg;
    cfg_stream_opt.value() >> cfg;
    if (!cfg.contains("foot_frames")) return leg_to_frame;

    std::vector<std::string> frames;
    const auto& ff = cfg["foot_frames"];
    if (ff.is_array()) {
        for (const auto& x : ff) frames.push_back(x.get<std::string>());
    } else if (ff.is_object()) {
        for (auto it = ff.begin(); it != ff.end(); ++it) frames.push_back(it.value().get<std::string>());
    }

    for (const auto& frame : frames) {
        if (containsToken(frame, "LF") || containsToken(frame, "FL")) leg_to_frame["LF"] = frame;
        else if (containsToken(frame, "RF") || containsToken(frame, "FR")) leg_to_frame["RF"] = frame;
        else if (containsToken(frame, "LH") || containsToken(frame, "HL") || containsToken(frame, "RL")) leg_to_frame["LH"] = frame;
        else if (containsToken(frame, "RH") || containsToken(frame, "HR") || containsToken(frame, "RR")) leg_to_frame["RH"] = frame;
    }

    std::cout << "Foot-frame mapping used:\n";
    for (const auto& leg : LEGS) std::cout << "  contact_" << leg << " -> " << leg_to_frame.at(leg) << "\n";
    return leg_to_frame;
}

std::map<std::string, size_t> makeHeaderIndex(const std::vector<std::string>& header) {
    std::map<std::string, size_t> idx;
    for (size_t i = 0; i < header.size(); ++i) idx[header[i]] = i;
    return idx;
}

void requireColumn(const std::map<std::string, size_t>& idx, const std::string& name) {
    if (idx.count(name) == 0) throw std::runtime_error("Missing required CSV column: " + name);
}

double valueAt(const std::vector<std::string>& row,
               const std::map<std::string, size_t>& idx,
               const std::string& col) {
    const auto it = idx.find(col);
    if (it == idx.end()) throw std::runtime_error("Column not found: " + col);
    if (it->second >= row.size()) throw std::runtime_error("Row too short at column: " + col);
    return std::stod(row[it->second]);
}

serow::ImuMeasurement makeImu(const std::vector<std::string>& row,
                              const std::map<std::string, size_t>& idx) {
    const double t = valueAt(row, idx, "t");
    serow::ImuMeasurement imu;
    imu.timestamp = t;
    imu.angular_velocity = Eigen::Vector3d(
        valueAt(row, idx, "imu_wx"), valueAt(row, idx, "imu_wy"), valueAt(row, idx, "imu_wz"));
    imu.linear_acceleration = Eigen::Vector3d(
        valueAt(row, idx, "imu_ax"), valueAt(row, idx, "imu_ay"), valueAt(row, idx, "imu_az"));
    return imu;
}

std::map<std::string, serow::JointMeasurement>
makeJoints(const std::vector<std::string>& row,
           const std::map<std::string, size_t>& idx,
           const std::string& joint_suffix) {
    const double t = valueAt(row, idx, "t");
    std::map<std::string, serow::JointMeasurement> joints;
    for (const auto& j : JOINTS) {
        serow::JointMeasurement jm;
        jm.timestamp = t;
        jm.position = valueAt(row, idx, "joint_pos_" + j);
        jm.velocity = valueAt(row, idx, "joint_vel_" + j);
        jm.effort = valueAt(row, idx, "joint_eff_" + j);
        joints[j + joint_suffix] = jm;
    }
    return joints;
}

std::map<std::string, serow::ForceTorqueMeasurement>
makeDummyFootForces(const std::map<std::string, std::string>& leg_to_frame, double t) {
    std::map<std::string, serow::ForceTorqueMeasurement> ft;
    for (const auto& [leg, frame] : leg_to_frame) {
        serow::ForceTorqueMeasurement m;
        m.timestamp = t;
        m.force = Eigen::Vector3d::Zero();
        // Safe even for point feet; prevents a missing-torque exception if point_feet=false.
        m.torque = Eigen::Vector3d::Zero();
        ft[frame] = m;
    }
    return ft;
}

std::map<std::string, serow::ContactMeasurement>
makeBinaryContacts(const std::vector<std::string>& row,
                   const std::map<std::string, size_t>& idx,
                   const std::map<std::string, std::string>& leg_to_frame) {
    std::map<std::string, serow::ContactMeasurement> contacts;
    for (const auto& leg : LEGS) {
        const double flag = valueAt(row, idx, "contact_" + leg);
        contacts[leg_to_frame.at(leg)] = (flag > 0.5) ? 1.0 : 0.0;
    }
    return contacts;
}

void writeCsvHeader(std::ofstream& out) {
    out << "t,x,y,z,qx,qy,qz,qw,vx,vy,vz,"
        << "com_x,com_y,com_z,com_vx,com_vy,com_vz,"
        << "bias_ax,bias_ay,bias_az,bias_gx,bias_gy,bias_gz,"
        << "contact_LF,contact_RF,contact_LH,contact_RH\n";
}

void writeTumLine(std::ofstream& tum,
                  double t,
                  const Eigen::Vector3d& p,
                  const Eigen::Quaterniond& q) {
    tum << std::fixed << std::setprecision(9)
        << t << " " << p.x() << " " << p.y() << " " << p.z() << " "
        << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string config_path = (argc > 1) ? argv[1] : DEFAULT_CONFIG;
        const std::string sensor_csv = (argc > 2) ? argv[2] : DEFAULT_SENSOR_CSV;
        const std::string output_csv = (argc > 3) ? argv[3] : DEFAULT_OUTPUT_CSV;
        const std::string output_tum = (argc > 4) ? argv[4] : DEFAULT_OUTPUT_TUM;
        const std::string joint_suffix = (argc > 5) ? argv[5] : "";

        std::cout << "SEROW config:        " << config_path << "\n";
        std::cout << "Sensor input CSV:    " << sensor_csv << "\n";
        std::cout << "Prediction CSV:      " << output_csv << "\n";
        std::cout << "Prediction TUM:      " << output_tum << "\n";
        std::cout << "Joint-name suffix:   '" << joint_suffix << "'\n";

        serow::Serow estimator;
        if (!estimator.initialize(config_path)) {
            throw std::runtime_error("Failed to initialize Serow. Check config/model paths.");
        }

        const auto leg_to_frame = loadLegToFootFrame(config_path);

        std::ifstream in(sensor_csv);
        if (!in.is_open()) throw std::runtime_error("Could not open sensor CSV: " + sensor_csv);

        const auto output_csv_parent = std::filesystem::path(output_csv).parent_path();
        const auto output_tum_parent = std::filesystem::path(output_tum).parent_path();
        if (!output_csv_parent.empty()) std::filesystem::create_directories(output_csv_parent);
        if (!output_tum_parent.empty()) std::filesystem::create_directories(output_tum_parent);

        std::ofstream out(output_csv);
        if (!out.is_open()) throw std::runtime_error("Could not create output CSV: " + output_csv);
        std::ofstream tum(output_tum);
        if (!tum.is_open()) throw std::runtime_error("Could not create TUM output: " + output_tum);

        out << std::fixed << std::setprecision(12);
        writeCsvHeader(out);

        std::string header_line;
        if (!std::getline(in, header_line)) throw std::runtime_error("Empty sensor CSV.");
        const char delimiter = (header_line.find('\t') != std::string::npos) ? '\t' : ',';
        const auto header = split(header_line, delimiter);
        const auto idx = makeHeaderIndex(header);

        // Validate required columns before running.
        requireColumn(idx, "t");
        requireColumn(idx, "imu_wx"); requireColumn(idx, "imu_wy"); requireColumn(idx, "imu_wz");
        requireColumn(idx, "imu_ax"); requireColumn(idx, "imu_ay"); requireColumn(idx, "imu_az");
        for (const auto& j : JOINTS) {
            requireColumn(idx, "joint_pos_" + j);
            requireColumn(idx, "joint_vel_" + j);
            requireColumn(idx, "joint_eff_" + j);
        }

        if (USE_CONTACT_FLAGS) {
            for (const auto& leg : LEGS) {
                requireColumn(idx, "contact_" + leg);
            }
        }
        size_t input_rows = 0;
        size_t filter_ok = 0;
        size_t written_rows = 0;
        std::string line;
        auto start = std::chrono::high_resolution_clock::now();

        while (std::getline(in, line)) {
            if (trim(line).empty()) continue;
            ++input_rows;
            const auto row = split(line, delimiter);
            const double t = valueAt(row, idx, "t");

            serow::ImuMeasurement imu = makeImu(row, idx);
            auto joints = makeJoints(row, idx, joint_suffix);

            // SEROW currently initializes/updates contact state only when a force-torque map exists.
            // Since this ANYmal CSV has no force measurements, we provide zero dummy forces and pass
            // contact_LF/RF/LH/RH as the contact probabilities used by the estimator.
            // auto dummy_ft = makeDummyFootForces(leg_to_frame, t);
            // auto binary_contacts = makeBinaryContacts(row, idx, leg_to_frame);

            std::optional<std::map<std::string, serow::ForceTorqueMeasurement>> force_torque = std::nullopt;
            std::optional<std::map<std::string, serow::ContactMeasurement>> contacts_probability = std::nullopt;

            if (USE_CONTACT_FLAGS) {
                force_torque = makeDummyFootForces(leg_to_frame, t);
                contacts_probability = makeBinaryContacts(row, idx, leg_to_frame);
            }

            const bool ok = estimator.filter(
                imu,
                joints,
                force_torque,          // dummy FT only when USE_CONTACT_FLAGS=true
                std::nullopt,          // no odometry
                contacts_probability,  // binary contact flags only when USE_CONTACT_FLAGS=true
                std::nullopt);         // no base-pose ground truth
            if (!ok) continue;
            ++filter_ok;

            auto state = estimator.getState(true);
            if (!state.has_value()) continue;

            const Eigen::Vector3d p = state->getBasePosition();
            const Eigen::Quaterniond q = state->getBaseOrientation();
            const Eigen::Vector3d v = state->getBaseLinearVelocity();
            const Eigen::Vector3d com_p = state->getCoMPosition();
            const Eigen::Vector3d com_v = state->getCoMLinearVelocity();
            const Eigen::Vector3d b_acc = state->getImuLinearAccelerationBias();
            const Eigen::Vector3d b_gyr = state->getImuAngularVelocityBias();

            auto contact_state = estimator.getContactState(true);

            double c_lf = 0.0;
            double c_rf = 0.0;
            double c_lh = 0.0;
            double c_rh = 0.0;

            if (contact_state.has_value()) {
                const auto& cp = contact_state->contacts_probability;

                c_lf = cp.count(leg_to_frame.at("LF")) ? cp.at(leg_to_frame.at("LF")) : 0.0;
                c_rf = cp.count(leg_to_frame.at("RF")) ? cp.at(leg_to_frame.at("RF")) : 0.0;
                c_lh = cp.count(leg_to_frame.at("LH")) ? cp.at(leg_to_frame.at("LH")) : 0.0;
                c_rh = cp.count(leg_to_frame.at("RH")) ? cp.at(leg_to_frame.at("RH")) : 0.0;
            }

            out << t << ','
                << p.x() << ',' << p.y() << ',' << p.z() << ','
                << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << ','
                << v.x() << ',' << v.y() << ',' << v.z() << ','
                << com_p.x() << ',' << com_p.y() << ',' << com_p.z() << ','
                << com_v.x() << ',' << com_v.y() << ',' << com_v.z() << ','
                << b_acc.x() << ',' << b_acc.y() << ',' << b_acc.z() << ','
                << b_gyr.x() << ',' << b_gyr.y() << ',' << b_gyr.z() << ','
                << c_lf << ',' << c_rf << ',' << c_lh << ',' << c_rh << '\n';

            writeTumLine(tum, t, p, q);
            ++written_rows;
        }

        auto end = std::chrono::high_resolution_clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Done. Input rows: " << input_rows
                  << ", filter accepted: " << filter_ok
                  << ", written: " << written_rows
                  << ", elapsed: " << us << " us\n";

        if (written_rows == 0) {
            std::cerr << "Warning: no states were written. Common causes: IMU calibration still running, "
                      << "timestamp mismatch, wrong joint names, or wrong foot frame names.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
