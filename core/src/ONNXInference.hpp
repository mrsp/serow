#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif
#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>
#include <vector>
#include "common.hpp"

namespace serow {

class ONNXInference {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Constructor that initializes ONNX Runtime environment and memory info
     */
    ONNXInference() : env_(ORT_LOGGING_LEVEL_WARNING, "serow-onnx"),
                      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

    /**
     * @brief Initializes the ONNX inference with actor and critic models
     * @param robot_name Name of the robot (e.g., "go2")
     */
    void init(const std::string& robot_name);

    /**
     * @brief Gets the action from the actor model
     * @param state Current state vector
     * @return Action vector
     */
    Eigen::VectorXd getAction(const Eigen::VectorXd& state);

    /**
     * @brief Gets the value from the critic model
     * @param state Current state vector
     * @param action Current action vector
     * @return Value estimate
     */
    double getValue(const Eigen::VectorXd& state, const Eigen::VectorXd& action);

    /**
     * @brief Gets the state dimension from the actor model
     * @return State dimension
     */
    int getStateDim() const { return state_dim_; }

    /**
     * @brief Gets the action dimension from the actor model
     * @return Action dimension
     */
    int getActionDim() const { return action_dim_; }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> actor_session_;
    std::unique_ptr<Ort::Session> critic_session_;
    
    // Input/output names - store as strings to ensure lifetime
    std::string actor_input_name_;
    std::string actor_output_name_;
    std::vector<std::string> critic_input_names_;
    
    // Input/output dimensions
    int state_dim_;
    int action_dim_;
    
    // Memory allocator
    Ort::MemoryInfo memory_info_;

    /**
     * @brief Converts an Eigen vector to an ONNX Runtime tensor
     * @param eigen_vec Input Eigen vector
     * @param shape Desired tensor shape
     * @return ONNX Runtime tensor
     */
    Ort::Value eigenToOrtTensor(const Eigen::VectorXd& eigen_vec, const std::vector<int64_t>& shape);
};

} // namespace serow 
