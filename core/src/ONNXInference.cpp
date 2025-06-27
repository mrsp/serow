#include "ONNXInference.hpp"
#include <iostream>
#include <filesystem>

namespace serow {

void ONNXInference::init(const std::string& robot_name) {
    std::cout << "Initializing ONNX inference for " << robot_name << std::endl;
    
    // Check if model files exist
    const std::string actor_filename = "trained_policy_" + robot_name + "_actor.onnx";
    const std::string critic_filename = "trained_policy_" + robot_name + "_critic.onnx";
        
    const std::string actor_path = findFilepath(actor_filename);
    const std::string critic_path = findFilepath(critic_filename);
        
    if (!std::filesystem::exists(actor_path)) {
        throw std::runtime_error("Actor model file not found: " + actor_path);
    }
    if (!std::filesystem::exists(critic_path)) {
        throw std::runtime_error("Critic model file not found: " + critic_path);
    }

    std::cout << "Actor path: " << actor_path << std::endl;
    std::cout << "Critic path: " << critic_path << std::endl;

    // Initialize ONNX Runtime environment
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "serow-onnx");
    
    // Create memory info
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Load actor model
    actor_session_ = std::make_unique<Ort::Session>(env_, actor_path.c_str(), Ort::SessionOptions{});
    
    // Load critic model
    critic_session_ = std::make_unique<Ort::Session>(env_, critic_path.c_str(), Ort::SessionOptions{});
    
    // Get input names
    auto actor_input_name = actor_session_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    actor_input_name_ = actor_input_name.get();
    
    // Get output names
    auto actor_output_name = actor_session_->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    actor_output_name_ = actor_output_name.get();

    auto critic_input_name0 = critic_session_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto critic_input_name1 = critic_session_->GetInputNameAllocated(1, Ort::AllocatorWithDefaultOptions());
    critic_input_names_.push_back(critic_input_name0.get());
    critic_input_names_.push_back(critic_input_name1.get());
    
    // Get input/output dimensions
    state_dim_ = actor_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];
    action_dim_ = actor_session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];
    
    std::cout << "Initialized ONNX inference for " << robot_name << std::endl;
    std::cout << "State dimension: " << state_dim_ << std::endl;
    std::cout << "Action dimension: " << action_dim_ << std::endl;
}

Eigen::VectorXd ONNXInference::getAction(const Eigen::VectorXd& state) {
    // Prepare input tensor
    std::vector<int64_t> input_shape = {1, state_dim_};
    auto input_tensor = eigenToOrtTensor(state, input_shape);
    
    // Store input and output names in vectors to ensure they stay alive during Run()
    std::vector<const char*> input_names = {actor_input_name_.c_str()};
    std::vector<const char*> output_names = {actor_output_name_.c_str()};
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    
    // Run inference
    auto output_tensors = actor_session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_names.size(),
        output_names.data(),
        output_names.size()
    );
    
    // Get output data
    auto output_data = output_tensors[0].GetTensorMutableData<float>();
    Eigen::VectorXd action(action_dim_);
    for (int i = 0; i < action_dim_; ++i) {
        action(i) = std::max(static_cast<float>(output_data[i]), 1e-6f); // Ensure positive values
    }
    return action;
}

double ONNXInference::getValue(const Eigen::VectorXd& state, const Eigen::VectorXd& action) {
    // Prepare input tensors
    std::vector<int64_t> state_shape = {1, state_dim_};
    std::vector<int64_t> action_shape = {1, action_dim_};
    auto state_tensor = eigenToOrtTensor(state, state_shape);
    auto action_tensor = eigenToOrtTensor(action, action_shape);
    
    // Prepare input names and tensors
    std::vector<const char*> input_names = {critic_input_names_[0].c_str(), critic_input_names_[1].c_str()};
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(state_tensor));
    input_tensors.push_back(std::move(action_tensor));
    
    // Run inference
    auto output_tensors = critic_session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        2,
        nullptr,
        1
    );
    
    // Get output value
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return output_data[0];
}

Ort::Value ONNXInference::eigenToOrtTensor(const Eigen::VectorXd& eigen_vec, const std::vector<int64_t>& shape) {
    // Create tensor
    std::vector<float> float_data(eigen_vec.size());
    for (Eigen::Index i = 0; i < eigen_vec.size(); ++i) {
        float_data[i] = static_cast<float>(eigen_vec[i]);
    }
    
    auto tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        float_data.data(),
        float_data.size(),
        shape.data(),
        shape.size()
    );
    return tensor;
}

} // namespace serow 
