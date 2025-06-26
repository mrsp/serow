#!/usr/bin/env python3

import numpy as np
import onnxruntime as ort

from read_mcap import(
    read_imu_measurements, 
    read_joint_measurements, 
    read_force_torque_measurements, 
    read_base_pose_ground_truth,    
    read_contact_states,
    read_base_states,
    read_joint_states
)

from env import SerowEnv

class ONNXInference:
    def __init__(self, robot, path, device='cpu', name="PPO"):
        self.checkpoint_dir = path
        self.device = device
        self.robot = robot
        self.name = name
        
        # Initialize ONNX Runtime sessions
        self.actor_session = ort.InferenceSession(
            f'{path}/trained_policy_{robot}_actor.onnx',
            providers=['CPUExecutionProvider']
        )
        self.critic_session = ort.InferenceSession(
            f'{path}/trained_policy_{robot}_critic.onnx',
            providers=['CPUExecutionProvider']
        )
        
        # Get input names
        self.actor_input_name = self.actor_session.get_inputs()[0].name
        self.critic_input_names = [input.name for input in self.critic_session.get_inputs()]
        
        # Get input shapes
        self.state_dim = self.actor_session.get_inputs()[0].shape[1]
        self.action_dim = self.actor_session.get_outputs()[0].shape[1]
        self.min_action = 1e-10  # This should match your PPO params
        
        print(f"Initialized ONNX inference for {robot}")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")

    def get_action(self, state, deterministic=True):
        # Prepare input
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        
        # Run actor inference
        actor_output = self.actor_session.run(
            None, 
            {self.actor_input_name: state}
        )[0]
        
        # Get action from actor output and apply softplus scaling
        action = actor_output[0]
        action_scaled = np.log1p(np.exp(action)) + self.min_action  # softplus scaling
        return np.array([action_scaled])

    def get_value(self, state, action):
        # Prepare inputs
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        action = np.array(action, dtype=np.float32).reshape(1, -1)
        
        # Run critic inference
        critic_output = self.critic_session.run(
            None,
            {self.critic_input_names[0]: state, self.critic_input_names[1]: action}
        )[0]
        
        return critic_output[0][0]

if __name__ == "__main__":
    # Read the data
    imu_measurements  = read_imu_measurements("/tmp/serow_measurements.mcap")
    joint_measurements = read_joint_measurements("/tmp/serow_measurements.mcap")
    force_torque_measurements = read_force_torque_measurements("/tmp/serow_measurements.mcap")
    base_pose_ground_truth = read_base_pose_ground_truth("/tmp/serow_measurements.mcap")
    base_states = read_base_states("/tmp/serow_proprioception.mcap")
    contact_states = read_contact_states("/tmp/serow_proprioception.mcap")
    joint_states = read_joint_states("/tmp/serow_proprioception.mcap")

    # Create test dataset
    test_dataset = {
        'imu': imu_measurements,
        'joints': joint_measurements,
        'ft': force_torque_measurements,
        'base_states': base_states,
        'contact_states': contact_states,
        'joint_states': joint_states,
        'base_pose_ground_truth': base_pose_ground_truth
    }

    # Initialize ONNX inference
    robot = "go2"
    device = "cpu"
    policy_path = "policy/ppo"
    agent = ONNXInference(robot, path=policy_path, device=device, name="PPO")

    # Get contacts frame from the first measurement
    contacts_frame = set(contact_states[0].contacts_status.keys())
    print(f"Contacts frame: {contacts_frame}")

    # Define the dimensions of your state and action spaces
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()
    state_dim = 3 
 
    # Evaluate the policy
    serow_env = SerowEnv(robot, joint_states[0], base_states[0], contact_states[0], action_dim, state_dim)
    serow_env.evaluate(test_dataset, agent)
