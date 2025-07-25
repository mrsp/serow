#!/usr/bin/env python3

import numpy as np
import json

from env import SerowEnv
from ac import ONNXInference
from train import PreStepPPO

if __name__ == "__main__":
    # Initialize ONNX inference
    robot = "go2"
    device = "cpu"

    # Read the data
    test_dataset = np.load(f"{robot}_log.npz", allow_pickle=True)
    stats = json.load(open(f"models/{robot}_stats.json"))
    print(stats)

    # Load the ONNX model
    agent_onnx = ONNXInference(robot, path="models")

    # Load the saved PPO model
    agent_ppo = PreStepPPO.load(f"models/{robot}_ppo")
    agent_ppo.eval()

    # Get contacts frame from the first measurement
    contact_states = test_dataset["contact_states"]
    contacts_frame = list(contact_states[0].contacts_status.keys())
    state_dim = 2 + 3 + 9 + 3 + 4
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()
    min_action = np.array([1e-4], dtype=np.float32)
    max_action = np.array([5e1], dtype=np.float32)

    test_env = SerowEnv(
        robot,
        contacts_frame[0],
        test_dataset["joint_states"][0],
        test_dataset["base_states"][0],
        test_dataset["contact_states"][0],
        action_dim,
        state_dim,
        min_action,
        max_action,
        test_dataset["imu"],
        test_dataset["joints"],
        test_dataset["ft"],
        test_dataset["base_pose_ground_truth"],
    )

    # Use the loaded PPO model for evaluation
    (
        ppo_timestamps,
        ppo_base_positions,
        ppo_base_orientations,
        ppo_gt_positions,
        ppo_gt_orientations,
    ) = test_env.evaluate(agent_ppo, stats)

    # Use the ONNX model for evaluation
    (
        onnx_timestamps,
        onnx_base_positions,
        onnx_base_orientations,
        onnx_gt_positions,
        onnx_gt_orientations,
    ) = test_env.evaluate(agent_onnx, stats)

    # These must be equal
    assert np.allclose(ppo_timestamps, onnx_timestamps)
    assert np.allclose(ppo_base_positions, onnx_base_positions)
    assert np.allclose(ppo_base_orientations, onnx_base_orientations)
    assert np.allclose(ppo_gt_positions, onnx_gt_positions)
    assert np.allclose(ppo_gt_orientations, onnx_gt_orientations)
    print("All tests passed")
