#!/usr/bin/env python3

import numpy as np
import json
import onnx
import onnx.numpy_helper
import os

from env import SerowEnv
from ac import ONNXInference
from train import PreStepPPO


def get_onnx_weights_biases(onnx_model_path):
    """
    Loads an ONNX model from its file path and extracts its initializers
    (weights and biases).

    Args:
        onnx_model_path (str): Path to the ONNX model file.

    Returns:
        dict: A dictionary where keys are the names of the initializers
              (weights/biases) and values are their NumPy array
              representations.
    """
    # Load the ONNX model from the file path
    onnx_model = onnx.load(onnx_model_path)
    weights_biases = {}
    for initializer in onnx_model.graph.initializer:
        # Initializers are TensorProto objects, use numpy_helper to convert
        # to numpy array
        np_array = onnx.numpy_helper.to_array(initializer)
        weights_biases[initializer.name] = np_array
    return weights_biases


def compare_onnx_ppo(onnx_model_path, ppo_model, robot):
    """
    Compares the weights and biases of an ONNX model with a PPO model.

    Args:
        onnx_model_path (str): Path to the ONNX model file.
        ppo_model (PreStepPPO): PPO model to compare with.
    """
    onnx_filepath = os.path.join(onnx_model_path, f"{robot}_ppo.onnx")
    params = get_onnx_weights_biases(onnx_filepath)

    # Compare the weights and biases to the agent_ppo model
    print("Comparing parameters:")
    ppo_policy_params = dict(ppo_model.policy.named_parameters())

    # Create a mapping from ONNX parameter names to PPO parameter names
    # Remove the 'policy.' prefix from ONNX parameter names
    def get_ppo_param_name(onnx_name):
        if onnx_name.startswith("policy."):
            return onnx_name[7:]  # Remove 'policy.' prefix
        return onnx_name

    for name, onnx_param in params.items():
        ppo_name = get_ppo_param_name(name)
        if ppo_name in ppo_policy_params:
            ppo_param = ppo_policy_params[ppo_name].detach().cpu().numpy()
            is_close = np.allclose(onnx_param, ppo_param, rtol=1e-5, atol=1e-8)
            status = "✓" if is_close else "✗"
            print(f"  {name} -> {ppo_name}: {status} (shape: {onnx_param.shape})")
            if not is_close:
                max_diff = np.max(np.abs(onnx_param - ppo_param))
                mean_diff = np.mean(np.abs(onnx_param - ppo_param))
                print(f"    Max difference: {max_diff}")
                print(f"    Mean difference: {mean_diff}")
        else:
            print(f"  {name} -> {ppo_name}: ✗ (not found in PPO policy)")

    # Check if all parameters match
    match = all(
        get_ppo_param_name(name) in ppo_policy_params
        and np.allclose(
            onnx_param,
            ppo_policy_params[get_ppo_param_name(name)].detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-8,
        )
        for name, onnx_param in params.items()
    )

    if match:
        print("\n✓ All weights and biases are the same")
    else:
        print("\n✗ Some weights and biases differ")


def compare_onnx_ppo_predictions(agent_onnx, agent_ppo, state_dim):
    """
    Compares the predictions of an ONNX model with a PPO model.

    Args:
        agent_onnx (ONNXInference): ONNX model to compare with.
        agent_ppo (PreStepPPO): PPO model to compare with.
        state_dim (int): Dimension of the state space.
    """

    # Generate a few random observations
    for i in range(100):
        # Generate a random observation
        obs = np.random.randn(1, state_dim).astype(np.float32)

        # Get the PPO model prediction
        ppo_action, ppo_value = agent_ppo.predict(obs, deterministic=True)

        # Get the ONNX model prediction
        onnx_action, onnx_value = agent_onnx.predict(obs, deterministic=True)

        # Compare the predictions
        assert np.allclose(ppo_action, onnx_action, atol=1e-4)
        assert np.allclose(ppo_value, onnx_value, atol=1e-4)
    print("ONNX and PPO predictions match")


if __name__ == "__main__":
    # Initialize ONNX inference
    robot = "go2"
    device = "cpu"
    model_dir = "models"

    # Read the data
    test_dataset = np.load(f"{robot}_log.npz", allow_pickle=True)
    try:
        stats = json.load(open(f"{model_dir}/{robot}_stats.json"))
    except FileNotFoundError:
        stats = None

    # Get contacts frame from the first measurement
    contact_states = test_dataset["contact_states"]
    contacts_frame = list(contact_states[0].contacts_status.keys())
    state_dim = 3 + 9 + 3 + 4
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()
    min_action = np.array([1e-8], dtype=np.float32)
    max_action = np.array([1e2], dtype=np.float32)

    # Load the saved PPO model
    agent_ppo = PreStepPPO.load(f"models/{robot}_ppo")
    agent_ppo.eval()

    # Compare the ONNX model with the PPO model
    compare_onnx_ppo(model_dir, agent_ppo, robot)

    # Load the ONNX model
    agent_onnx = ONNXInference(robot, path="models")

    # Compare the ONNX model predictions with the PPO model predictions
    compare_onnx_ppo_predictions(agent_onnx, agent_ppo, state_dim)

    test_env = SerowEnv(
        robot,
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
        ppo_rewards,
    ) = test_env.evaluate(agent_ppo, stats, plot=True)

    # Use the ONNX model for evaluation
    (
        onnx_timestamps,
        onnx_base_positions,
        onnx_base_orientations,
        onnx_gt_positions,
        onnx_gt_orientations,
        onnx_rewards,
    ) = test_env.evaluate(agent_onnx, stats, plot=True)

    # These must be equal
    assert np.allclose(ppo_timestamps, onnx_timestamps, atol=1e-3)
    assert np.allclose(ppo_base_positions, onnx_base_positions, atol=1e-3)
    assert np.allclose(ppo_base_orientations, onnx_base_orientations, atol=1e-3)
    print("All tests passed")
