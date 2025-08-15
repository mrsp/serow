#!/usr/bin/env python3

import numpy as np
import json
import onnx
import onnx.numpy_helper
import os
import onnxruntime as ort

from env import SerowEnv
from train import PreStepDQN


class ONNXInference:
    def __init__(self, robot, path):
        # Initialize ONNX Runtime sessions
        model_path = f"{path}/{robot}_dqn.onnx"

        print(f"Loading ONNX model from: {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        print("Session created successfully")
        print(f"Session providers: {self.session.get_providers()}")
        print(f"Available providers: {ort.get_available_providers()}")

        # Get input names
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        print(f"Number of inputs: {len(inputs)}")
        print(f"Number of outputs: {len(outputs)}")

        if len(inputs) > 0:
            self.input_name = inputs[0].name
            print(f"Input names: {self.input_name}")
            self.state_dim = inputs[0].shape[1]
        else:
            raise ValueError("No inputs found in ONNX model")

        if len(outputs) > 0:
            print(f"Output 0 shape: {outputs[0].shape}")
            print(f"Output 0 name: {outputs[0].name}")

            # Handle dynamic shapes - if shape is a list with only one element,
            # it means the output dimension is dynamic and we need to infer it
            if len(outputs[0].shape) == 1 and outputs[0].shape[0] == "batch_size":
                # This is a dynamic shape, we'll need to infer the actual
                # dimension. For now, let's use a default value or try to get
                # it from the model
                print(
                    "Warning: Dynamic output shape detected, "
                    "using default action dimension"
                )
                self.action_dim = 1  # Default action dimension
            else:
                self.action_dim = outputs[0].shape[1]
        else:
            raise ValueError("No outputs found in ONNX model")

        # Action space - discrete choices for measurement noise scaling
        self.discrete_actions = np.array(
            [
                1e-5,
                5e-5,
                1e-4,
                5e-4,
                1e-3,
                5e-3,
                1e-2,
                5e-2,
                1e-1,
                5e-1,
                1.0,
                5.0,
                10.0,
                50.0,
                100.0,
                500.0,
                1000.0,
            ],
            dtype=np.float32,
        )

        print(f"Initialized ONNX inference for {robot} with dqn model")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")

    def forward(self, observation, deterministic=True):
        # Prepare input
        observation = np.array(observation, dtype=np.float32).reshape(1, -1)
        output = self.session.run(None, {self.input_name: observation})

        # DQN outputs Q-values, we need to select the action with highest Q-value
        q_values = output[0]
        action = np.argmax(q_values, axis=1)
        return action, q_values

    def predict(self, observation, deterministic=True):
        """
        Predict action given observation.
        Matches the interface expected by SerowEnv.evaluate().
        Returns action and value
        """
        return self.forward(observation, deterministic=deterministic)


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


def compare_onnx_dqn_predictions(agent_onnx, agent_dqn, state_dim):
    """
    Compares the predictions of an ONNX model with a DQN model.

    Args:
        agent_onnx (ONNXInference): ONNX model to compare with.
        agent_dqn (PreStepDQN): DQN model to compare with.
        state_dim (int): Dimension of the state space.
    """

    # Generate a few random observations
    for i in range(100):
        # Generate a random observation
        obs = np.random.randn(1, state_dim).astype(np.float32)

        # Get the DQN model prediction
        dqn_action, _ = agent_dqn.predict(obs, deterministic=True)
        dqn_action = np.array(
            [agent_onnx.discrete_actions[dqn_action.item()]], dtype=np.float32
        )

        # Get the ONNX model prediction
        onnx_action, _ = agent_onnx.predict(obs, deterministic=True)
        onnx_action = np.array(
            [agent_onnx.discrete_actions[onnx_action.item()]], dtype=np.float32
        )

        # Compare the actions
        assert np.allclose(dqn_action, onnx_action, atol=1e-4)
    print("ONNX and DQN action predictions match")


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
    history_size = 100
    state_dim = 3 + 9 + 3 + 4 + 3 * history_size + 1 * history_size
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()

    # Load the saved DQN model
    try:
        agent_dqn = PreStepDQN.load(f"models/{robot}_dqn")
        print("Loaded DQN model successfully")
    except Exception as e:
        print(f"Could not load DQN model: {e}")
        agent_dqn = None

    # Load the ONNX model
    agent_onnx = ONNXInference(robot, path="models")

    # Compare the ONNX model predictions with the DQN model predictions
    if agent_dqn is not None:
        compare_onnx_dqn_predictions(agent_onnx, agent_dqn, state_dim)

    test_env = SerowEnv(
        contacts_frame[0],
        robot,
        test_dataset["joint_states"][0],
        test_dataset["base_states"][0],
        test_dataset["contact_states"][0],
        action_dim,
        state_dim,
        test_dataset["imu"],
        test_dataset["joints"],
        test_dataset["ft"],
        test_dataset["base_pose_ground_truth"],
        history_size,
    )

    # Use the loaded DQN model for evaluation
    if agent_dqn is not None:
        (
            dqn_timestamps,
            dqn_base_positions,
            dqn_base_orientations,
            dqn_gt_positions,
            dqn_gt_orientations,
            dqn_rewards,
            _,
            _,
        ) = test_env.evaluate(agent_dqn, stats, plot=True)

    # Use the ONNX model for evaluation
    (
        onnx_timestamps,
        onnx_base_positions,
        onnx_base_orientations,
        onnx_gt_positions,
        onnx_gt_orientations,
        onnx_rewards,
        _,
        _,
    ) = test_env.evaluate(agent_onnx, stats, plot=True)

    # These must be equal if DQN model was loaded
    if agent_dqn is not None:
        assert np.allclose(dqn_timestamps, onnx_timestamps, atol=1e-3)
        assert np.allclose(dqn_base_positions, onnx_base_positions, atol=1e-3)
        assert np.allclose(dqn_base_orientations, onnx_base_orientations, atol=1e-3)
        print("All tests passed")
    else:
        print("DQN model not loaded, skipping comparison tests")
