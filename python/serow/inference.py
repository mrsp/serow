#!/usr/bin/env python3

import numpy as np
import onnxruntime as ort
import json
from env import SerowEnv


class ONNXInference:
    def __init__(
        self, robot, path, device="cpu", name="PPO", action_min=None, action_max=None
    ):
        self.checkpoint_dir = path
        self.device = device
        self.robot = robot
        self.name = name

        # Store action limits (matching ac.py)
        if action_min is not None and action_max is not None:
            self.action_min = np.array(action_min, dtype=np.float32)
            self.action_max = np.array(action_max, dtype=np.float32)
        else:
            # Default action bounds if not provided
            self.action_min = np.array([1e-4], dtype=np.float32)
            self.action_max = np.array([5e1], dtype=np.float32)

        # Initialize ONNX Runtime sessions
        self.actor_session = ort.InferenceSession(
            f"{path}/{robot}_ppo_actor.onnx",
            providers=["CPUExecutionProvider"],
        )
        self.critic_session = ort.InferenceSession(
            f"{path}/{robot}_ppo_critic.onnx",
            providers=["CPUExecutionProvider"],
        )

        # Get input names
        self.actor_input_name = self.actor_session.get_inputs()[0].name
        self.critic_input_name = self.critic_session.get_inputs()[0].name
        print(f"Actor input names: {self.actor_input_name}")
        print(f"Critic input names: {self.critic_input_name}")

        # Get input shapes
        self.state_dim = self.actor_session.get_inputs()[0].shape[1]
        self.action_dim = self.actor_session.get_outputs()[0].shape[1]

        print(f"Initialized ONNX inference for {robot}")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")
        print(f"Action bounds: [{self.action_min}, {self.action_max}]")

    def _scale_actions(self, raw_actions):
        """
        Scale actions from [0, 1] (sigmoid output) to [action_min, action_max]
        Matching the _scale_actions method from ac.py
        """
        # Scale from [0, 1] to [action_min, action_max]
        scaled_actions = self.action_min + raw_actions * (
            self.action_max - self.action_min
        )
        return scaled_actions

    def _clip_actions(self, actions):
        """
        Clip actions to be within bounds
        Matching the _clip_actions method from ac.py
        """
        return np.clip(actions, self.action_min, self.action_max)

    def get_action(self, state, deterministic=True):
        # Prepare input
        state = np.array(state, dtype=np.float32).reshape(1, -1)

        # Run actor inference
        actor_output = self.actor_session.run(None, {self.actor_input_name: state})[0]

        # Get raw actions from actor output (sigmoid output in [0, 1])
        raw_actions = actor_output[0]

        # Scale actions from [0, 1] to [action_min, action_max]
        actions = self._scale_actions(raw_actions)

        # Clip actions to ensure they're within bounds
        actions = self._clip_actions(actions)

        return actions

    def predict(self, observation, deterministic=True):
        """
        Predict action given observation.
        Matches the interface expected by SerowEnv.evaluate().
        Returns (action, state) tuple.
        """
        action = self.get_action(observation, deterministic=deterministic)
        value = self.get_value(observation)
        return action, value

    def get_value(self, state):
        # Prepare inputs
        state = np.array(state, dtype=np.float32).reshape(1, -1)

        # Run critic inference
        critic_output = self.critic_session.run(
            None,
            {self.critic_input_name: state},
        )[0]
        return critic_output[0]


if __name__ == "__main__":
    # Initialize ONNX inference
    robot = "go2"
    device = "cpu"

    # Read the data
    test_dataset = np.load(f"{robot}_log.npz", allow_pickle=True)
    stats = json.load(open(f"models/{robot}_stats.json"))
    print(stats)

    contact_states = test_dataset["contact_states"]
    contact_frame = list(contact_states[0].contacts_status.keys())
    print(f"Contact frames: {contact_frame}")

    # Define action bounds (matching the environment setup)
    min_action = np.array([1e-4], dtype=np.float32)
    max_action = np.array([5e1], dtype=np.float32)

    agent = ONNXInference(
        robot,
        path="models",
        device=device,
        name="PPO",
        action_min=min_action,
        action_max=max_action,
    )

    # Get contacts frame from the first measurement
    contacts_frame = set(contact_states[0].contacts_status.keys())
    print(f"Contacts frame: {contacts_frame}")
    state_dim = 2 + 3 + 9 + 3 + 4
    print(f"State dimension: {state_dim}")
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()

    test_env = SerowEnv(
        robot,
        contact_frame[0],
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

    test_env.evaluate(agent, stats)
