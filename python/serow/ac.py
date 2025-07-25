from typing import Callable, Tuple
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import gymnasium as gym
import onnxruntime as ort
import numpy as np


class CustomActorCritic(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable[[float], float],
        action_min: float = None,
        action_max: float = None,
        *args,
        **kwargs,
    ):
        # Store action limits
        if isinstance(action_space, gym.spaces.Box):
            self.action_min = (
                torch.tensor(action_space.low, dtype=torch.float32)
                if action_min is None
                else torch.tensor(action_min, dtype=torch.float32)
            )
            self.action_max = (
                torch.tensor(action_space.high, dtype=torch.float32)
                if action_max is None
                else torch.tensor(action_max, dtype=torch.float32)
            )
        else:
            self.action_min = action_min
            self.action_max = action_max

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Custom shared layers
        self.shared_net = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Actor network (policy)
        self.policy_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Critic network (value function)
        self.value_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Initialize weights and biases
        for layer in self.shared_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
        for layer in self.policy_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
        for layer in self.value_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)

        # Create a custom MLP extractor that matches the expected interface
        class CustomMLPExtractor(nn.Module):
            def __init__(self, shared_net, policy_net, value_net):
                super().__init__()
                self.shared_net = shared_net
                self.policy_net = policy_net
                self.value_net = value_net
                self.latent_dim_pi = 1
                self.latent_dim_vf = 1

            def forward(self, features):
                shared_features = self.shared_net(features)
                latent_pi = self.policy_net(shared_features)
                latent_vf = self.value_net(shared_features)
                return latent_pi, latent_vf

            def forward_actor(self, features):
                shared_features = self.shared_net(features)
                return self.policy_net(shared_features)

            def forward_critic(self, features):
                shared_features = self.shared_net(features)
                return self.value_net(shared_features)

        self.mlp_extractor = CustomMLPExtractor(
            self.shared_net, self.policy_net, self.value_net
        )

    def _scale_actions(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """
        Scale actions from [0, 1] (sigmoid output) to [action_min, action_max]
        """
        if self.action_min is not None and self.action_max is not None:
            # Move tensors to same device as raw_actions
            action_min = self.action_min.to(raw_actions.device)
            action_max = self.action_max.to(raw_actions.device)

            # Scale from [0, 1] to [action_min, action_max]
            scaled_actions = action_min + raw_actions * (action_max - action_min)
            return scaled_actions
        return raw_actions

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Clip actions to be within bounds
        """
        if self.action_min is not None and self.action_max is not None:
            action_min = self.action_min.to(actions.device)
            action_max = self.action_max.to(actions.device)
            return torch.clamp(actions, action_min, action_max)
        return actions

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        """
        # Get latent codes for actor and critic
        latent_pi, latent_vf = self.mlp_extractor.forward(obs)

        # Get action distribution and value
        distribution = self._get_action_dist_from_latent(latent_pi)
        raw_actions = distribution.get_actions(deterministic=deterministic)

        actions = self._scale_actions(raw_actions)
        actions = self._clip_actions(actions)

        log_prob = distribution.log_prob(actions)
        # The framework expects latent_vf to be the final value
        values = latent_vf

        return actions, values, log_prob

    def predict(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the policy action from an observation.
        """
        self.set_training_mode(False)

        obs_tensor, _ = self.obs_to_tensor(obs)

        actions, values, _ = self.forward(obs_tensor, deterministic)
        return actions.detach().cpu().numpy()[0], values.detach().cpu().numpy()[0]

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the
        observations.
        """
        return self.mlp_extractor.forward_critic(obs)


class ONNXInference:
    def __init__(self, robot, path):
        # Initialize ONNX Runtime sessions
        self.session = ort.InferenceSession(
            f"{path}/{robot}_ppo.onnx",
            providers=["CPUExecutionProvider"],
        )

        # Get input names
        self.input_name = self.session.get_inputs()[0].name
        print(f"Input names: {self.input_name}")

        # Get input shapes
        self.state_dim = self.session.get_inputs()[0].shape[1]
        self.action_dim = self.session.get_outputs()[0].shape[1]

        print(f"Initialized ONNX inference for {robot}")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")

    def forward(self, observation, deterministic=True):
        # Prepare input
        observation = np.array(observation, dtype=np.float32).reshape(1, -1)
        output = self.session.run(None, {self.input_name: observation})
        return output[0], output[1], output[2]

    def predict(self, observation, deterministic=True):
        """
        Predict action given observation.
        Matches the interface expected by SerowEnv.evaluate().
        Returns (action, value) tuple.
        """
        action, value, _ = self.forward(observation, deterministic=deterministic)
        return action, value
