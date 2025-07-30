from typing import Callable, Tuple, Optional, Union, List, Dict, Type
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.policies import BaseFeaturesExtractor


class CustomMLPExtractor(nn.Module):
    """
    Custom MLP extractor with batch normalization example.

    This demonstrates how you can create custom feature extraction networks.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Dict[str, List[int]],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ):
        super().__init__()

        device = (
            device
            if device != "auto"
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Shared layers (if any)
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.latent_dim_pi = 64

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.latent_dim_vf = 64
        self.to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both policy and value networks.

        Args:
            features: Input features

        Returns:
            Tuple of (policy_latent, value_latent)
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through policy network only."""
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network only."""
        return self.value_net(self.shared_net(features))


class CustomActorCritic(ActorCriticPolicy):
    """
    Custom Actor-Critic policy that inherits from stable-baselines3's ActorCriticPolicy.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: callable,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = None,
        features_extractor_kwargs: Optional[Dict] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize the custom actor-critic policy.

        Args:
            observation_space: The observation space
            action_space: The action space
            lr_schedule: Learning rate schedule function
            net_arch: Network architecture specification
            activation_fn: Activation function for the networks
            ortho_init: Whether to use orthogonal initialization
            use_sde: Whether to use State Dependent Exploration
            log_std_init: Initial value for log standard deviation
            full_std: Whether to use full covariance matrix for continuous actions
            use_expln: Whether to use expln function for std
            squash_output: Whether to squash the output using tanh
            features_extractor_class: Features extractor class
            features_extractor_kwargs: Features extractor kwargs
            share_features_extractor: Whether to share features extractor between actor and critic
            normalize_images: Whether to normalize images
            optimizer_class: Optimizer class
            optimizer_kwargs: Optimizer kwargs
        """

        # Set default network architecture if not provided
        if net_arch is None:
            net_arch = dict(pi=[64, 64], vf=[64, 64])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        MAIN METHOD TO OVERRIDE: Build the MLP feature extractor.

        This method creates the shared feature extractor and separate
        policy (actor) and value (critic) networks.
        """
        self.mlp_extractor = CustomMLPExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )


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
        return output[0], output[1]

    def predict(self, observation, deterministic=True):
        """
        Predict action given observation.
        Matches the interface expected by SerowEnv.evaluate().
        Returns action and value
        """
        return self.forward(observation, deterministic=deterministic)
