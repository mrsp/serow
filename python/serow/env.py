import serow
import numpy as np
import gymnasium as gym
import copy
from utils import (
    quaternion_to_rotation_matrix,
    logMap,
    sync_and_align_data,
    plot_trajectories,
)


class SerowEnv(gym.Env):
    def __init__(
        self,
        robot,
        contact_frame,
        joint_state,
        base_state,
        contact_state,
        action_dim,
        state_dim,
        action_low,
        action_high,
        imu_data,
        joint_data,
        ft_data,
        gt_data,
    ):
        super(SerowEnv, self).__init__()

        # Environment parameters
        self.robot = robot
        self.serow_framework = serow.Serow()
        self.serow_framework.initialize(f"{robot}_rl.json")
        self.initial_state = self.serow_framework.get_state(allow_invalid=True)
        self.initial_state.set_joint_state(joint_state)
        self.initial_state.set_base_state(base_state)
        self.initial_state.set_contact_state(contact_state)
        self.serow_framework.set_state(self.initial_state)
        self.cf = contact_frame  # contact frame to control
        self.contact_frames = [
            cf for cf in contact_state.contacts_status.keys() if cf != self.cf
        ]
        self.all_contact_frames = [cf for cf in contact_state.contacts_status.keys()]
        self.action_dim = action_dim
        self.state_dim = state_dim
        # Action space
        self.action_space = gym.spaces.Box(
            low=action_low, high=action_high, shape=(action_dim,), dtype=np.float32
        )
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Training data
        max_steps = len(imu_data) - 1
        self.max_steps = max_steps

        # Get the measurements and the ground truth
        self.imu_data = imu_data[:max_steps]
        self.joint_data = joint_data[:max_steps]
        self.ft_data = ft_data[:max_steps]
        self.gt_data = gt_data[:max_steps]
        self.kin = None
        self.imu = None

    def _compute_reward(self, cf, state, gt, step, max_steps):
        reward = 0.0
        done = False

        # Position error
        position_error = np.linalg.norm(state.get_base_position() - gt.position)

        # Orientation error
        orientation_error = np.linalg.norm(
            logMap(
                quaternion_to_rotation_matrix(gt.orientation).transpose()
                @ quaternion_to_rotation_matrix(state.get_base_orientation())
            )
        )

        # Compute innovation and S
        success, innovation, covariance = (
            self.serow_framework.get_contact_position_innovation(cf)
        )

        max_position_error = 3.0
        max_orientation_error = 1.0
        if (
            position_error > max_position_error
            or orientation_error > max_orientation_error
        ):
            done = True
            reward = -10.0
        else:
            if success:
                nis = innovation @ np.linalg.inv(covariance) @ innovation.T
                position_reward = 5.0 * np.exp(-2.0 * position_error)
                innovation_reward = 1.0 * np.exp(-2.0 * nis)
                orientation_reward = 10.0 * np.exp(-2.0 * orientation_error)
                step_reward = 1.0 * (step + 1) / max_steps
                reward = innovation_reward + position_reward + orientation_reward
                reward += step_reward
                # Scale down the reward to prevent value function issues
                reward = reward * 0.005

        return reward, done

    def _get_observation(self, cf, state, kin):
        if (
            kin is None
            or not kin.contacts_status[cf]
            or state.get_contact_position(cf) is None
        ):
            # Return a more informative observation instead of all zeros
            # Include base state information even when contact is not available
            P_pos_trace = np.trace(state.get_base_position_cov())
            P_ori_trace = np.trace(state.get_base_orientation_cov())
            return np.concatenate(
                [
                    [P_pos_trace],
                    [P_ori_trace],
                    np.zeros(3),  # innovation (3D)
                    np.zeros(9),  # R covariance (3x3 flattened)
                    state.get_base_linear_velocity(),
                    state.get_base_orientation(),
                ],
                axis=0,
            ).astype(np.float32)

        R_base = quaternion_to_rotation_matrix(state.get_base_orientation()).transpose()
        local_pos = R_base @ (
            state.get_contact_position(cf) - state.get_base_position()
        )
        local_kin_pos = kin.contacts_position[cf]
        innovation = local_kin_pos - local_pos
        R = (kin.contacts_position_noise[cf] + kin.position_cov).flatten()
        P_pos_trace = np.trace(state.get_base_position_cov())
        P_ori_trace = np.trace(state.get_base_orientation_cov())

        return np.concatenate(
            [
                [P_pos_trace],
                [P_ori_trace],
                innovation,
                R,
                state.get_base_linear_velocity(),
                state.get_base_orientation(),
            ],
            axis=0,
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.serow_framework.reset()
        self.serow_framework.set_state(self.initial_state)
        self.step_count = 0
        obs = self._get_observation(self.cf, self.initial_state, self.kin)
        return obs, {}

    def get_observation_for_action(self, cf=None):
        """Get the observation that should be used for action computation."""
        if cf is None:
            cf = self.cf

        # Run prediction step with current control input
        self.imu = copy.copy(self.imu_data[self.step_count])
        joint = copy.copy(self.joint_data[self.step_count])
        ft = copy.copy(self.ft_data[self.step_count])
        prior_state, self.kin = self.predict_step(self.imu, joint, ft)

        # Get the observation that the policy should use
        obs = self._get_observation(cf, prior_state, self.kin)
        return obs

    def step(self, action):
        reward = 0.0
        done = False
        truncated = False
        valid = False
        obs = np.zeros((self.state_dim,))

        if self.kin.contacts_status[self.cf]:
            post_state, reward, done = self.update_step(
                self.cf,
                action,
            )
            obs = self._get_observation(self.cf, post_state, self.kin)

        if not np.all(obs == np.zeros((self.state_dim,))):
            valid = True

        info = {"step_count": self.step_count, "reward": reward, "valid": valid}

        for cf in self.contact_frames:
            self.update_step(cf, np.zeros((self.action_dim, 1), dtype=np.float32))

        self.finish_update(self.imu, self.kin)
        self.step_count += 1

        if self.step_count >= self.max_steps:
            done = False
            truncated = True

        return obs, reward, done, truncated, info

    def predict_step(self, imu, joint, ft):
        # Process the measurements
        result = self.serow_framework.process_measurements(imu, joint, ft, None)

        if result is None:
            return None, None
        imu, kin, ft = result

        # Predict the base state
        self.serow_framework.base_estimator_predict_step(imu, kin)

        # Get the state
        state = self.serow_framework.get_state(allow_invalid=True)
        return state, kin

    def update_step(self, cf, action):
        # Set the action
        self.serow_framework.set_action(cf, action)

        # Run the update step with the contact position
        self.serow_framework.base_estimator_update_with_contact_position(cf, self.kin)

        # Get the post state
        post_state = self.serow_framework.get_state(allow_invalid=True)

        # Compute the reward
        reward, done = self._compute_reward(
            cf,
            post_state,
            self.gt_data[self.step_count],
            self.step_count,
            self.max_steps,
        )

        return post_state, reward, done

    def finish_update(self, imu, kin):
        self.serow_framework.base_estimator_finish_update(imu, kin)

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step: {self.step_count}")

    def evaluate(self, model=None, stats=None, plot=True):
        # After training, evaluate the policy
        self.reset()

        # Run SEROW
        timestamps = []
        base_positions = []
        base_orientations = []
        gt_positions = []
        gt_orientations = []
        gt_timestamps = []

        for _ in range(self.max_steps):
            # Run prediction step with current control input
            self.imu = copy.copy(self.imu_data[self.step_count])
            joint = copy.copy(self.joint_data[self.step_count])
            ft = copy.copy(self.ft_data[self.step_count])
            prior_state, self.kin = self.predict_step(self.imu, joint, ft)

            # Run the update step with the contact positions
            post_state = prior_state
            for cf in self.all_contact_frames:
                action = np.zeros((self.action_dim, 1), dtype=np.float32)
                if model is not None:
                    obs = self._get_observation(cf, post_state, self.kin)
                    if stats is not None:
                        obs = np.array(
                            (obs - np.array(stats["obs_mean"]))
                            / np.sqrt(np.array(stats["obs_var"])),
                            dtype=np.float32,
                        )
                    action, _ = model.predict(obs, deterministic=True)
                post_state, _, _ = self.update_step(cf, action)

            self.finish_update(self.imu, self.kin)

            # Save the data
            timestamps.append(self.imu_data[self.step_count].timestamp)
            gt_positions.append(self.gt_data[self.step_count].position)
            gt_orientations.append(self.gt_data[self.step_count].orientation)
            gt_timestamps.append(self.gt_data[self.step_count].timestamp)
            base_positions.append(post_state.get_base_position())
            base_orientations.append(post_state.get_base_orientation())

            # Progress to the next sample
            self.step_count += 1

        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        base_positions = np.array(base_positions)
        base_orientations = np.array(base_orientations)
        gt_positions = np.array(gt_positions)
        gt_orientations = np.array(gt_orientations)

        # Sync and align the data
        (
            timestamps,
            base_positions,
            base_orientations,
            gt_positions,
            gt_orientations,
        ) = sync_and_align_data(
            timestamps,
            base_positions,
            base_orientations,
            gt_timestamps,
            gt_positions,
            gt_orientations,
            align=True,
        )

        # Plot the trajectories
        if plot:
            plot_trajectories(
                timestamps,
                base_positions,
                base_orientations,
                gt_positions,
                gt_orientations,
            )
        return (
            timestamps,
            base_positions,
            base_orientations,
            gt_positions,
            gt_orientations,
        )
