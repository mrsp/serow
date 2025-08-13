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
        joint_state,
        base_state,
        contact_state,
        action_dim,
        state_dim,
        imu_data,
        joint_data,
        ft_data,
        gt_data,
        history_size=20,
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
        self.contact_frames = [cf for cf in contact_state.contacts_status.keys()]
        self.action_dim = action_dim
        self.state_dim = state_dim

        # Action space - discrete choices for measurement noise scaling
        self.discrete_actions = np.array(
            [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
        )
        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))

        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Training data
        max_steps = len(imu_data)
        self.raw_imu_data = imu_data[:max_steps]
        self.raw_joint_data = joint_data[:max_steps]
        self.raw_ft_data = ft_data[:max_steps]
        self.gt_data = gt_data[:max_steps]
        self.valid_prediction = False
        self.max_steps = max_steps
        self.history_size = history_size
        self.measurement_history = [np.zeros(3, dtype=np.float32)] * self.history_size
        self.action_history = [
            np.zeros((self.action_dim,), dtype=np.float32)
        ] * self.history_size

        # Compute the baseline rewards, imu data, and kinematics
        (
            _,
            _,
            _,
            _,
            _,
            self.baseline_rewards,
            self.imu_data,
            self.kinematics,
        ) = self.evaluate(
            model=None,
            stats=None,
            plot=False,
            sync=False,
        )
        self.reset()

    def _compute_reward(self, cf, state, gt):
        reward = 0.0
        done = False

        # Position error
        position_error = np.linalg.norm(state.get_base_position() - gt.position, 1)

        # Orientation error
        orientation_error = np.linalg.norm(
            logMap(
                quaternion_to_rotation_matrix(gt.orientation).transpose()
                @ quaternion_to_rotation_matrix(state.get_base_orientation()),
            ),
            1,
        )

        # Compute innovation and S
        success, innovation, covariance = (
            self.serow_framework.get_contact_position_innovation(cf)
        )

        max_position_error = 3.0
        max_nis = 10.0
        if success:
            nis = innovation @ np.linalg.inv(covariance) @ innovation.T
            nis = np.clip(nis, 0, max_nis)
            position_reward = -position_error / max_position_error
            innovation_reward = -nis / max_nis
            orientation_reward = -orientation_error
            reward = (
                50.0 * innovation_reward
                + 1.0 * position_reward
                + 5.0 * orientation_reward
            )
            if hasattr(self, "baseline_rewards"):
                reward = reward - self.baseline_rewards[self.step_count][cf]

        done = position_error > max_position_error
        return reward, done

    def _get_observation(self, cf, state, kin):
        if not kin.contacts_status[cf] or state.get_contact_position(cf) is None:
            return np.zeros((self.state_dim,))

        R_base = quaternion_to_rotation_matrix(state.get_base_orientation()).transpose()
        local_pos = R_base @ (
            state.get_contact_position(cf) - state.get_base_position()
        )
        local_kin_pos = kin.contacts_position[cf]
        innovation = local_kin_pos - local_pos
        R = (kin.contacts_position_noise[cf] + kin.position_cov).flatten()

        # Ensure histories are properly sized before computing observation
        while len(self.measurement_history) > self.history_size:
            self.measurement_history.pop(0)
        while len(self.action_history) > self.history_size:
            self.action_history.pop(0)

        measurement_history = np.array(self.measurement_history).flatten()
        action_history = np.array(self.action_history).flatten()

        obs = np.concatenate(
            [
                innovation,
                R,
                state.get_base_linear_velocity(),
                state.get_base_orientation(),
                measurement_history,
                action_history,
            ],
            axis=0,
        ).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.serow_framework.reset()
        self.serow_framework.set_state(self.initial_state)
        self.step_count = 0
        self.valid_prediction = False
        self.cf = None
        self.measurement_history = [np.zeros(3)] * self.history_size
        self.action_history = [
            np.zeros((self.action_dim,), dtype=np.float32)
        ] * self.history_size
        obs = np.zeros((self.state_dim,))
        return obs, {}

    def get_observation_for_action(self):
        """Get the observation that should be used for action computation."""
        self.valid_prediction = False
        obs = np.zeros((self.state_dim,))
        # Run prediction step with current control input
        imu = self.imu_data[self.step_count]
        kin = self.kinematics[self.step_count]
        next_kin = self.kinematics[self.step_count + 1]
        prior_state = self.predict_step(imu, kin)

        # Find all the frames that have contact with the ground
        contact_frames = []
        for cf in self.contact_frames:
            if (
                kin.contacts_status[cf]
                and next_kin.contacts_status[cf]
                and prior_state.get_contact_position(cf) is not None
            ):
                contact_frames.append(cf)

        # Pick a random frame in contact for this step
        if len(contact_frames) > 0:
            self.cf = np.random.choice(contact_frames)
            self.valid_prediction = True
            # Get the observation that the policy should use
            obs = self._get_observation(self.cf, prior_state, kin)
        else:
            self.cf = np.random.choice(self.contact_frames)

        return obs

    def step(self, action):
        reward = 0.0
        done = False
        truncated = False
        valid = self.valid_prediction
        obs = np.zeros((self.state_dim,))
        imu = self.imu_data[self.step_count]
        kin = self.kinematics[self.step_count]
        next_kin = self.kinematics[self.step_count + 1]
        # Map the action to the discrete action
        action = np.array([self.discrete_actions[action]], dtype=np.float32)

        if valid:
            post_state = self.update_step(
                self.cf,
                kin,
                action,
            )

            # Compute the reward
            reward, done = self._compute_reward(
                self.cf,
                post_state,
                self.gt_data[self.step_count],
            )

            # Save the action and measurement
            self.action_history.append(action)
            self.measurement_history.append(abs(kin.contacts_position[self.cf]))

            # Get the observation
            obs = self._get_observation(self.cf, post_state, next_kin)

        info = {"step_count": self.step_count, "reward": reward, "valid": valid}

        for cf in self.contact_frames:
            if cf == self.cf and valid:
                continue
            self.update_step(cf, kin, np.array([0.0], dtype=np.float32))

        self.serow_framework.base_estimator_finish_update(imu, kin)
        self.step_count += 1

        truncated = self.step_count == self.max_steps - 1
        if truncated:
            done = False

        return obs, reward, done, truncated, info

    def predict_step(self, imu, kin):
        # Predict the base state
        self.serow_framework.base_estimator_predict_step(imu, kin)

        # Get the state
        state = self.serow_framework.get_state(allow_invalid=True)
        return state

    def update_step(self, cf, kin, action):
        # Set the action
        self.serow_framework.set_action(cf, action)

        # Run the update step with the contact position
        self.serow_framework.base_estimator_update_with_contact_position(cf, kin)

        # Get the post state
        post_state = self.serow_framework.get_state(allow_invalid=True)

        return post_state

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step: {self.step_count}")

    def evaluate(self, model=None, stats=None, plot=True, sync=True):
        # After training, evaluate the policy
        self.reset()

        # Run SEROW
        timestamps = []
        base_positions = []
        base_orientations = []
        gt_positions = []
        gt_orientations = []
        gt_timestamps = []
        rewards = []
        kinematics = []
        imu_data = []
        for _ in range(self.max_steps):
            # Run prediction step with current control input
            imu = copy.copy(self.raw_imu_data[self.step_count])
            joint = copy.copy(self.raw_joint_data[self.step_count])
            ft = copy.copy(self.raw_ft_data[self.step_count])
            imu, kin, ft = self.serow_framework.process_measurements(
                imu, joint, ft, None
            )
            imu_data.append(imu)
            kinematics.append(kin)

            self.serow_framework.base_estimator_predict_step(imu, kin)

            # Run the update step with the contact positions
            post_state = self.serow_framework.get_state(allow_invalid=True)
            reward = {cf: 0.0 for cf in self.contact_frames}
            for cf in self.contact_frames:
                action = np.array([0.0], dtype=np.float32)
                if model is not None:
                    obs = self._get_observation(cf, post_state, kin)
                    if not np.all(obs == np.zeros((self.state_dim,))):
                        if stats is not None:
                            obs = np.array(
                                (obs - np.array(stats["obs_mean"]))
                                / np.sqrt(np.array(stats["obs_var"])),
                                dtype=np.float32,
                            )
                        action, _ = model.predict(obs, deterministic=True)
                        action = np.array(
                            [self.discrete_actions[action]], dtype=np.float32
                        )
                post_state = self.update_step(cf, kin, action)
                reward[cf] = self._compute_reward(
                    cf, post_state, self.gt_data[self.step_count]
                )[0]

            self.serow_framework.base_estimator_finish_update(imu, kin)

            # Save the data
            timestamps.append(self.raw_imu_data[self.step_count].timestamp)
            gt_positions.append(self.gt_data[self.step_count].position)
            gt_orientations.append(self.gt_data[self.step_count].orientation)
            gt_timestamps.append(self.gt_data[self.step_count].timestamp)
            base_positions.append(post_state.get_base_position())
            base_orientations.append(post_state.get_base_orientation())
            rewards.append(reward)

            # Progress to the next sample
            self.step_count += 1

        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        base_positions = np.array(base_positions)
        base_orientations = np.array(base_orientations)
        gt_positions = np.array(gt_positions)
        gt_orientations = np.array(gt_orientations)

        # Sync and align the data
        if sync:
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
            rewards,
            imu_data,
            kinematics,
        )
