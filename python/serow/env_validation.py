import numpy as np
import serow
import matplotlib.pyplot as plt
import unittest
import copy


def run_serow(dataset, robot, start_idx=0):
    serow_framework = serow.Serow()
    serow_framework.initialize(f"{robot}_rl.json")
    initial_state = serow_framework.get_state(allow_invalid=True)
    initial_state.set_joint_state(dataset["joint_states"][start_idx])
    initial_state.set_base_state(dataset["base_states"][start_idx])
    initial_state.set_contact_state(dataset["contact_states"][start_idx])
    serow_framework.set_state(initial_state)
    contact_frames = initial_state.get_contacts_frame()

    base_positions = []
    base_orientations = []
    for imu, joint, ft, _ in zip(
        dataset["imu"],
        dataset["joints"],
        dataset["ft"],
        dataset["base_pose_ground_truth"],
    ):
        result = serow_framework.process_measurements(imu, joint, ft, None)
        if result is None:
            continue

        imu, kin, ft = result

        serow_framework.base_estimator_predict_step(imu, kin)
        serow_framework.base_estimator_update_with_imu_orientation(imu)
        for cf in contact_frames:
            serow_framework.base_estimator_update_with_contact_position(cf, kin)
        serow_framework.base_estimator_finish_update(imu, kin)

        state = serow_framework.get_state(allow_invalid=True)

        base_positions.append(state.get_base_position())
        base_orientations.append(state.get_base_orientation())

    return np.array(base_positions), np.array(base_orientations)


def run_serow1(dataset, robot, start_idx=0):
    serow_framework = serow.Serow()
    serow_framework.initialize(f"{robot}_rl.json")
    initial_state = serow_framework.get_state(allow_invalid=True)
    initial_state.set_joint_state(dataset["joint_states"][start_idx])
    initial_state.set_base_state(dataset["base_states"][start_idx])
    initial_state.set_contact_state(dataset["contact_states"][start_idx])
    serow_framework.set_state(initial_state)

    base_positions = []
    base_orientations = []
    for imu, joint, ft, _ in zip(
        dataset["imu"],
        dataset["joints"],
        dataset["ft"],
        dataset["base_pose_ground_truth"],
    ):
        serow_framework.filter(imu, joint, ft, None)
        state = serow_framework.get_state(allow_invalid=True)
        base_positions.append(state.get_base_position())
        base_orientations.append(state.get_base_orientation())

    return np.array(base_positions), np.array(base_orientations)


class TestSerow(unittest.TestCase):
    def setUp(self):
        self.robot = "go2"
        self.dataset0 = np.load(f"{self.robot}_training_dataset.npz", allow_pickle=True)
        self.dataset1 = copy.copy(self.dataset0)

    def test_serow(self):
        base_positions0, base_orientations0 = run_serow(self.dataset0, self.robot)
        base_positions1, base_orientations1 = run_serow(self.dataset1, self.robot)

        position_error = base_positions0 - base_positions1
        orientation_error = base_orientations0 - base_orientations1
        print(f"Position error: {position_error.sum()}, {position_error.max()}")
        print(
            f"Orientation error: {orientation_error.sum()}, {orientation_error.max()}"
        )
        assert np.allclose(position_error, np.zeros_like(position_error), atol=1e-3)
        assert np.allclose(
            orientation_error, np.zeros_like(orientation_error), atol=1e-3
        )

        # Plot the base position and orientation
        fig, axs = plt.subplots(2, 1)
        base_positions0 = np.array(base_positions0)
        base_positions1 = np.array(base_positions1)
        base_orientations0 = np.array(base_orientations0)
        base_orientations1 = np.array(base_orientations1)
        # Plot the base position and orientation
        axs[0].plot(base_positions0[:, 0], label="x SEROW 0")
        axs[0].plot(base_positions1[:, 0], label="x SEROW 1")
        axs[0].plot(base_positions0[:, 1], label="y SEROW 0")
        axs[0].plot(base_positions1[:, 1], label="y SEROW 1")
        axs[0].plot(base_positions0[:, 2], label="z SEROW 0")
        axs[0].plot(base_positions1[:, 2], label="z SEROW 1")

        axs[1].plot(base_orientations0[:, 0], label="qw SEROW 0")
        axs[1].plot(base_orientations1[:, 0], label="qw SEROW 1")
        axs[1].plot(base_orientations0[:, 1], label="qx SEROW 0")
        axs[1].plot(base_orientations1[:, 1], label="qx SEROW 1")
        axs[1].plot(base_orientations0[:, 2], label="qy SEROW 0")
        axs[1].plot(base_orientations1[:, 2], label="qy SEROW 1")
        axs[1].plot(base_orientations0[:, 3], label="qz SEROW 0")
        axs[1].plot(base_orientations1[:, 3], label="qz SEROW 1")

        axs[0].set_title("Base Position")
        axs[1].set_title("Base Orientation")
        axs[0].legend()
        axs[1].legend()
        plt.show()

    def test_serow_random_start(self):
        start_idx = 8000
        base_positions0, base_orientations0 = run_serow(
            self.dataset0, self.robot, start_idx=start_idx
        )
        base_positions1, base_orientations1 = run_serow(
            self.dataset1, self.robot, start_idx=start_idx
        )

        position_error = base_positions0 - base_positions1
        orientation_error = base_orientations0 - base_orientations1
        print(f"Position error: {position_error.sum()}, {position_error.max()}")
        print(
            f"Orientation error: {orientation_error.sum()}, {orientation_error.max()}"
        )
        assert np.allclose(position_error, np.zeros_like(position_error), atol=1e-3)
        assert np.allclose(
            orientation_error, np.zeros_like(orientation_error), atol=1e-3
        )

        # Plot the base position and orientation
        fig, axs = plt.subplots(2, 1)
        base_positions0 = np.array(base_positions0)
        base_positions1 = np.array(base_positions1)
        base_orientations0 = np.array(base_orientations0)
        base_orientations1 = np.array(base_orientations1)
        # Plot the base position and orientation
        axs[0].plot(base_positions0[:, 0], label="x SEROW 0")
        axs[0].plot(base_positions1[:, 0], label="x SEROW 1")
        axs[0].plot(base_positions0[:, 1], label="y SEROW 0")
        axs[0].plot(base_positions1[:, 1], label="y SEROW 1")
        axs[0].plot(base_positions0[:, 2], label="z SEROW 0")
        axs[0].plot(base_positions1[:, 2], label="z SEROW 1")

        axs[1].plot(base_orientations0[:, 0], label="qw SEROW 0")
        axs[1].plot(base_orientations1[:, 0], label="qw SEROW 1")
        axs[1].plot(base_orientations0[:, 1], label="qx SEROW 0")
        axs[1].plot(base_orientations1[:, 1], label="qx SEROW 1")
        axs[1].plot(base_orientations0[:, 2], label="qy SEROW 0")
        axs[1].plot(base_orientations1[:, 2], label="qy SEROW 1")
        axs[1].plot(base_orientations0[:, 3], label="qz SEROW 0")
        axs[1].plot(base_orientations1[:, 3], label="qz SEROW 1")

        axs[0].set_title("Base Position")
        axs[1].set_title("Base Orientation")
        axs[0].legend()
        axs[1].legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()
