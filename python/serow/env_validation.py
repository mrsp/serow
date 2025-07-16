import numpy as np
import serow
import matplotlib.pyplot as plt
import unittest


def run_serow(dataset, robot, start_idx=0):
    serow_framework = serow.Serow()
    serow_framework.initialize(f"{robot}.json")

    base_positions = []
    base_orientations = []
    for imu, joint, ft in zip(dataset["imu"], dataset["joints"], dataset["ft"]):
        status = serow_framework.filter(imu, joint, ft, None)
        if status:
            state = serow_framework.get_state(allow_invalid=True)
            base_positions.append(state.get_base_position())
            base_orientations.append(state.get_base_orientation())

    return np.array(base_positions), np.array(base_orientations)


def run_serow_playback(dataset, start_idx=0):
    base_positions = []
    base_orientations = []
    i = 0
    for bs in dataset["base_states"]:
        if i < start_idx:
            i += 1
            continue
        base_positions.append(bs.base_position)
        base_orientations.append(bs.base_orientation)

    return np.array(base_positions), np.array(base_orientations)


class TestSerow(unittest.TestCase):
    def setUp(self):
        self.robot = "go2"
        self.dataset = np.load(f"{self.robot}_log.npz", allow_pickle=True)
        print(f"Length of dataset base states: {len(self.dataset['base_states'])}")
        print(f"Length of dataset joint states: {len(self.dataset['joints'])}")
        print(f"Length of dataset imu: {len(self.dataset['imu'])}")
        print(f"Length of dataset ft: {len(self.dataset['ft'])}")

    def test_serow(self):
        base_positions, base_orientations = run_serow(self.dataset, self.robot)
        actual_base_positions, actual_base_orientations = run_serow_playback(
            self.dataset
        )
        print(f"Base positions: {len(base_positions)}")
        print(f"Actual base positions: {len(actual_base_positions)}")
        print(f"Base orientations: {len(base_orientations)}")
        print(f"Actual base orientations: {len(actual_base_orientations)}")

        assert len(base_positions) == len(actual_base_positions)
        assert len(base_orientations) == len(actual_base_orientations)

        position_error = base_positions - actual_base_positions
        orientation_error = base_orientations - actual_base_orientations
        print(f"Actual Position error: {position_error.sum()}, {position_error.max()}")
        print(
            f"Actual Orientation error: {orientation_error.sum()}, {orientation_error.max()}"
        )

        # Plot the base position and orientation
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(actual_base_positions[:, 0], label="x Actual")
        axs[0].plot(base_positions[:, 0], label="x SEROW")

        axs[0].plot(actual_base_positions[:, 1], label="y Actual")
        axs[0].plot(base_positions[:, 1], label="y SEROW")

        axs[0].plot(actual_base_positions[:, 2], label="z Actual")
        axs[0].plot(base_positions[:, 2], label="z SEROW")

        axs[1].plot(actual_base_orientations[:, 0], label="qw Actual")
        axs[1].plot(base_orientations[:, 0], label="qw SEROW")

        axs[1].plot(actual_base_orientations[:, 1], label="qx Actual")
        axs[1].plot(base_orientations[:, 1], label="qx SEROW")

        axs[1].plot(actual_base_orientations[:, 2], label="qy Actual")
        axs[1].plot(base_orientations[:, 2], label="qy SEROW")

        axs[1].plot(actual_base_orientations[:, 3], label="qz Actual")
        axs[1].plot(base_orientations[:, 3], label="qz SEROW")

        axs[0].set_title("Base Position")
        axs[1].set_title("Base Orientation")
        axs[0].legend()
        axs[1].legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()
