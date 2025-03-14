import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Configuration ===
FILENAME = 'data/slope/est_elevation_map.bin'  # Replace with your actual file name
MAP_SIZE = 1024  # 128 x 128
CELL_COUNT = MAP_SIZE * MAP_SIZE
HEIGHT_DTYPE = np.float32
TIMESTAMP_DTYPE = np.float64


def read_next_measurement(file):
    """Reads one measurement (timestamp + height map) from file."""
    timestamp_data = file.read(8)
    if not timestamp_data:
        return None, None  # End of file

    timestamp = struct.unpack('d', timestamp_data)[0]
    height_data = file.read(CELL_COUNT * 4)

    if len(height_data) != CELL_COUNT * 4:
        print("Incomplete record or end of file.")
        return None, None

    heights = np.frombuffer(height_data, dtype=HEIGHT_DTYPE).reshape((MAP_SIZE, MAP_SIZE))
    return timestamp, heights


def visualize_elevation_map_live(file):
    """Continuously updates the 3D elevation map in real-time."""
    x = np.linspace(0, MAP_SIZE - 1, MAP_SIZE)
    y = np.linspace(0, MAP_SIZE - 1, MAP_SIZE)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize an empty surface
    heights = np.zeros((MAP_SIZE, MAP_SIZE))  # Placeholder for initial frame
    surf = ax.plot_surface(X, Y, heights, cmap='terrain', edgecolor='none')

    # Configure the plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.set_title("Live Elevation Map")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(elev=60, azim=240)  # Adjust view angle

    plt.ion()  # Turn on interactive mode

    measurement_idx = 0
    while True:
        timestamp, heights = read_next_measurement(file)
        if timestamp is None:
            print("End of file reached.")
            break

        print(f"Measurement {measurement_idx}, Timestamp: {timestamp:.3f}")

        # Update surface data
        surf.remove()  # Remove the old surface
        surf = ax.plot_surface(X, Y, heights, cmap='terrain', edgecolor='none')

        ax.set_title(f"Elevation Map at timestamp {timestamp:.3f}")  # Update title

        plt.pause(0.5)  # Pause to create a smooth update effect
        measurement_idx += 1

    plt.ioff()  # Turn off interactive mode
    plt.show()


def main():
    with open(FILENAME, 'rb') as file:
        visualize_elevation_map_live(file)


if __name__ == "__main__":
    main()
