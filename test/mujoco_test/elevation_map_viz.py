import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Configuration ===
FILENAME = 'data/slope/est_elevation_map.bin'  # <-- Replace with your actual file name
MAP_SIZE = 1024  # 1024 x 1024
CELL_COUNT = MAP_SIZE * MAP_SIZE
HEIGHT_DTYPE = np.float32
TIMESTAMP_DTYPE = np.float64
RECORD_SIZE_BYTES = 8 + 4 * CELL_COUNT  # timestamp (8) + all heights

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

    heights = np.frombuffer(height_data, dtype=HEIGHT_DTYPE)
    heights = heights.reshape((MAP_SIZE, MAP_SIZE))
    return timestamp, heights

def visualize_elevation_map(x, y, heights, timestamp):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, heights, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_title(f"Elevation Map at timestamp {timestamp:.3f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    plt.show()

def main():
    # Precompute X and Y grid
    x = np.linspace(0, MAP_SIZE-1, MAP_SIZE)
    y = np.linspace(0, MAP_SIZE-1, MAP_SIZE)
    X, Y = np.meshgrid(x, y)

    with open(FILENAME, 'rb') as file:
        measurement_idx = 0
        while True:
            timestamp, heights = read_next_measurement(file)
            if timestamp is None:
                print("End of file reached.")
                break

            print(f"Measurement {measurement_idx}, Timestamp: {timestamp:.3f}")
            visualize_elevation_map(X, Y, heights, timestamp)

            measurement_idx += 1
       

if __name__ == "__main__":
    main()