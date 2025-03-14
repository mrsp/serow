import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
# === Configuration ===
FILENAME = 'data/flat/est_elevation_map.bin'  # Replace with your actual file name
ESTIMATION_FILE = 'data/flat/serow_predictions.h5'
MAP_SIZE = 1024  # 128 x 128
CELL_COUNT = MAP_SIZE * MAP_SIZE
HEIGHT_DTYPE = np.float32
TIMESTAMP_DTYPE = np.float64

contact_timestamps = None
contact_positions = {}
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

def load_contact_positions(h5_file):
    global contact_timestamps, contact_positions

    with h5py.File(h5_file, 'r') as f:
        print("Available keys in HDF5 file:", list(f.keys()))  # Debugging step
        contact_timestamps = np.array(f[f'/timestamp/t'])
        if 'contact_positions' not in f:
            raise KeyError("Missing '/contact_positions' group in HDF5 file.")

        for foot in ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']:
            x = np.array(f[f'/contact_positions/{foot}/x'])
            y = np.array(f[f'/contact_positions/{foot}/y'])
            z = np.array(f[f'/contact_positions/{foot}/z'])
            contact_positions[foot] = np.stack((x, y, z), axis=1)  # shape: (N, 3)

def visualize_elevation_map_live(file):
    """Continuously updates the 3D elevation map in real-time along with contact positions."""
    global contact_timestamps, contact_positions  # If you're treating them as global

    x = np.linspace(0, MAP_SIZE - 1, MAP_SIZE)
    y = np.linspace(0, MAP_SIZE - 1, MAP_SIZE)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize an empty surface
    heights = np.zeros((MAP_SIZE, MAP_SIZE))
    surf = ax.plot_surface(X, Y, heights, cmap='terrain', edgecolor='none')

    # Placeholder scatter plots for contacts
    scatter_plots = {}
    for foot in contact_positions:
        # Initially dummy point, label added now so legend works
        scatter_plots[foot] = ax.scatter([], [], [], s=50, label=foot)

    # Plot config
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.set_title("Live Elevation Map")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(elev=60, azim=240)
    ax.legend()

    plt.ion()

    measurement_idx = 0
    while True:
        timestamp, heights = read_next_measurement(file)
        if timestamp is None:
            print("End of file reached.")
            break

        print(f"Measurement {measurement_idx}, Timestamp: {timestamp:.3f}")

        # Update surface
        surf.remove()
        surf = ax.plot_surface(X, Y, heights, cmap='terrain', edgecolor='none')

        # Match contact timestamp
        if measurement_idx < contact_timestamps.shape[0]:
            contact_idx = measurement_idx
        else:
            contact_idx = -1  # fallback to last known

        # Update scatter plots for all feet
        for foot in contact_positions:
            pos = contact_positions[foot][contact_idx]  # (x, y, z)
            if (np.linalg.norm(pos) > 0.1):
                scatter_plots[foot]._offsets3d = ([pos[0]+1024//2], [pos[1]+1024//2], [pos[2]])

        ax.set_title(f"Elevation Map at timestamp {timestamp:.3f}")
        plt.pause(0.5)
        measurement_idx += 1

    plt.ioff()
    plt.show()

def visualize_contact_positions_live():
    """Continuously updates the 3D contact positions in real-time."""
    global contact_timestamps, contact_positions  # Assuming these are global variables

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Placeholder scatter plots for contacts (initialize with empty data)
    scatter_plots = {}
    for foot in contact_positions:
        scatter_plots[foot] = ax.scatter([], [], [], s=50, label=foot)

    # Plot config
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Contact Positions in 3D")
    ax.view_init(elev=60, azim=240)  # Adjust view angle
    ax.legend()

    plt.ion()  # Turn on interactive mode

    measurement_idx = 0
    while True:
        # Check if we have a timestamp to process
        if measurement_idx >= len(contact_timestamps):
            print("End of data reached.")
            break

        timestamp = contact_timestamps[measurement_idx]  # Get the current timestamp

        # Update scatter plots for all feet
        for foot in contact_positions:
            pos = contact_positions[foot][measurement_idx]  # (x, y, z)
            if np.linalg.norm(pos) > 0.1:  # Only update if contact position is valid
                scatter_plots[foot]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        ax.set_title(f"Contact Positions at timestamp {timestamp:.3f}")
        
        # Reduce the pause time to speed up the visualization
        plt.pause(0.01)  # Reduced pause time for faster updates
        measurement_idx += 1

    plt.ioff()  # Turn off interactive mode
    plt.show()


    
def main():
    load_contact_positions(ESTIMATION_FILE)
    print(contact_timestamps.shape, contact_positions['FL_foot'].shape)
    # with open(FILENAME, 'rb') as file:
    #     visualize_elevation_map_live(file)
    visualize_contact_positions_live()

if __name__ == "__main__":
    main()