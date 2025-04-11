#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from read_mcap import read_kinematic_measurements

def plot_relative_contact_positions(file_path: str):
    """Plot contact positions from kinematic measurements."""
    # Read kinematic measurements
    measurements = read_kinematic_measurements(file_path)
    
    # Extract contact positions and timestamps
    timestamps = []
    contact_positions = {}
    
    for measurement in measurements:
        timestamps.append(measurement.timestamp)
        for name, position in measurement.contacts_position.items():
            if name not in contact_positions:
                contact_positions[name] = []
            contact_positions[name].append(position)
    
    # Convert to numpy arrays
    timestamps = np.array(timestamps)
    for name in contact_positions:
        contact_positions[name] = np.array(contact_positions[name])
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each contact's trajectory
    for name, positions in contact_positions.items():
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                label=name, marker='o', markersize=2)
    
    # Plot settings
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Contact Positions Over Time')
    ax.legend()
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Create 2D subplots for each axis
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    for name, positions in contact_positions.items():
        ax1.plot(timestamps, positions[:, 0], label=name)
        ax2.plot(timestamps, positions[:, 1], label=name)
        ax3.plot(timestamps, positions[:, 2], label=name)
    
    ax1.set_ylabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_xlabel('Time (s)')
    
    ax1.set_title('Contact Positions Over Time')
    ax1.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_relative_contact_positions("/tmp/serow_measurements.mcap") 
