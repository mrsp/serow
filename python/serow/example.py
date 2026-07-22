#!/usr/bin/env python3

import numpy as np
from serow import (
    ForceTorqueMeasurement,
    ImuMeasurement,
    JointMeasurement,
    Serow,
)


def main():
    # Initialize SEROW
    serow = Serow()
    serow.initialize("nao.json")

    # Create the IMU measurement for the IMU in the base
    g = np.array([0.0, 0.0, -9.81])
    imu = ImuMeasurement()
    imu.timestamp = 0.01
    imu.linear_acceleration = np.array([0.1, -0.1, 0.05]) - g
    imu.angular_velocity = np.array([-0.1, 0.1, 0.0])

    # Create the joint measurements for all joints
    joint_names = (
        "HeadYaw",
        "HeadPitch",
        "LHipYawPitch",
        "LHipRoll",
        "LHipPitch",
        "LKneePitch",
        "LAnklePitch",
        "LAnkleRoll",
        "LShoulderPitch",
        "LShoulderRoll",
        "LElbowYaw",
        "LElbowRoll",
        "LWristYaw",
        "LHand",
        "RHipYawPitch",
        "RHipRoll",
        "RHipPitch",
        "RKneePitch",
        "RAnklePitch",
        "RAnkleRoll",
        "RShoulderPitch",
        "RShoulderRoll",
        "RElbowYaw",
        "RElbowRoll",
        "RWristYaw",
        "RHand",
    )
    joints = {}
    for name in joint_names:
        jm = JointMeasurement()
        jm.timestamp = 0.01
        jm.position = 0.0
        joints[name] = jm

    # Create the force torque measurements for the leg end-effectors
    force_torque = {}
    for frame in ("l_ankle", "r_ankle"):
        ft = ForceTorqueMeasurement()
        ft.timestamp = 0.01
        ft.force = np.array([0.0, 0.0, 40.0])
        ft.torque = np.array([0.0, 0.0, 0.0])
        force_torque[frame] = ft

    # Run SEROW
    serow.filter(imu, joints, force_torque)

    # Get the state
    state = serow.get_state(allow_invalid=True)

    # Print parts of the state
    print(f"Base position in world frame: {state.get_base_position()}")
    print(f"Base velocity in world frame: {state.get_base_linear_velocity()}")
    print(f"Base orientation w.r.t the world frame: {state.get_base_orientation()}")
    print(
        f"Left leg contact position in world frame: {state.get_contact_position('l_ankle')}"
    )
    print(
        f"Right leg contact position in world frame: {state.get_contact_position('r_ankle')}"
    )
    print(
        "Left leg contact orientation w.r.t the world frame: "
        f"{state.get_contact_orientation('l_ankle')}"
    )
    print(
        "Right leg contact orientation w.r.t the world frame: "
        f"{state.get_contact_orientation('r_ankle')}"
    )
    print(f"CoM position in world frame: {state.get_com_position()}")
    print(f"CoM linear velocity in world frame: {state.get_com_linear_velocity()}")
    print(f"CoM external forces in world frame: {state.get_com_external_forces()}")


if __name__ == "__main__":
    main()
