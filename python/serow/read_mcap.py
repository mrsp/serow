import mcap
from mcap.reader import make_reader
import numpy as np
from serow.measurement import KinematicMeasurement, ImuMeasurement
from serow.state import BaseState
import flatbuffers
import sys
import os

# Add the build directory to Python path to find generated schemas
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # This points to serow directory
build_dir = os.path.join(project_root, 'build', 'generated')
print(f"Looking for schemas in: {build_dir}")  # Debug print

if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)
else:
    raise ImportError(f"Could not find generated schemas in {build_dir}. Please build the project with Python code generation enabled:\n"
                     f"1. cd {project_root}\n"
                     f"2. mkdir -p build\n"
                     f"3. cd build\n"
                     f"4. cmake -DGENERATE_PYTHON_SCHEMAS=ON ..\n"
                     f"5. make")

try:
    from foxglove.Vector3 import Vector3
    from foxglove.Quaternion import Quaternion
    from foxglove.Matrix3 import Matrix3
    from foxglove.Time import Time
    from foxglove.BaseState import BaseState as FbBaseState
    from foxglove.KinematicMeasurement import KinematicMeasurement as FbKinematicMeasurement
    from foxglove.ImuMeasurement import ImuMeasurement as FbImuMeasurement
except ImportError as e:
    raise ImportError(f"Failed to import FlatBuffer schemas. Please ensure the project is built with Python code generation enabled. Error: {e}")

def decode_imu_measurement(data: bytes) -> ImuMeasurement:
    """Decode a FlatBuffer message into an ImuMeasurement object."""
    fb_msg = FbImuMeasurement.GetRootAsImuMeasurement(data, 0)
    msg = ImuMeasurement()
    
    # Decode timestamp
    timestamp = fb_msg.Timestamp()
    if timestamp:
        msg.timestamp = timestamp.Sec() + timestamp.Nsec() * 1e-9
    
    # Decode linear acceleration
    if fb_msg.LinearAcceleration():
        msg.linear_acceleration = np.array([
            fb_msg.LinearAcceleration().X(),
            fb_msg.LinearAcceleration().Y(),
            fb_msg.LinearAcceleration().Z()
        ])
    
    # Decode angular velocity
    if fb_msg.AngularVelocity():
        msg.angular_velocity = np.array([
            fb_msg.AngularVelocity().X(),
            fb_msg.AngularVelocity().Y(),
            fb_msg.AngularVelocity().Z()
        ])
    
    # Decode orientation
    if fb_msg.Orientation():
        msg.orientation = np.array([
            fb_msg.Orientation().W(),
            fb_msg.Orientation().X(),
            fb_msg.Orientation().Y(),
            fb_msg.Orientation().Z()
        ])
    
    # Decode linear acceleration covariance
    if fb_msg.LinearAccelerationCov():
        matrix = fb_msg.LinearAccelerationCov()
        msg.linear_acceleration_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    # Decode angular velocity covariance
    if fb_msg.AngularVelocityCov():
        matrix = fb_msg.AngularVelocityCov()
        msg.angular_velocity_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    # Decode angular velocity bias covariance
    if fb_msg.AngularVelocityBiasCov():
        matrix = fb_msg.AngularVelocityBiasCov()
        msg.angular_velocity_bias_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    # Decode linear acceleration bias covariance
    if fb_msg.LinearAccelerationBiasCov():
        matrix = fb_msg.LinearAccelerationBiasCov()
        msg.linear_acceleration_bias_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    # Decode angular acceleration
    if fb_msg.AngularAcceleration():
        msg.angular_acceleration = np.array([
            fb_msg.AngularAcceleration().X(),
            fb_msg.AngularAcceleration().Y(),
            fb_msg.AngularAcceleration().Z()
        ])
    
    return msg

def decode_kinematic_measurement(data: bytes) -> KinematicMeasurement:
    """Decode a FlatBuffer message into a KinematicMeasurement object."""
    fb_msg = FbKinematicMeasurement.GetRootAsKinematicMeasurement(data, 0)
    msg = KinematicMeasurement()
    
    # Decode timestamp
    timestamp = fb_msg.Timestamp()
    if timestamp:
        msg.timestamp = timestamp.Sec() + timestamp.Nsec() * 1e-9
    
    # Decode base linear velocity
    if fb_msg.BaseLinearVelocity():
        msg.base_linear_velocity = np.array([
            fb_msg.BaseLinearVelocity().X(),
            fb_msg.BaseLinearVelocity().Y(),
            fb_msg.BaseLinearVelocity().Z()
        ])
    
    # Decode base orientation
    if fb_msg.BaseOrientation():
        msg.base_orientation = np.array([
            fb_msg.BaseOrientation().W(),
            fb_msg.BaseOrientation().X(),
            fb_msg.BaseOrientation().Y(),
            fb_msg.BaseOrientation().Z()
        ])
    
    # Decode contact names and status
    contacts_status = {}
    contacts_probability = {}
    contacts_position = {}
    base_to_foot_positions = {}
    contacts_position_noise = {}
    contacts_orientation = {}
    contacts_orientation_noise = {}

    for i in range(fb_msg.ContactNamesLength()):
        name = fb_msg.ContactNames(i).decode()
        if i < fb_msg.ContactsStatusLength():
            contacts_status[name] = fb_msg.ContactsStatus(i)
        if i < fb_msg.ContactsProbabilityLength():
            contacts_probability[name] = fb_msg.ContactsProbability(i)
        if i < fb_msg.ContactsPositionLength():
            pos = fb_msg.ContactsPosition(i)
            contacts_position[name] = np.array([pos.X(), pos.Y(), pos.Z()])
        if i < fb_msg.BaseToFootPositionsLength():
            pos = fb_msg.BaseToFootPositions(i)
            base_to_foot_positions[name] = np.array([pos.X(), pos.Y(), pos.Z()])
        if i < fb_msg.ContactsPositionNoiseLength():
            matrix = fb_msg.ContactsPositionNoise(i)
            contacts_position_noise[name] = np.array([
                [matrix.M00(), matrix.M01(), matrix.M02()],
                [matrix.M10(), matrix.M11(), matrix.M12()],
                [matrix.M20(), matrix.M21(), matrix.M22()]
            ])
        if i < fb_msg.ContactsOrientationLength():
            quat = fb_msg.ContactsOrientation(i)
            contacts_orientation[name] = np.array([
                quat.W(), quat.X(), quat.Y(), quat.Z()
            ])
        if i < fb_msg.ContactsOrientationNoiseLength():
            matrix = fb_msg.ContactsOrientationNoise(i)
            contacts_orientation_noise[name] = np.array([
                [matrix.M00(), matrix.M01(), matrix.M02()],
                [matrix.M10(), matrix.M11(), matrix.M12()],
                [matrix.M20(), matrix.M21(), matrix.M22()]
            ])
    
    msg.contacts_status = contacts_status
    msg.contacts_probability = contacts_probability
    msg.contacts_position = contacts_position
    msg.base_to_foot_positions = base_to_foot_positions
    msg.contacts_position_noise = contacts_position_noise
    msg.contacts_orientation = contacts_orientation
    msg.contacts_orientation_noise = contacts_orientation_noise

    # Decode COM measurements
    if fb_msg.ComAngularMomentumDerivative():
       msg.com_angular_momentum_derivative = np.array([
            fb_msg.ComAngularMomentumDerivative().X(),
            fb_msg.ComAngularMomentumDerivative().Y(),
            fb_msg.ComAngularMomentumDerivative().Z()
        ])
    
    if fb_msg.ComPosition():
        msg.com_position = np.array([
            fb_msg.ComPosition().X(),
            fb_msg.ComPosition().Y(),
            fb_msg.ComPosition().Z()
        ])
    
    if fb_msg.ComLinearAcceleration():
        msg.com_linear_acceleration = np.array([
            fb_msg.ComLinearAcceleration().X(),
            fb_msg.ComLinearAcceleration().Y(),
            fb_msg.ComLinearAcceleration().Z()
        ])
    
    # Decode covariance matrices
    if fb_msg.BaseLinearVelocityCov():
        matrix = fb_msg.BaseLinearVelocityCov()
        msg.base_linear_velocity_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.BaseOrientationCov():
        matrix = fb_msg.BaseOrientationCov()
        msg.base_orientation_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.PositionSlipCov():
        matrix = fb_msg.PositionSlipCov()
        msg.position_slip_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.OrientationSlipCov():
        matrix = fb_msg.OrientationSlipCov()
        msg.orientation_slip_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.PositionCov():
        matrix = fb_msg.PositionCov()
        msg.position_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.OrientationCov():
        matrix = fb_msg.OrientationCov()
        msg.orientation_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ComPositionProcessCov():
        matrix = fb_msg.ComPositionProcessCov()
        msg.com_position_process_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ComLinearVelocityProcessCov():
        matrix = fb_msg.ComLinearVelocityProcessCov()
        msg.com_linear_velocity_process_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ExternalForcesProcessCov():
        matrix = fb_msg.ExternalForcesProcessCov()
        msg.external_forces_process_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ComPositionCov():
        matrix = fb_msg.ComPositionCov()
        msg.com_position_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ComLinearAccelerationCov():
        matrix = fb_msg.ComLinearAccelerationCov()
        msg.com_linear_acceleration_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    return msg

def decode_base_state(data: bytes) -> BaseState:
    """Decode a FlatBuffer message into an BaseState object."""
    fb_msg = FbBaseState.GetRootAsBaseState(data, 0)
    msg = BaseState()
    
    # Decode timestamp
    timestamp = fb_msg.Timestamp()
    if timestamp:
        msg.timestamp = timestamp.Sec() + timestamp.Nsec() * 1e-9
    
    # Decode base position
    if fb_msg.BasePosition():
        msg.base_position = np.array([
            fb_msg.BasePosition().X(),
            fb_msg.BasePosition().Y(),
            fb_msg.BasePosition().Z()
        ])
    
    # Decode base orientation
    if fb_msg.BaseOrientation():
        msg.base_orientation = np.array([
            fb_msg.BaseOrientation().W(),
            fb_msg.BaseOrientation().X(),
            fb_msg.BaseOrientation().Y(),
            fb_msg.BaseOrientation().Z()
        ])
    
    # Decode base linear velocity
    if fb_msg.BaseLinearVelocity():
        msg.base_linear_velocity = np.array([
            fb_msg.BaseLinearVelocity().X(),
            fb_msg.BaseLinearVelocity().Y(),
            fb_msg.BaseLinearVelocity().Z()
        ])
    
    # Decode base angular velocity
    if fb_msg.BaseAngularVelocity():
        msg.base_angular_velocity = np.array([
            fb_msg.BaseAngularVelocity().X(),
            fb_msg.BaseAngularVelocity().Y(),
            fb_msg.BaseAngularVelocity().Z()
        ])
    
    # Decode base linear acceleration
    if fb_msg.BaseLinearAcceleration():
        msg.base_linear_acceleration = np.array([
            fb_msg.BaseLinearAcceleration().X(),
            fb_msg.BaseLinearAcceleration().Y(),
            fb_msg.BaseLinearAcceleration().Z()
        ])
    
    # Decode base angular acceleration
    if fb_msg.BaseAngularAcceleration():
        msg.base_angular_acceleration = np.array([
            fb_msg.BaseAngularAcceleration().X(),
            fb_msg.BaseAngularAcceleration().Y(),
            fb_msg.BaseAngularAcceleration().Z()
        ])
    
    # Decode IMU linear acceleration bias
    if fb_msg.ImuLinearAccelerationBias():
        msg.imu_linear_acceleration_bias = np.array([
            fb_msg.ImuLinearAccelerationBias().X(),
            fb_msg.ImuLinearAccelerationBias().Y(),
            fb_msg.ImuLinearAccelerationBias().Z()
        ])
    
    # Decode IMU angular velocity bias
    if fb_msg.ImuAngularVelocityBias():
        msg.imu_angular_velocity_bias = np.array([
            fb_msg.ImuAngularVelocityBias().X(),
            fb_msg.ImuAngularVelocityBias().Y(),
            fb_msg.ImuAngularVelocityBias().Z()
        ])
    
    # Decode covariance matrices
    if fb_msg.BasePositionCov():
        matrix = fb_msg.BasePositionCov()
        msg.base_position_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.BaseOrientationCov():
        matrix = fb_msg.BaseOrientationCov()
        msg.base_orientation_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.BaseLinearVelocityCov():
        matrix = fb_msg.BaseLinearVelocityCov()
        msg.base_linear_velocity_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.BaseAngularVelocityCov():
        matrix = fb_msg.BaseAngularVelocityCov()
        msg.base_angular_velocity_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ImuLinearAccelerationBiasCov():
        matrix = fb_msg.ImuLinearAccelerationBiasCov()
        msg.imu_linear_acceleration_bias_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    if fb_msg.ImuAngularVelocityBiasCov():
        matrix = fb_msg.ImuAngularVelocityBiasCov()
        msg.imu_angular_velocity_bias_cov = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    # Decode contact positions
    contacts_position = {}
    for i in range(fb_msg.ContactsPositionLength()):
        pos = fb_msg.ContactsPosition(i)
        contacts_position[fb_msg.ContactNames(i).decode()] = np.array([
            pos.X(), pos.Y(), pos.Z()
        ])
    
    # Decode contact orientations
    contacts_orientation = {}
    for i in range(fb_msg.ContactsOrientationLength()):
        quat = fb_msg.ContactsOrientation(i)
        contacts_orientation[fb_msg.ContactNames(i).decode()] = np.array([
            quat.W(), quat.X(), quat.Y(), quat.Z()
        ])
    
    # Decode contact position covariances
    contacts_position_cov = {}
    for i in range(fb_msg.ContactsPositionCovLength()):
        matrix = fb_msg.ContactsPositionCov(i)
        contacts_position_cov[fb_msg.ContactNames(i).decode()] = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    # Decode contact orientation covariances
    contacts_orientation_cov = {}
    for i in range(fb_msg.ContactsOrientationCovLength()):
        matrix = fb_msg.ContactsOrientationCov(i)
        contacts_orientation_cov[fb_msg.ContactNames(i).decode()] = np.array([
            [matrix.M00(), matrix.M01(), matrix.M02()],
            [matrix.M10(), matrix.M11(), matrix.M12()],
            [matrix.M20(), matrix.M21(), matrix.M22()]
        ])
    
    msg.contacts_position = contacts_position
    msg.contacts_orientation = contacts_orientation
    msg.contacts_position_cov = contacts_position_cov
    msg.contacts_orientation_cov = contacts_orientation_cov

    return msg

def read_imu_measurements(file_path: str):
    """Read and decode messages from an MCAP file."""
    with open(file_path, "rb") as f:
        reader = make_reader(f)
        imu_measurements = []
        
        for schema, channel, message in reader.iter_messages():
            if channel.topic == "/imu":
                msg = decode_imu_measurement(message.data)
                imu_measurements.append(msg)
        
        return imu_measurements

def read_kinematic_measurements(file_path: str):
    """Read and decode messages from an MCAP file."""
    with open(file_path, "rb") as f:
        reader = make_reader(f)
        kinematic_measurements = []
        
        for schema, channel, message in reader.iter_messages():
            if channel.topic == "/kin":
                msg = decode_kinematic_measurement(message.data)
                kinematic_measurements.append(msg)
        
        return kinematic_measurements

def read_base_states(file_path: str):
    """Read and decode the base states from an MCAP file.
    
    Args:
        file_path: Path to the MCAP file
        
    Returns:
        BaseState: The decoded base states, or None if not found
    """
    with open(file_path, "rb") as f:
        reader = make_reader(f)
        base_states = []    
        for schema, channel, message in reader.iter_messages():
            if channel.topic == "/base_state":
                base_states.append(decode_base_state(message.data))
        
        return base_states
