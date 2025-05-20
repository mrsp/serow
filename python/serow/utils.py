import numpy as np

def normalize_quaternion(quaternion):
    """
    Normalize a quaternion to ensure it represents a valid rotation.
    
    Args:
        quaternion (list or numpy array): A quaternion [x, y, z, w]

    Returns:
        numpy array: A normalized quaternion
    """
    # Convert quaternion to numpy array if it's a list
    if isinstance(quaternion, list):
        quaternion = np.array(quaternion)

    # Normalize the quaternion
    quaternion = quaternion / np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion

    return quaternion

def normalize_vector(vector, min_value, max_value, target_range=(0, 1)):
    """
    Normalize a vector to ensure it is within the specified target range.
    
    Args:
        vector (numpy array or list): The vector to normalize
        min_value (numpy array or list): The minimum value of the input range. 
        max_value (numpy array or list): The maximum value of the input range. 
        target_range (tuple, optional): The target range (min, max) for normalization. Defaults to (0, 1)

    Returns:
        numpy array: The normalized vector
    """
    # Convert to numpy array if input is a list
    if isinstance(vector, list):
        vector = np.array(vector)
    if isinstance(min_value, list):
        min_value = np.array(min_value)
    if isinstance(max_value, list):
        max_value = np.array(max_value)

    # Check if min and max are different to avoid division by zero
    if np.all(min_value == max_value):
        return np.full_like(vector, target_range[0])
    
    # Normalize to [0, 1] first
    normalized = (vector - min_value) / (max_value - min_value)
    
    # Scale to target range
    return normalized * (target_range[1] - target_range[0]) + target_range[0]
