from env import SerowEnv
import numpy as np

if __name__ == "__main__":
    # Load and preprocess the data
    robot = "go2"
    dataset = np.load(f"{robot}_log.npz", allow_pickle=True)
    test_dataset = dataset
    imu_measurements = dataset["imu"]
    contact_states = dataset["contact_states"]
    dataset_size = len(imu_measurements) - 1

    # Define the dimensions of your state and action spaces
    history_buffer_size = 100
    state_history_dim = (
        9 * history_buffer_size + 3 * history_buffer_size + history_buffer_size
    )
    state_fb_dim = 3 * 3 + 3 + 2
    state_dim = state_fb_dim + state_history_dim
    print(f"State dimension: {state_dim}")
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()
    min_action = np.array([1e-8])
    max_action = np.array([1e4])

    # Create the evaluation environment and get the contacts frames
    contact_frame = list(contact_states[0].contacts_status.keys())[0]
    env = SerowEnv(
        robot,
        contact_frame,
        dataset["joint_states"][0],
        dataset["base_states"][0],
        dataset["contact_states"][0],
        action_dim,
        state_dim,
    )
    contact_frames = env.contact_frames
    print(f"Contacts frame: {contact_frames}")

    env.evaluate(dataset)
