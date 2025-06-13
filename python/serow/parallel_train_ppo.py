#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import multiprocessing
import traceback

from env import SerowEnv
from ppo import PPO

from read_mcap import(
    read_base_states,
    read_contact_states,
    read_force_torque_measurements,
    read_joint_measurements,
    read_imu_measurements,
    read_base_pose_ground_truth,
    read_joint_states
)

from utils import(
    quaternion_to_rotation_matrix,
    export_models_to_onnx
)

class SharedNetwork(nn.Module):
    def __init__(self, params):
        super(SharedNetwork, self).__init__()
        try:
            # Verify state_dim is valid
            if not isinstance(params['state_dim'], int) or params['state_dim'] <= 0:
                raise ValueError(f"Invalid state_dim: {params['state_dim']}")
            
            self.layer1 = nn.Linear(params['state_dim'], 64)
            self.layer2 = nn.Linear(64, 64)
            
            try:
                nn.init.xavier_uniform_(self.layer1.weight)
                nn.init.xavier_uniform_(self.layer2.weight)
                torch.nn.init.zeros_(self.layer1.bias)
                torch.nn.init.zeros_(self.layer2.bias)
            except Exception as e:
                print(f"SharedNetwork: Weight initialization traceback: {traceback.format_exc()}")
                raise
        except Exception as e:
            print(f"SharedNetwork: Traceback: {traceback.format_exc()}", flush=True)
            raise

    def forward(self, state):
        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return x

# Actor network per leg end-effector
class Actor(nn.Module):
    def __init__(self, params, shared_network):
        super(Actor, self).__init__()
        try:
            self.shared_network = shared_network
            self.mean_layer = nn.Linear(64, params['action_dim'])
            self.log_std = nn.Parameter(torch.zeros(params['action_dim']))
            self.min_action = params['min_action']
            self.action_dim = params['action_dim']
            try:
                nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
                torch.nn.init.constant_(self.mean_layer.bias, 0.0)
            except Exception as e:
                print(f"Actor: Traceback: {traceback.format_exc()}", flush=True)
                raise
        except Exception as e:
            print(f"Actor: Traceback: {traceback.format_exc()}", flush=True)
            raise

    def forward(self, state):
        x = self.shared_network(state)
        x = F.relu(x)
        mean = self.mean_layer(x)
        log_std = self.log_std.clamp(-20, 2)
        return mean, log_std

    def get_action(self, state, deterministic=False):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()

        action_scaled = F.softplus(action) + self.min_action

        # Calculate log probability
        if not deterministic:
            # Log prob with softplus correction
            log_prob = normal.log_prob(action).sum(dim=-1)
            # Apply softplus correction (derivative of softplus is sigmoid)
            log_prob -= torch.log(torch.sigmoid(action) + 1e-8).sum(dim=-1)
        else:
            log_prob = torch.zeros(1)

        return action_scaled.detach().cpu().numpy(), log_prob.detach().cpu().item()

    def evaluate_actions(self, states, actions):
        """Evaluate log probabilities and entropy for given state-action pairs"""
        mean, log_std = self.forward(states)
        std = log_std.exp()

        # Convert actions back to raw space (inverse of softplus scaling)
        actions_raw = torch.log(actions - self.min_action + 1e-8)

        # Calculate log probabilities
        normal = torch.distributions.Normal(mean, std)
        log_probs = normal.log_prob(actions_raw).sum(dim=-1, keepdim=True)

        # Apply softplus correction (derivative of softplus is sigmoid)
        log_probs -= torch.log(torch.sigmoid(actions_raw) + 1e-8).sum(dim=-1, keepdim=True)

        # Calculate entropy
        entropy = normal.entropy().sum(dim=-1)
        return log_probs, entropy

class Critic(nn.Module):
    def __init__(self, params, shared_network):
        super(Critic, self).__init__()
        try:
            self.shared_network = shared_network
            self.value_layer = nn.Linear(64, 1)
            try:
                nn.init.orthogonal_(self.value_layer.weight, gain=1.0)
                torch.nn.init.constant_(self.value_layer.bias, 0.0)
            except Exception as e:
                print(f"Critic: Traceback: {traceback.format_exc()}", flush=True)
                raise
        except Exception as e:
            print(f"Critic: Traceback: {traceback.format_exc()}", flush=True)
            raise

    def forward(self, state):
        x = self.shared_network(state)
        x = F.relu(x)
        return self.value_layer(x)


def collect_experience_worker(
    worker_id, dataset, params, shared_actor_state_dict, shared_critic_state_dict,
    shared_buffer, training_event, update_event, device, episode_queue, max_steps
):
    """
    Worker process to collect experience from an environment and add to a shared buffer.
    """
    try:
        print(f"[Worker {worker_id}] Starting initialization...", flush=True)
        # Initialize a local agent for experience collection
        shared_network_worker = SharedNetwork(params)
        actor_worker = Actor(params, shared_network_worker).to(device)
        critic_worker = Critic(params, shared_network_worker).to(device)
        agent_worker = PPO(actor_worker, critic_worker, params, device=device, normalize_state=True)
        agent_worker.train()
    except Exception as e:
        print(f"[Worker {worker_id}] Traceback: {traceback.format_exc()}", flush=True)
        raise
    print(f"[Worker {worker_id}] Agent initialized successfully", flush=True)

    # Get dataset measurements
    imu_measurements = dataset['imu']
    joint_measurements = dataset['joints']
    force_torque_measurements = dataset['ft']
    base_pose_ground_truth = dataset['base_pose_ground_truth']
    contact_states = dataset['contact_states']
    joint_states = dataset['joint_states']
    base_states = dataset['base_states']
    max_episodes = params['max_episodes']
    print(f"[Worker {worker_id}] Starting experience collection.")
    
    best_return = float('-inf')
    best_return_episode = 0
    for episode in range(max_episodes):
        serow_env = SerowEnv(params['robot'], joint_states[0], base_states[0], contact_states[0])
        contact_frames = serow_env.contact_frames

        episode_return = 0.0
        # Run episode
        for time_step, (imu, joints, ft, gt, cs, next_cs) in enumerate(zip(
            imu_measurements[:max_steps],
            joint_measurements[:max_steps],
            force_torque_measurements[:max_steps],
            base_pose_ground_truth[:max_steps],
            contact_states[:max_steps],
            contact_states[1:max_steps+1]
        )):
            # Workers ALWAYS wait for update_event to be set by main process before each step.
            # This ensures they are paused during training phases.
            update_event.wait() 
            
            kin, prior_state = serow_env.predict_step(imu, joints, ft)

            for cf in contact_frames:
                if (prior_state.get_contact_position(cf) is not None
                        and cs.contacts_status[cf] and next_cs.contacts_status[cf]):

                    x = serow_env.compute_state(cf, prior_state, cs)
                    action, value, log_prob = agent_worker.get_action(x, deterministic=False) 
                    post_state, reward, done = serow_env.update_step(cf, kin, action, gt, time_step)
                    next_x = serow_env.compute_state(cf, post_state, next_cs)

                    if reward is not None:
                        try:
                            # Add experience to the shared buffer
                            shared_buffer.append((x, action, reward, next_x, done, value, log_prob))
                            episode_return = params['gamma'] * episode_return + reward
                        except Exception as e:
                            print(f"[Worker {worker_id}] Error appending to shared buffer: {str(e)}")
                            print(f"[Worker {worker_id}] Traceback: {traceback.format_exc()}")
                            # Optionally, you might want to break or handle this error differently
                            continue
                    else:
                        action = np.ones(serow_env.action_dim)
                        post_state, _, _ = serow_env.update_step(cf, kin, action, gt, time_step)

                    prior_state = post_state
                serow_env.finish_update(imu, kin)

            # Check if enough steps are collected for training (outside contact_frames loop)
            # This ensures the check and signal happens once per time_step, and only if needed.
            if len(shared_buffer) >= params['n_steps']:
                # First wait for update_event to be set before proceeding
                update_event.wait()
                # Then clear it to prevent other workers from proceeding
                update_event.clear()
                # Signal main process to start training
                training_event.set()
                # Wait for training to complete
                training_event.wait()
                # Load the updated policy
                actor_worker.load_state_dict(dict(shared_actor_state_dict))
                critic_worker.load_state_dict(dict(shared_critic_state_dict))
                # Set update_event to allow other workers to proceed
                update_event.set()

        # Print the reward accumulated so far
        print(f"[Worker {worker_id}] -[{episode}/{max_episodes - 1}] - [{time_step}/{max_steps - 1}] Current return: {episode_return} Best return: {best_return} at episode {best_return_episode}")
        
        # Send episode data to main process instead of logging directly
        episode_queue.put((worker_id, episode_return, time_step))
        
        # At the end of an episode, check for new best reward
        if episode_return > best_return:
            best_return = episode_return
            best_return_episode = episode
            print(f"[Worker {worker_id}] - [{episode}/{max_episodes}] New best return: {best_return} at episode {best_return_episode}")

def train_ppo_parallel(datasets, agent, params, max_steps):
    # Set to train mode
    agent.train()

    # Create a manager for shared data structures
    manager = multiprocessing.Manager()
    shared_buffer = manager.list()  # Use a managed list for the replay buffer
    shared_buffer._timeout = 5.0  # Add a 5-second timeout for operations

    # Use managed dictionaries for shared model state_dicts
    shared_actor_state_dict = manager.dict(agent.actor.state_dict())
    shared_critic_state_dict = manager.dict(agent.critic.state_dict())

    # Events for synchronization
    training_event = manager.Event() # Set by workers when buffer is full, cleared by main after training
    update_event = manager.Event()   # Set by main after model update, cleared by main before training

    # Create a queue for episode data
    episode_queue = manager.Queue()

    # Start workers for parallel data collection
    processes = []
    num_workers = len(datasets)
    print(f"[Main process] Starting {num_workers} worker processes...")
    for i, dataset in enumerate(datasets):
        p = multiprocessing.Process(
            target=collect_experience_worker,
            args=(i, dataset, params, shared_actor_state_dict, shared_critic_state_dict,
                  shared_buffer, training_event, update_event, agent.device, episode_queue, max_steps)
        )
        processes.append(p)
        p.start()
    print(f"[Main process] All {num_workers} worker processes started")

    # Initial signal to workers to start collecting
    update_event.set()

    # Check if at least one worker is still running
    while any(p.is_alive() for p in processes):
        try:
            # Check for episode data from workers
            try:
                while not episode_queue.empty():
                    worker_id, episode_return, time_step = episode_queue.get_nowait()
                    agent.logger.log_episode(episode_return, time_step)
            except Exception as e:
                print(f"[Main process] Error processing episode data: {str(e)}")
                print(f"[Main process] Traceback: {traceback.format_exc()}")

            # Main process waits until enough data is collected
            training_event.wait(timeout=5.0) # Wait for a worker to signal that buffer is full
            buffer_size = len(shared_buffer)
            
            if buffer_size < params['n_steps']:
                training_event.clear()
                update_event.set()
                continue

            # Clear training_event to prevent other workers from proceeding
            training_event.clear()

            # Safely get the collected experiences
            try:
                collected_experiences = list(shared_buffer)
                shared_buffer[:] = [] # Clear the shared buffer
            except Exception as e:
                print(f"[Main process] Error accessing shared buffer: {str(e)}")
                print(f"[Main process] Traceback: {traceback.format_exc()}")
                training_event.set()  # Allow workers to continue
                continue

            # Add collected experiences to the agent's internal buffer
            for exp in collected_experiences:
                agent.add_to_buffer(*exp) # Unpack the tuple

            actor_loss, critic_loss, _, converged = agent.train()

            # if actor_loss is not None and critic_loss is not None:
            #     print(f"[Main process] Policy Loss: {actor_loss:.4f}, Value Loss: {critic_loss:.4f}")

            # Update shared model parameters
            shared_actor_state_dict.update(agent.actor.state_dict())
            shared_critic_state_dict.update(agent.critic.state_dict())

            if converged:
                break

            # Signal workers to resume collection by setting training_event
            training_event.set()
        except Exception as e:
            print(f"[Main process] Error in training loop: {str(e)}")
            print(f"[Main process] Traceback: {traceback.format_exc()}")
            training_event.set()  # Allow workers to continue
            continue

    # Terminate worker processes
    print("[Main process] Training finished. Terminating worker processes.")
    for p in processes:
        p.terminate()
        p.join()

    # Load the best policy and export ONNX
    print("[Main process] Loading best policy and exporting to ONNX.")
    agent.load_checkpoint(os.path.join(agent.checkpoint_dir, f'trained_policy_{params["robot"]}.pth'))
    export_models_to_onnx(agent, params['robot'], params, agent.checkpoint_dir)

    # Plot training curves
    agent.logger.plot_training_curves()
    return agent


if __name__ == "__main__":
    
    # Read datasets from dataset folder
    datasets = []
    # get the path of the current file
    current_file = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_file))
    dataset_path = os.path.join(parent_dir, "datasets")
    # For each folder in the dataset path, read the data
    max_steps = 1e6
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            imu_measurements = read_imu_measurements(os.path.join(folder_path, "serow_measurements.mcap"))
            joint_measurements = read_joint_measurements(os.path.join(folder_path, "serow_measurements.mcap"))
            force_torque_measurements = read_force_torque_measurements(os.path.join(folder_path, "serow_measurements.mcap"))
            base_pose_ground_truth = read_base_pose_ground_truth(os.path.join(folder_path, "serow_measurements.mcap"))
            base_states = read_base_states(os.path.join(folder_path, "serow_proprioception.mcap"))
            contact_states = read_contact_states(os.path.join(folder_path, "serow_proprioception.mcap"))
            joint_states = read_joint_states(os.path.join(folder_path, "serow_proprioception.mcap"))
            dataset = {
                'imu': imu_measurements,
                'joints': joint_measurements,
                'ft': force_torque_measurements,
                'base_states': base_states,
                'contact_states': contact_states,
                'joint_states': joint_states,
                'base_pose_ground_truth': base_pose_ground_truth
            }
            datasets.append(dataset)
            max_steps = min(max_steps, len(imu_measurements) - 1)

    print(f"Found {len(datasets)} datasets")
    print(f"Max steps: {max_steps}")
    test_dataset = datasets[0]
    train_datasets = datasets[1:]

    # Compute the dt
    dt = []
    for i in range(len(test_dataset['imu']) - 1):
        dt.append(test_dataset['imu'][i+1].timestamp - test_dataset['imu'][i].timestamp)
    dt = np.median(np.array(dt))
    print(f"dt: {dt}")

    # Define the dimensions of your state and action spaces
    state_dim = 7
    action_dim = 1  # Based on the action vector used in ContactEKF.setAction()
    min_action = 1e-10
    robot = "go2"

    serow_env = SerowEnv(robot, test_dataset['joint_states'][0], test_dataset['base_states'][0], test_dataset['contact_states'][0])
    timestamps, base_positions, base_orientations, gt_positions, gt_orientations, cumulative_rewards = \
    serow_env.evaluate(test_dataset, agent=None) # agent is None here, so it's a base evaluation

    # Compute max and min state values (assuming these are constant across datasets)
    base_states = test_dataset['base_states']
    contact_states = test_dataset['contact_states']
    feet_positions = []
    base_linear_velocities = []
    for base_state in base_states:
        R_base = quaternion_to_rotation_matrix(base_state.base_orientation).transpose()
        base_linear_velocities.append(R_base @ base_state.base_linear_velocity)
        for cf in serow_env.contact_frames: 
            if base_state.contacts_position[cf] is not None:
                local_pos = R_base @ (base_state.base_position - base_state.contacts_position[cf])
                feet_positions.append(np.array([abs(local_pos[0]), abs(local_pos[1]), local_pos[2]]))

    # Convert base_linear_velocities and feet_positions to numpy array for easier manipulation
    base_linear_velocities = np.array(base_linear_velocities)
    feet_positions = np.array(feet_positions)
    contact_probabilities = []
    for contact_state in contact_states:
        for cf in serow_env.contact_frames: 
            if contact_state.contacts_status[cf]:
                contact_probabilities.append(contact_state.contacts_probability[cf])
    contact_probabilities = np.array(contact_probabilities)

    # Create max and min state values with correct dimensions
    print("[Main process] Calculating state dimensions...")
    print(f"[Main process] Feet positions shape: {feet_positions.shape}")
    print(f"[Main process] Base linear velocities shape: {base_linear_velocities.shape}")
    print(f"[Main process] Contact probabilities shape: {len(contact_probabilities)}")
    
    max_state_value = np.concatenate([np.max(feet_positions, axis=0),
                                      np.max(base_linear_velocities, axis=0),
                                      [np.max(contact_probabilities)]])
    min_state_value = np.concatenate([np.min(feet_positions, axis=0),
                                      np.min(base_linear_velocities, axis=0),
                                      [np.min(contact_probabilities)]])
    print(f"[Main process] RL state max values: {max_state_value}")
    print(f"[Main process] RL state min values: {min_state_value}")
    print(f"[Main process] State dimension: {len(max_state_value)}")

    params = {
        'robot': robot,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': None,
        'min_action': min_action,
        'clip_param': 0.2,  
        'value_clip_param': 0.2,
        'value_loss_coef': 0.2,  
        'entropy_coef': 0.01,  
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ppo_epochs': 5,  
        'batch_size': 64,  
        'max_grad_norm': 0.3,  
        'buffer_size': 10000,  
        'max_episodes': 50,
        'actor_lr': 1e-5, 
        'critic_lr': 1e-5,  
        'max_state_value': max_state_value,
        'min_state_value': min_state_value,
        'update_lr': True,
        'n_steps': 512,
        'convergence_threshold': 0.1,
        'critic_convergence_threshold': 0.1,
        'return_window_size': 15,
        'value_loss_window_size': 15,
        'checkpoint_dir': 'policy/ppo',
        'total_steps': 10000, 
        'final_lr_ratio': 0.01,  # Learning rate will decay to 1% of initial value
    }

    device = 'cpu' # For multiprocessing, it's often simpler to stick to CPU for shared memory,
    loaded = False
    print(f"[Main process] Initializing agent for {robot}")
    shared_network = SharedNetwork(params)
    actor = Actor(params, shared_network).to(device)
    critic = Critic(params, shared_network).to(device)
    agent = PPO(actor, critic, params, device=device, normalize_state=True)

    policy_path = params['checkpoint_dir']
    # Try to load a trained policy for this robot if it exists
    try:
        agent.load_checkpoint(f'{policy_path}/trained_policy_{robot}.pth')
        print(f"[Main process] Loaded trained policy for {robot} from '{policy_path}/trained_policy_{robot}.pth'")
        loaded = True
    except FileNotFoundError:
        print(f"[Main process] No trained policy found for {robot}. Training new policy...")

    if not loaded:
        # Train the policy using the parallelized function
        best_agent = train_ppo_parallel(train_datasets, agent, params, max_steps)
        # The best policy is already loaded and exported within train_ppo_parallel
        serow_env.evaluate(test_dataset, best_agent)
    else:
        # Just evaluate the loaded policy
        serow_env.evaluate(test_dataset, agent)
