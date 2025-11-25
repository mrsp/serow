import numpy as np
import gymnasium as gym
import copy
from utils import quaternion_to_rotation_matrix
import sys
import serow  # local import so unit tests can run without serow installed

# Orientation error
def quat_geodesic_angle_wxyz(q_est, q_gt):
    q_est = np.asarray(q_est, dtype=float); q_est /= max(1e-12, np.linalg.norm(q_est))
    q_gt  = np.asarray(q_gt,  dtype=float); q_gt  /= max(1e-12, np.linalg.norm(q_gt))
    dot = np.clip(abs(np.dot(q_est, q_gt)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)

# Scales a to a positive scale in [10^min_exp, 10^max_exp]
def action_to_scale(a, min_exp=-6.0, max_exp=-2.0):
    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)

    # Affine map from [-1,1] to [min_exp, max_exp]
    exp = (a + 1.0) * 0.5 * (max_exp - min_exp) + min_exp  # log10(scale)

    # scale = 10**exp (computed stably)
    scale = np.exp(exp.astype(np.float64) * np.log(10.0))

    if scale.ndim == 0:
        scale = float(scale)
        exp = float(exp)
    return scale


class SerowEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset, target_cf):
        super(SerowEnv,self).__init__()
        self.serow = serow.Serow()
        print("Initializing Serow Environment")
        
        self.target_cf = target_cf # which contact frame the environment will control
        # --- Load data from npz
        self.imu     = dataset["imu"]
        self.joints  = dataset["joints"]
        self.ft      = dataset["ft"]
        self.base_states = dataset["base_states"]
        self.joint_states = dataset["joint_states"] 
        self.pose_gt = dataset["base_pose_ground_truth"]  # assumes .position (3,), .orientation (wxyz)
        self.velocity_gt = dataset["base_velocity_ground_truth"]
        self.contact_status = dataset["contact_states"]
        self.contact_frames = list(self.contact_status[0].contacts_status.keys())
        
        self.prev_pos = None
        self.prev_ori = None
        self.starting_t = 0
        self.reward_history = []
        
        self.last_obs = None
        self.swing_phase = False
        print("Loaded dataset with IMU: ", len(self.imu), " samples", " Joints: ", len(self.joints), " samples", " FT: ", len(self.ft), " samples", " GT Pose: ", len(self.pose_gt), " samples", " Contact states: ", len(self.contact_status), " samples"  )
        
        self.cf_index = {name: i for i, name in enumerate(self.contact_frames)}
        
        self.pos_error_threshold = 0.8
        self.ori_error_threshold = 0.7

        self.action_history_size = 10 # How many past actions to include in the state
        self.state_dim = 9 + self.action_history_size 
          
        self.start_episode_time = 0 
        self.t = 0
        self.convergence_cycles = 100  # number of cycles to run the filter on reset until valid state
        self.max_start = len(dataset["imu"]) - self.convergence_cycles  # max index to start an episode (on reset) -> Hyperaparameter   

        # Extract kinematic measurements for every time step
        self.serow.initialize("go2_rl.json")  # Initialize SEROW instance
        self.initial_state = self.serow.get_state(allow_invalid=False) # Store initial state for reset
        self.imu_data = []
        self.kin_data = []
        for step_count in range (len(self.imu)):
            imu = copy.copy(self.imu[step_count])
            joint = copy.copy(self.joints[step_count])
            ft = copy.copy(self.ft[step_count])
            imu, kin, ft = self.serow.process_measurements(
                imu, joint, ft, None
            )
            self.imu_data.append(imu)
            self.kin_data.append(kin)    
        
        print("Serow Processed dataset with IMU: ", len(self.imu_data), " samples", " Kinematic Measurements: ", len(self.kin_data), " samples")
        
        self.N = len(self.contact_status) # Number of samples in the dataset

        # --- RL bits ---
        self.action_space = gym.spaces.Box(low=-1.0, high= 1.0, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        
        self.obs = np.zeros((self.state_dim,), dtype=np.float32)
        self.action_history_buffer = np.zeros((self.action_history_size,), dtype=np.float32) # buffer of floats

    def _compute_reward(self, state):
        # --- Estimated pose ---
        pos_est  = np.asarray(state.get_base_position(),    dtype=np.float64)
        quat_est = np.asarray(state.get_base_orientation(), dtype=np.float64)  # wxyz

        # --- Ground-truth pose ---
        pos_gt  = np.asarray(self.pose_gt[self.t].position,    dtype=np.float64)
        quat_gt = np.asarray(self.pose_gt[self.t].orientation, dtype=np.float64)  # wxyz

        # --- Errors ---
        pos_err = float(np.linalg.norm(pos_est[0:2] - pos_gt[0:2]) + 2 * np.linalg.norm(pos_est[2]-pos_gt[2]))                    # meters
        ori_err = float(quat_geodesic_angle_wxyz(quat_est, quat_gt))         # radians
                
        # Hyperparam for error computation (usually robot's height), 0.3m is roughly half the robot height
        L = 0.4 

        # Uncertainty-weighted SE(3) distance
        sigma_pos = 0.1  # estimated from sensor/filter
        sigma_ori = 0.3

        # SE(3) geodesic with proper scaling
        d_pos = pos_err / sigma_pos
        d_ori = (L * ori_err) / sigma_ori

        # Combined metric (Euclidean in normalized space)
        reward = -np.sqrt(d_pos**2 + d_ori**2)
        self.reward_history.append(reward)
        print(f"Step {self.t}: pos_err={pos_err:.2f} m, ori_err={ori_err:.2f} rad, reward={reward:.4f}")

        return float(reward), pos_err, ori_err
    
    def get_observation(self, cf, state, kin):
        # 1) Force of chosen contact frame (convert from Eigen/pybind to np)
        ft_t = self.ft[self.t]  # dict: {'FL_foot': ForceTorqueMeasurement, ...}
        meas = ft_t.get(cf, None)
        if meas is None:
            self.obs[0:3] = 0.0
        else:
            force = np.asarray(meas.force, dtype=np.float32).ravel()
            if force.shape != (3,):
                force = force[:3]
            self.obs[0:3] = force

        # 2) Innovation and NIS (use solve/Cholesky instead of explicit inverse)
        success, innovation, S = self.serow.get_contact_position_innovation(cf)

        if not success or innovation is None or S is None:
            self.obs[3:6] = 0.0
            self.obs[6] = 0.0
        else:
            v = np.asarray(innovation, dtype=np.float64).reshape(3,)
            S = np.asarray(S, dtype=np.float64).reshape(3, 3)
            # Symmetrize + tiny jitter for numerical robustness
            S = 0.5 * (S + S.T)
            S.flat[::4] += 1e-9

            # NIS = vᵀ S⁻¹ v, computed stably
            try:
                L = np.linalg.cholesky(S)
                y = np.linalg.solve(L, v)
                y = np.linalg.solve(L.T, y)
                nis = float(v @ y)
            except np.linalg.LinAlgError:
                y = np.linalg.solve(S, v)
                nis = float(v @ y)

            self.obs[3:6] = v.astype(np.float32)
            # log-scale NIS
            self.obs[6] = np.float32(np.log1p(nis))
            # self.obs[6] = np.float32(nis)

        # 3) Frobenius norm of 3×3 measurement covariance for this contact
        Rm = np.asarray(kin.contacts_position_noise[cf] + kin.position_cov, dtype=np.float64).reshape(3, 3)
        frob_R = np.linalg.norm(Rm, ord='fro')
        # Optionally log-scale:
        self.obs[7] = np.float32(frob_R)

        # 4) Action history (ensure bounds and dtype)
        n = self.action_history_size
        ah = np.asarray(self.action_history_buffer[-n:], dtype=np.float32)
        if ah.size < n:
            ah = np.pad(ah, (n - ah.size, 0), constant_values=0.0)
        self.obs[8:8+n] = ah

        return self.obs
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        print(f"\033[92mEpisode Finished started at {self.starting_t} and ended at {self.t} with duration {self.t-self.starting_t} timesteps\033[0m")

        max_valid_start = self.max_start - self.convergence_cycles

        # Sample a random t
        self.t = int(self.np_random.integers(0, max_valid_start - 1))

        # Re-sample until all the next 100 timesteps are in contact
        while not all(
            self.kin_data[self.t + i].contacts_status[self.target_cf]
            for i in range(self.convergence_cycles)
        ):
            self.t = int(self.np_random.integers(0, max_valid_start))
            
        self.starting_t = self.t
        self.start_episode_time = self.t # Store the initial point of the episode
        
        self.serow.reset()
        
        current_state = self.serow.get_state(allow_invalid=True)
        current_state.set_joint_state(self.joint_states[self.t])
        current_state.set_base_state(self.base_states[self.t])
        current_state.set_base_state_pose(self.pose_gt[self.t].position, self.pose_gt[self.t].orientation)
        current_state.set_base_state_velocity(self.velocity_gt[self.t].linear_velocity)
        current_state.set_contact_state(self.contact_status[self.t])
        current_state.set_base_pose_cov(np.eye(6) * 1e-5)
        
        self.serow.set_state(current_state)

            
        for i in range(self.starting_t, len(self.imu)):
            imu = self.imu[i]
            joint = self.joints[i]
            ft = self.ft[i]
            status = self.serow.filter(imu, joint, ft, None)
            if status:
                state = self.serow.get_state(allow_invalid=False)
                if (self.serow.is_state_valid()):
                    break
                
        
        
        reward, pos_err, ori_err = self._compute_reward(state)                  
        if (pos_err > 0.5) or ori_err > 0.4:
            print("ON RESET: Diverged position ", state.get_base_position() , " GT Position " , self.pose_gt[self.t].position, " at step ", self.t , " With errors ", pos_err , "  " , ori_err)
        else:
            print("SEROW converged...", state.get_base_position(), " ", state.get_base_orientation() ,
              " ", self.pose_gt[self.t].position, " ", self.pose_gt[self.t].orientation)
        
        # Get observation at time t
        self.obs = self.get_observation(self.target_cf, current_state, self.kin_data[self.t])
        
        self.prev_ori = None
        self.prev_pos = None
        info = {}
        self.action_history_buffer.fill(0.0)
        return self.obs, info



    def step(self, action):
        diverged = False
        truncated  = False
        terminated = False
        pos_err = 0.0
        reward = 0.0
        self.t += 1
        if (self.t >= self.N-2):
            truncated = True
            return self.obs, float(reward), terminated, truncated, {}

        if self.kin_data[self.t].contacts_status[self.target_cf] == False:
            # truncated = True
            reward = 0.0
            if self.last_obs is not None:
                self.obs = self.last_obs.copy()
            self.swing_phase = True
        else:
            if (self.swing_phase):
                current_state = self.serow.get_state(allow_invalid=True)
                current_state.set_joint_state(self.joint_states[self.t])
                current_state.set_base_state(self.base_states[self.t])
                current_state.set_base_state_pose(self.pose_gt[self.t].position, self.pose_gt[self.t].orientation)
                current_state.set_base_state_velocity(self.velocity_gt[self.t].linear_velocity)
                current_state.set_contact_state(self.contact_status[self.t])
                self.serow.set_state(current_state)
                    
            self.swing_phase = False
            
            self.serow.base_estimator_predict_step(self.imu_data[self.t], self.kin_data[self.t])
            
            state = self.serow.get_state(allow_invalid=False)
           
            if self.prev_pos is None or self.prev_ori is None:
                self.prev_pos = state.get_base_position()
                self.prev_ori = state.get_base_orientation()
                
            action = np.asarray(action, dtype=np.float32)
            
            # map action in [-1,1] -> positive 'scale' and apply it
            scaled_action = action_to_scale(action).astype(np.float64)  
    
            self.serow.set_action(self.target_cf, scaled_action)
            self.serow.base_estimator_update_with_contact_position(self.target_cf, self.kin_data[self.t])      
            
            for cf in self.contact_frames:
                if cf == self.target_cf:
                    continue
                self.serow.set_action(cf, np.array([1.0]))

                # Run the update step with the contact position
                self.serow.base_estimator_update_with_contact_position(cf, self.kin_data[self.t])

                # Get the state
                state = self.serow.get_state(allow_invalid=True)
                             
            self.serow.base_estimator_finish_update(self.imu_data[self.t], self.kin_data[self.t])

            # update history buffer
            self.action_history_buffer = np.roll(self.action_history_buffer, -1)
            self.action_history_buffer[-1] = action[0]
            
            state = self.serow.get_state(allow_invalid=False)
            

            self.current_action = scaled_action

            reward, pos_err, ori_err = self._compute_reward(state) 
                
            

            self.obs = self.get_observation(self.target_cf, state, self.kin_data[self.t])
                        
            if (pos_err > self.pos_error_threshold) or ori_err > self.ori_error_threshold:
                    diverged = True
            else:
                self.prev_pos = state.get_base_position()
                self.prev_ori = state.get_base_orientation()

            terminated = bool(diverged)
            if (terminated):
                print("Diverged position ", state.get_base_position() , " GT Position " , self.pose_gt[self.t].position, " at step ", self.t)

                print(f"\033[91mFilter Diverged with pos err {pos_err} at step {self.t} and lasted {self.t - self.start_episode_time} timesteps\033[0m")
                # self.obs = np.zeros(self.state_dim, dtype=np.float32)

            if truncated:
                print(f"\033[91mEpisode Trancated\033[0m")
        

        # 7) Info dict for logging (helps tuning but not used by the policy)
        info = {
            "t": int(self.t),
            "diverged": bool(diverged),
            "terminated": bool(terminated),
            "pos_err": float(pos_err)#, "ori_err": float(ori_err),
        }
        self.last_obs = self.obs.copy()
        # 8) Return (Gymnasium API)
        return self.obs, float(reward), terminated, truncated, info
