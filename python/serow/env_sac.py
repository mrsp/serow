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
def action_to_scale(a, min_exp=-7.0, max_exp=-1.0):
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

    def __init__(self, dataset, target_cf, w_pos=1.0, w_ori=0.5):
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
        print("Loaded dataset with IMU: ", len(self.imu), " samples", " Joints: ", len(self.joints), " samples", " FT: ", len(self.ft), " samples", " GT Pose: ", len(self.pose_gt), " samples", " Contact states: ", len(self.contact_status), " samples"  )
        
        self.cf_index = {name: i for i, name in enumerate(self.contact_frames)}
        
        
        # --- Hyperparameters ---
        self.action_history_size = 10 # How many past actions to include in the state
        self.state_dim = 9 + self.action_history_size 
          
        self.start_episode_time = 0 
        self.max_start = 39000 # max index to start an episode (on reset)    

        # Extract kinematic measurements for every time step
        self.serow.initialize("go2_rl.json")  # Initialize SEROW instance
        self.initial_state = self.serow.get_state(allow_invalid=True) # Store initial state for reset
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
        # Total obs = concat over contacts
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        
        # --- reward weights ---
        self.w_pos = float(w_pos)
        self.w_ori = float(w_ori)

        self.obs = np.zeros((self.state_dim,), dtype=np.float32)
        self.action_history_buffer = np.zeros((self.action_history_size,), dtype=np.float32) # buffer of floats
        # --- step bookkeeping ---
        self.t = 0
    
    def _update_all_contacts_if_available(self, kin):
        """
        Minimal, no-branch update: if kin exposes contact flags, update those; else skip.
        This keeps things robust if your log sometimes lacks contacts.
        """
        if hasattr(kin, "contacts_status"):
            for cf, is_on in kin.contacts_status.items():
                if is_on:
                    self.serow.base_estimator_update_with_contact_position(cf, kin)



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
        # Base term: linear penalty (SAC-friendly)
        reward = -(self.w_pos * pos_err + self.w_ori * ori_err)
        
        self.reward_history.append(reward)


        if reward < -50.0:    
                
            print("ERRORS --> " , pos_err, "  " ,  ori_err, "  ", reward)
            print("ACTION --> " , self.current_action)
            print("EST POSE --> " , pos_est , "  " , quat_est)
            print("Prev Estimated pose --> " , self.prev_pos , "  " , self.prev_ori)
            print("GT  POSE --> " , pos_gt  , "  " , quat_gt)
            print(" FT MEAS --> " , self.ft[self.t]["FL_foot"].force)
            print(" IMU MEAS --> " , self.imu[self.t].linear_acceleration)
            print(" Angular vel --> " , self.imu[self.t].angular_velocity)

            sys.exit()
        return float(reward), pos_err
    
    def frobenius_norm(R: np.ndarray) -> float:
        R = np.asarray(R, dtype=float).reshape(3, 3)
        return np.linalg.norm(R, ord='fro')
    
    def get_observation(self, cf, state, kin):
        # 1) Force of chosen contact frame (convert from Eigen/pybind to np)
        ft_t = self.ft[self.t]  # dict: {'FL_foot': ForceTorqueMeasurement, ...}
        meas = ft_t.get(cf, None)
        if meas is None:
            self.obs[0:3] = 0.0
        else:
            force = np.asarray(meas.force, dtype=np.float32).ravel()
            if force.shape != (3,):
                force = force[:3]  # defensive
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
            # Optionally log-scale; NIS can be heavy-tailed:
            # self.obs[6] = np.float32(np.log1p(nis))
            self.obs[6] = np.float32(nis)

        # 3) Frobenius norm of 3×3 measurement covariance for this contact
        #    (don’t flatten before the norm)
        Rm = np.asarray(kin.contacts_position_noise[cf] + kin.position_cov, dtype=np.float64).reshape(3, 3)
        frob_R = np.linalg.norm(Rm, ord='fro')  # sqrt(sum of squares)
        # Optionally log-scale:
        # frob_R = np.log1p(frob_R)
        self.obs[7] = np.float32(frob_R)

        # 4) Action history (ensure bounds and dtype)
        n = self.action_history_size
        ah = np.asarray(self.action_history_buffer[-n:], dtype=np.float32)
        if ah.size < n:
            ah = np.pad(ah, (n - ah.size, 0), constant_values=0.0)
        self.obs[8:8+n] = ah

        return self.obs
    
    def reset(self, *, seed=None, options=None):
        # if seed is not None:
        super().reset(seed=seed)

        print(f"\033[92mEpisode Finished started at {self.starting_t} and ended at {self.t} with duration {self.t-self.starting_t} timesteps\033[0m")
        self.t = int(self.np_random.integers(0, self.max_start + 1)) # Get a new random start point
        # Make sure we start at a point where the target contact frame is in contact
        while (self.kin_data[self.t].contacts_status[self.target_cf] == False
               and self.kin_data[self.t + 1].contacts_status[self.target_cf] == False ):
            self.t = int(self.np_random.integers(0, self.max_start + 1)) # Get a new random start point
        self.starting_t = self.t
        self.start_episode_time = self.t # Store the initial point of the episode
        
        # initial_state.set_base_state_pose(self.pose_gt[self.t].position, self.pose_gt[self.t].orientation)
                
        # curr_state = self.serow.get_state(allow_invalid=True)
        self.serow.reset()
        self.initial_state = self.serow.get_state(allow_invalid=True) # Store initial state for reset
        self.initial_state.set_joint_state(self.joint_states[self.t])
        self.initial_state.set_base_state(self.base_states[self.t])
        self.initial_state.set_base_state_pose(self.pose_gt[self.t].position, self.pose_gt[self.t].orientation)
        # self.initial_state.set_base_state_velocity(self.velocity_gt[self.t].linear_velocity)
        self.initial_state.set_contact_state(self.contact_status[self.t])
        self.serow.set_state(self.initial_state)
        
        kin_next = self.kin_data[self.t + 1]  # measurements aligned with the new state index
        self.obs = self.get_observation(self.target_cf, self.initial_state, kin_next)

        # print(curr_state.get_base_position() , " " , self.pose_gt[self.t].position)
        # print(curr_state.get_base_orientation() , " " , self.pose_gt[self.t].orientation)
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
        if self.kin_data[self.t].contacts_status[self.target_cf] == False:
            truncated = True
            reward = 0.0
            obs = np.zeros(self.state_dim, dtype=np.float32)
        else:
            state = self.serow.get_state(allow_invalid=True)
            print("Current position ", state.get_base_position() , " GT Position " , self.pose_gt[self.t].position, " at step ", self.t)
            # debug_pos_error =float(np.linalg.norm(state.get_base_position() - self.pose_gt[self.t].position))
            # print("pos error" , debug_pos_error)

            if self.prev_pos is None or self.prev_ori is None:
                self.prev_pos = state.get_base_position()
                self.prev_ori = state.get_base_orientation()
            action = np.asarray(action, dtype=np.float32)
            
            kin = self.kin_data[self.t] # Extract current kinematic measurement

            # 1) map action in [-1,1] -> positive 'scale' and apply it
            scaled_action = action_to_scale(action).astype(np.float64)  # shape (4,)
          
    
            self.serow.set_action(self.target_cf, scaled_action)
            self.serow.base_estimator_update_with_contact_position(self.target_cf, kin)       
            self.serow.base_estimator_finish_update(self.imu[self.t], kin)


            state = self.serow.get_state(allow_invalid=True)
            

            self.current_action = scaled_action

            reward, pos_err = self._compute_reward(state) 
                
            kin_next = self.kin_data[self.t]  # measurements aligned with the new state index

            self.obs = self.get_observation(self.target_cf, state, kin_next)
                        
            if (pos_err > 0.2):
                    diverged = True
            else:
                self.prev_pos = state.get_base_position()
                self.prev_ori = state.get_base_orientation()

            # 4) Time limit / dataset end
            time_limit_reached = (self.t >= self.N - 1)

            # 5) If terminating now, you can zero obs (optional), otherwise keep obs
            terminated = bool(diverged)
            truncated  = bool(time_limit_reached)
            if (terminated):
                print("Diverged position ", state.get_base_position() , " GT Position " , self.pose_gt[self.t].position, " at step ", self.t)

                print(f"\033[91mFilter Diverged with pos err {pos_err} at step {self.t} and lasted {self.t - self.start_episode_time} timesteps\033[0m")
                # self.obs = np.zeros(self.state_dim, dtype=np.float32)

            if truncated:
                print(f"\033[91mEpisode Trancated\033[0m")
        
        # 6) Bookkeeping: move current -> last action AFTER computing reward/obs

        # 7) Info dict for logging (helps tuning but not used by the policy)
        info = {
            "t": int(self.t),
            "diverged": bool(diverged),
            "terminated": bool(terminated),
            "pos_err": float(pos_err)#, "ori_err": float(ori_err),
        }
        self.t += 1

        # 8) Return (Gymnasium API)
        return self.obs, float(reward), terminated, truncated, info
