# import random
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from scipy.integrate import solve_ivp
# from tqdm.notebook import tqdm
# from road_generator import *
# import config
# from model import *
# from trainer import *

# np.random.seed(config.seed)
# torch.manual_seed(config.seed)

# """ ODE Systems parameters """
# m1 = config.m1
# m2 = config.m2 
# m = config.m
# cb = config.cb
# kb = config.kb
# kw = config.kw
# dt = config.dt
# TIME = config.TIME

# """--------------------------------------------------------------
#     ODE (Ordinary Differential Equation) of Quarter Car model
#     :t  - timesteps
#     :y0 - initial state [xb, xw, d/dt(xb), d/dt(xw)]
#     :m  - tuple containing (m_body, m_wheel)
#     :cs - constant related to ...
#     :ks - spring stiffness
#     :kw - tire stiffness

#     :return [d/dt(xb), d/dt(xw), d2/dt(xb), d2/dt(xw)]
# --------------------------------------------------------------"""
# road_profile = list(RoadProfile().get_profile_by_class("E", config.t_stop, config.dt)[1][1:])

# def odefun(t, y0, m, cs, kw, ks):
#     m1=m[0];    # Body mass in kg
#     m2=m[1];    # Wheel mass in kg

#     """ Road condition at time step t"""
#     t_idx = min(round(t/dt), len(TIME)-1)
#     xr = road_profile[t_idx]
#     # print(t)

#     """ 1. Displacement & Velocity of body & wheel """
#     xb = y0[0]                # x_body        (x1)
#     xw = y0[1]                # x_wheel       (x2)
#     dxb = y0[2]               # d(x_body)/dt  (dx1/dt)
#     dxw = y0[3]               # d(x_wheel)/dt (dx2/dt)
#     """ 2. Acceleration of body & wheel """
#     d2xb = - ks/m1*(xb-xw) - cs/m1*(dxb-dxw) # + kw/m1*xr - kw/m1*xb
#     d2xw = ks/m2*(xb-xw) + cs/m2*(dxb-dxw) + kw/m2*(xr-xw)

#     return [dxb,dxw,d2xb,d2xw]

# """**********************************************************
#     Replay buffer class to enrich past experience
#     ->  avoid biased toward recent experience
#         (hence forgot past interactions / rare situations)
# **********************************************************"""
# class replayBuffer:
#     def __init__(self, buffer_size: int):
#         self.buffer_size = buffer_size
#         self.buffer = []
#         self._next_idx = 0

#     def add(self, item):
#         if len(self.buffer) > self._next_idx:
#             self.buffer[self._next_idx] = item
#         else:
#             self.buffer.append(item)
#         if self._next_idx == self.buffer_size - 1:
#             self._next_idx = 0
#         else:
#             self._next_idx = self._next_idx + 1

#     def sample(self, batch_size):
#         indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
#         states   = [self.buffer[i][0] for i in indices]
#         actions  = [self.buffer[i][1] for i in indices]
#         rewards  = [self.buffer[i][2] for i in indices]
#         n_states = [self.buffer[i][3] for i in indices]
#         dones    = [self.buffer[i][4] for i in indices]
#         return states, actions, rewards, n_states, dones

#     def length(self):
#         return len(self.buffer)

# """**********************************************************
#     Ornstein-Uhlenbeck noise implemented by OpenAI
#     Copied from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# **********************************************************"""
# class OrnsteinUhlenbeckActionNoise:
#     def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
#         self.theta = theta
#         self.mu = mu
#         self.sigma = sigma
#         self.dt = dt
#         self.x0 = x0
#         self.reset()

#     def __call__(self):
#         x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
#         self.x_prev = x
#         return x

#     def reset(self):
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

# """**********************************************************
#     Map Actor Network's output to Cs (System's Damping) action space
#     Range: [-600, +600] N
# **********************************************************"""
# def get_cs(a):
#     ca = 600*a
#     return cb + ca

# """**********************************************************
#     Map Actor Network's output to Ks (System's Stiffness) action space
#     Range: [-2500, +5000] N
# **********************************************************"""
# def get_ks(a):
#     # positive and negative stiffness has different range
#     ka = 5000*a if a > 0 else 2500*a
#     # ka = 4000*a if a > 0 else -4500*a
#     return kb + ka

# """**********************************************************
#     Execute training process
# **********************************************************"""
# def execute_training(ckpt_Q_origin, ckpt_mu_origin):
#     """ Initialize Trainer (model weights & optimizers)"""
#     trainer = Trainer(ckpt_Q_origin=ckpt_Q_origin, 
#                       ckpt_mu_origin=ckpt_mu_origin)
                      
#     """ Replay buffer """
#     # buffer = replayBuffer(buffer_size=1000000)
#     buffer = replayBuffer(buffer_size=100000)

#     """ Training history"""
#     reward_records = []
#     best_score = -99999
#     rl_cs = []
#     rl_ks = []

#     """ Episode Iteration"""
#     for episode in tqdm(range(config.num_episodes)):
#         """--------------------------------------------------------------
#             1. Randomly Generate new road profile each episode
#         --------------------------------------------------------------"""
#         global road_profile
#         # road_profile = generate_road(TIME, MAX_BUMP)
#         road_profile = list(RoadProfile().get_profile_by_class("E", config.t_stop, config.dt)[1][1:])
        
#         """--------------------------------------------------------------
#                 Exploration factor 
#         --------------------------------------------------------------"""
#         output_dim = config.output_dim

#         if episode < 100:
#             ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.5)
#         elif episode < 200:
#             ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.1)
#         else:
#             ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.05)
        
#         """--------------------------------------------------------------
#             2. Initialize the first state of the quarter car model & RL model
#                 * state_ODE \in R^4: [x_body, x_wheel, d/dt(x_body), d/dt(x_wheel)]
#                 * state_RL  \in R^5: [d/dt(x_body), d/dt(x_wheel), d/dt(road_profile)]
#         --------------------------------------------------------------"""
#         s_ode = [road_profile[0], road_profile[0], 0, 0]  # [xb, xw, dxb, dxw] e R4 -> state representation of the quarter car model
#         s_ode_prev = s_ode
#         s_rl = None # [xb, xw, dxb, dxw, d2xb] e R5  -> state representation of the RL model
#         done = False
#         total_reward = 0
#         temp_xb_time = []
#         temp_dxb_time = []
        
#         """--------------------------------------------------------------
#                 3. Each Episode optimization
#         --------------------------------------------------------------"""
#         for t_idx, t in enumerate(config.TIME):
#             """####################################
#                 I. State & Action: S{t} & A{t} COMPUTATION
#                     (current State & Action)
#             ####################################"""
#             """ 1.1. Get current state s{t} from ODE """
#             xb, xw, dxb, dxw = np.array(s_ode)[0:4]
#             dxr = (road_profile[t_idx] - road_profile[t_idx-1]) / dt
#             xb_prev, xw_prev, dxb_prev, dxw_prev = np.array(s_ode_prev)[0:4]
#             dxr_prev = (road_profile[max(t_idx-1, 0)] - road_profile[max(t_idx-2, 0)]) / dt
#             s_rl = [dxb, dxw, dxr, dxb_prev, dxw_prev, dxr_prev] # rescale
#             """ 1.2. Take action a{t} from RL's s{t}"""
#             a = trainer.pick_sample(s_rl, ou_action_noise)
#             a_cs, a_ks = a
#             """ 1.3. Compute d2/dt (x_body)"""
#             d2xb = (-get_cs(a_cs) * (dxb-dxw) - get_ks(a_ks) * (xb-xw)) / m1
            
#             """ 1.4. Record curves"""
#             temp_xb_time.append(xb)
#             temp_dxb_time.append(dxb)
#             rl_cs.append(get_cs(a_cs))
#             rl_ks.append(get_ks(a_ks))
            
            
#             """####################################
#                 II. S{t+1} TRANSITION (from S{t} & A{t})
#             ####################################"""
#             # s_next, r, done, _ = env.step(a)
#             """ 2.1. Solve ODE for s{t+1} """
#             # del xb, xw, dxb, dxw # delete these variables for debugging purpose
#             yout = solve_ivp(odefun, [t, t+dt], s_ode, args=(m, get_cs(a_cs), kw, get_ks(a_ks)), dense_output=True)
#             # yout.y is the solution of the ODE -> s{t+1}
#             xb_next, xw_next, dxb_next, dxw_next = yout.y[:,-1]
#             if t_idx < len(TIME)-1:
#                 dxr_next = (road_profile[t_idx+1] - road_profile[t_idx]) / dt
#             """ 2.2. Record next states"""
#             s_ode_next = [xb_next, xw_next, dxb_next, dxw_next]
#             # s_rl_next = [xb, xw, dxb]
#             # s_rl_next = [xb, xw, dxb, dxw, dxr]
#             # s_rl_next = [dxb_next, dxw_next, dxr_next]
#             s_rl_next = [dxb_next, dxw_next, dxr_next, dxb, dxw, dxr]
#             """ 2.3. Select a_next"""
#             ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.0)
#             a_next = trainer.pick_sample(s_rl_next, ou_action_noise)
#             a_cs_next, a_ks_next = a_next
#             d2xb_next = (-get_cs(a_cs_next) * (dxb_next-dxw_next) - get_ks(a_ks_next) * (xb_next-xw_next)) / m1
#             """ 2.4. Check final state"""
#             done = False if (t_idx < len(TIME)-1) else True
            
            
#             """####################################
#                 III. REWARDS, BUFFER, & MODEL UPDATES
#             ####################################"""
#             """ 3.1. Reward function"""
#             # r = -0.7*abs(xb) - 0.3*(dxb)**2
#             # r = -0.9*abs(xb) - 0.1*abs(dxb) # - abs(xb-xw)
#             # r = -0.1*abs(dxb) # -> TRAINED 1 USING THIS (with buffer size 1M)
#             r = -1/10*abs(dxb_next) # -> TRAINED 2,3,4 USING THIS (with buffer size 100K)
#             # r = -1/10*abs(dxb_next) - 1/100*abs(d2xb_next) # -> TRAINED 2,3,4 USING THIS (with buffer size 100K)
#             # r = -1/10*abs(xb) - 0.25e2*(abs(xb)**3)
#             # r = -10*abs(d2xb)
#             # r = -0.25e2*(abs(xb)**3)
#             # r = -1000*(abs(xb)**3)
#             # r = -1*(0.01*(abs(d2xb)**2))
#             # r = (-1/10*(abs(xb))) - (0.001*(abs(d2xb)**2))
#             # print(r)
            
#             total_reward += r
            
#             """ 3.2. buffer experience"""
#             buffer.add([s_rl, a, r, s_rl_next, float(done)])
            
#             """ 3.3. Update model based on buffered experience (st, at, rt, s{t+1}, done) """
#             if buffer.length() >= config.bs:
#                 states, actions, rewards, n_states, dones = buffer.sample(config.bs)
#                 trainer.optimize(states, actions, rewards, n_states, dones)
#                 trainer.update_target()

#             """####################################
#                 IV. STATE TRANSITION
#             ####################################"""
#             s_ode_prev = s_ode
#             s_ode = s_ode_next

#         # Output total rewards in episode (max 500)
#         print(f"Run episode {episode} with rewards {total_reward}")
#         reward_records.append(total_reward)
#         np.save('Reward.npy', np.array(reward_records))

#         if best_score < total_reward:
#             best_score = total_reward
#             print(f'New best score: {best_score}')
#             rl_xb_time = temp_xb_time
#             rl_dxb_time = temp_dxb_time
#             trainer.save_checkpoints()

#         # if episode % 200 == 0 and episode > 0:
#         #     trainer.reduce_lr()
#         #     print('  reduce lr.')

# if __name__ == '__main__':
#     save_dir = './checkpoints'
#     ckpt_Q_origin = f'{save_dir}/Q_origin.pt'
#     ckpt_mu_origin = f'{save_dir}/mu_origin.pt'
#     print('Start Training Process...')

#     execute_training(ckpt_Q_origin, ckpt_mu_origin)



################ver 2#######################
############################################

# import random
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from scipy.integrate import solve_ivp
# from tqdm import tqdm
# from road_generator import *
# import config
# from model import *
# from trainer import *

# np.random.seed(config.seed)
# torch.manual_seed(config.seed)

# """──────────────────────────────────────────────────────
#     ALGORITHM SELECTION — Change this one variable only
    
#     ALGORITHM = 'DDPG'  → original single-critic training
#     ALGORITHM = 'TD3'   → new twin-critic training
    
#     Everything else (road, ODE, buffer, reward) stays identical
#     so comparison between DDPG and TD3 is completely fair
# ──────────────────────────────────────────────────────"""
# ALGORITHM = 'TD3'   # <-- change to 'DDPG' to run original

# """──────────────────────────────────────────────────────
#     ODE System Parameters (from config)
# ──────────────────────────────────────────────────────"""
# m1   = config.m1
# m2   = config.m2
# m    = config.m
# cb   = config.cb
# kb   = config.kb
# kw   = config.kw
# dt   = config.dt
# TIME = config.TIME

# """──────────────────────────────────────────────────────
#     ODE of Quarter Car Model — UNCHANGED from original
    
#     Inputs:
#         t   → current time
#         y0  → [xb, xw, dxb, dxw] current state
#         m   → [m1, m2] masses
#         cs  → total damping  (cb + ca from RL)
#         kw  → tyre stiffness (fixed)
#         ks  → total stiffness (kb + ka from RL)
    
#     Returns: [dxb, dxw, d2xb, d2xw]
# ──────────────────────────────────────────────────────"""
# road_profile = list(
#     RoadProfile().get_profile_by_class("E", config.t_stop, config.dt)[1][1:]
# )

# def odefun(t, y0, m, cs, kw, ks):
#     m1 = m[0]
#     m2 = m[1]

#     t_idx = min(round(t/dt), len(TIME)-1)
#     xr    = road_profile[t_idx]

#     xb  = y0[0];  xw  = y0[1]
#     dxb = y0[2];  dxw = y0[3]

#     d2xb = -ks/m1*(xb-xw) - cs/m1*(dxb-dxw)
#     d2xw =  ks/m2*(xb-xw) + cs/m2*(dxb-dxw) + kw/m2*(xr-xw)

#     return [dxb, dxw, d2xb, d2xw]


# """──────────────────────────────────────────────────────
#     Replay Buffer — UNCHANGED from original
# ──────────────────────────────────────────────────────"""
# class replayBuffer:
#     def __init__(self, buffer_size: int):
#         self.buffer_size = buffer_size
#         self.buffer      = []
#         self._next_idx   = 0

#     def add(self, item):
#         if len(self.buffer) > self._next_idx:
#             self.buffer[self._next_idx] = item
#         else:
#             self.buffer.append(item)
#         self._next_idx = 0 if self._next_idx == self.buffer_size-1 else self._next_idx+1

#     def sample(self, batch_size):
#         indices  = [random.randint(0, len(self.buffer)-1) for _ in range(batch_size)]
#         states   = [self.buffer[i][0] for i in indices]
#         actions  = [self.buffer[i][1] for i in indices]
#         rewards  = [self.buffer[i][2] for i in indices]
#         n_states = [self.buffer[i][3] for i in indices]
#         dones    = [self.buffer[i][4] for i in indices]
#         return states, actions, rewards, n_states, dones

#     def length(self):
#         return len(self.buffer)


# """──────────────────────────────────────────────────────
#     OU Noise — UNCHANGED from original
# ──────────────────────────────────────────────────────"""
# class OrnsteinUhlenbeckActionNoise:
#     def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
#         self.theta = theta;  self.mu = mu
#         self.sigma = sigma;  self.dt = dt
#         self.x0    = x0;     self.reset()

#     def __call__(self):
#         x = (self.x_prev
#              + self.theta * (self.mu - self.x_prev) * self.dt
#              + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape))
#         self.x_prev = x
#         return x

#     def reset(self):
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# """──────────────────────────────────────────────────────
#     Action Scaling — UNCHANGED from original
# ──────────────────────────────────────────────────────"""
# def get_cs(a):
#     return cb + 600*a

# def get_ks(a):
#     ka = 5000*a if a > 0 else 2500*a
#     return kb + ka


# """──────────────────────────────────────────────────────
#     MAIN TRAINING FUNCTION
    
#     Only difference from original:
#         - DDPG → uses DDPGTrainer, single checkpoint set
#         - TD3  → uses TD3Trainer,  two Q checkpoints
    
#     Inner loop (ODE, buffer, reward) is byte-for-byte identical
#     so all results are directly comparable
# ──────────────────────────────────────────────────────"""
# def execute_training(save_dir='./checkpoints'):

#     # ── Initialise correct trainer based on ALGORITHM flag ──
#     if ALGORITHM == 'DDPG':
#         print('=' * 50)
#         print('  Running: DDPG (original single-critic)')
#         print('=' * 50)
#         trainer = DDPGTrainer(
#             ckpt_Q_origin  = f'{save_dir}/DDPG_Q_origin.pt',
#             ckpt_mu_origin = f'{save_dir}/DDPG_mu_origin.pt'
#         )

#     elif ALGORITHM == 'TD3':
#         print('=' * 50)
#         print('  Running: TD3 (twin-critic, delayed actor)')
#         print('=' * 50)
#         trainer = TD3Trainer(
#             ckpt_Q1_origin = f'{save_dir}/TD3_Q1_origin.pt',
#             ckpt_Q2_origin = f'{save_dir}/TD3_Q2_origin.pt',
#             ckpt_mu_origin = f'{save_dir}/TD3_mu_origin.pt',
#             actor_update_freq = 2,    # Actor updates every 2 critic steps
#             target_noise      = 0.2,  # noise added to target actions
#             target_noise_clip = 0.5   # noise clipped to this range
#         )
#     else:
#         raise ValueError(f"Unknown algorithm: {ALGORITHM}. Choose 'DDPG' or 'TD3'.")

#     # ── Buffer and tracking variables ───────────────────────
#     buffer         = replayBuffer(buffer_size=100000)
#     reward_records = []
#     best_score     = -99999

#     # ── Episode Loop ────────────────────────────────────────
#     for episode in tqdm(range(config.num_episodes)):

#         # 1. New random Class E road every episode (data augmentation)
#         global road_profile
#         road_profile = list(
#             RoadProfile().get_profile_by_class("E", config.t_stop, config.dt)[1][1:]
#         )

#         # 2. Exploration noise schedule — decays as agent learns
#         output_dim = config.output_dim
#         if episode < 100:
#             sigma = 0.5     # wide exploration early
#         elif episode < 200:
#             sigma = 0.1     # medium exploration
#         else:
#             sigma = 0.05    # narrow exploration (mostly exploit)
#         ou_action_noise = OrnsteinUhlenbeckActionNoise(
#             mu=np.zeros(output_dim), sigma=np.ones(output_dim)*sigma
#         )

#         # 3. Initialise car at rest
#         s_ode      = [road_profile[0], road_profile[0], 0, 0]
#         s_ode_prev = s_ode
#         total_reward = 0
#         temp_xb_time  = []
#         temp_dxb_time = []

#         # ── Timestep Loop (1000 steps = 10 seconds) ─────────
#         for t_idx, t in enumerate(config.TIME):

#             # ── Phase I: State and Action ────────────────────
#             xb, xw, dxb, dxw = np.array(s_ode)[0:4]
#             dxr      = (road_profile[t_idx] - road_profile[t_idx-1]) / dt
#             xb_prev, xw_prev, dxb_prev, dxw_prev = np.array(s_ode_prev)[0:4]
#             dxr_prev = (road_profile[max(t_idx-1,0)] - road_profile[max(t_idx-2,0)]) / dt

#             s_rl = [dxb, dxw, dxr, dxb_prev, dxw_prev, dxr_prev]

#             a       = trainer.pick_sample(s_rl, ou_action_noise)
#             a_cs, a_ks = a

#             temp_xb_time.append(xb)
#             temp_dxb_time.append(dxb)

#             # ── Phase II: Physics simulation ─────────────────
#             yout = solve_ivp(
#                 odefun, [t, t+dt], s_ode,
#                 args=(m, get_cs(a_cs), kw, get_ks(a_ks)),
#                 dense_output=True
#             )
#             xb_next, xw_next, dxb_next, dxw_next = yout.y[:,-1]

#             if t_idx < len(TIME)-1:
#                 dxr_next = (road_profile[t_idx+1] - road_profile[t_idx]) / dt

#             s_ode_next = [xb_next, xw_next, dxb_next, dxw_next]
#             s_rl_next  = [dxb_next, dxw_next, dxr_next, dxb, dxw, dxr]

#             # ── Phase III: Reward, Buffer, Train ─────────────
#             # Reward function — IDENTICAL for DDPG and TD3
#             # This ensures fair comparison: same reward signal, different algorithm
#             r = -1/10 * abs(dxb_next)
#             total_reward += r

#             done = False if (t_idx < len(TIME)-1) else True
#             buffer.add([s_rl, a, r, s_rl_next, float(done)])

#             if buffer.length() >= config.bs:
#                 states, actions, rewards, n_states, dones = buffer.sample(config.bs)
#                 trainer.optimize(states, actions, rewards, n_states, dones)
#                 trainer.update_target()

#             # ── Phase IV: State transition ───────────────────
#             s_ode_prev = s_ode
#             s_ode      = s_ode_next

#         # ── Episode summary ──────────────────────────────────
#         print(f'[{ALGORITHM}] Episode {episode:3d} | Reward: {total_reward:.4f}')
#         reward_records.append(total_reward)
#         np.save(f'Reward_{ALGORITHM}.npy', np.array(reward_records))

#         if best_score < total_reward:
#             best_score = total_reward
#             print(f'  → New best score: {best_score:.4f} — saving checkpoint')
#             trainer.save_checkpoints()


# """──────────────────────────────────────────────────────
#     Entry Point
# ──────────────────────────────────────────────────────"""
# if __name__ == '__main__':
#     import os
#     save_dir = './checkpoints'
#     os.makedirs(save_dir, exist_ok=True)
#     print(f'Starting {ALGORITHM} Training...')
#     execute_training(save_dir)

#########execute_training.py##############

import random
import numpy as np
import torch
from scipy.integrate import solve_ivp
from tqdm import tqdm
from road_generator import *
import config
from model import *
from trainer import *

"""──────────────────────────────────────────────────────
    ALGORITHM SELECTION
    ALGORITHM = 'DDPG'  → original, 1500 episodes
    ALGORITHM = 'SAC'   → new contribution, 1500 episodes
    
    SAC needs NO noise schedule — it explores via its
    stochastic policy automatically.
──────────────────────────────────────────────────────"""
ALGORITHM = 'SAC'    # ← change to 'DDPG' for baseline run

ALGO_CONFIG = {
    'DDPG': {'episodes': 1500, 'seed': 99},
    'SAC':  {'episodes': 1500, 'seed': 42},  # same episodes, different seed
}

NUM_EPISODES = ALGO_CONFIG[ALGORITHM]['episodes']
seed         = ALGO_CONFIG[ALGORITHM]['seed']

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

"""──────────────────────────────────────────────────────
    Physical constants
──────────────────────────────────────────────────────"""
m1   = config.m1;   m2 = config.m2;   m  = config.m
cb   = config.cb;   kb = config.kb;   kw = config.kw
dt   = config.dt;   TIME = config.TIME

road_profile = list(
    RoadProfile().get_profile_by_class("E", config.t_stop, config.dt)[1][1:]
)

def odefun(t, y0, m_, cs, kw_, ks):
    m1_ = m_[0];  m2_ = m_[1]
    t_idx = min(round(t/dt), len(TIME)-1)
    xr    = road_profile[t_idx]
    xb=y0[0]; xw=y0[1]; dxb=y0[2]; dxw=y0[3]
    d2xb = -ks/m1_*(xb-xw) - cs/m1_*(dxb-dxw)
    d2xw =  ks/m2_*(xb-xw) + cs/m2_*(dxb-dxw) + kw_/m2_*(xr-xw)
    return [dxb, dxw, d2xb, d2xw]


"""──────────────────────────────────────────────────────
    Replay Buffer
──────────────────────────────────────────────────────"""
class replayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer      = []
        self._next_idx   = 0

    def add(self, item):
        if len(self.buffer) > self._next_idx:
            self.buffer[self._next_idx] = item
        else:
            self.buffer.append(item)
        self._next_idx = (0 if self._next_idx == self.buffer_size-1
                          else self._next_idx+1)

    def sample(self, batch_size):
        idx = [random.randint(0, len(self.buffer)-1)
               for _ in range(batch_size)]
        return ([self.buffer[i][0] for i in idx],
                [self.buffer[i][1] for i in idx],
                [self.buffer[i][2] for i in idx],
                [self.buffer[i][3] for i in idx],
                [self.buffer[i][4] for i in idx])

    def length(self):
        return len(self.buffer)


"""──────────────────────────────────────────────────────
    OU Noise — only used by DDPG
    SAC ignores this via pick_sample(*args)
──────────────────────────────────────────────────────"""
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta=theta; self.mu=mu; self.sigma=sigma
        self.dt=dt; self.x0=x0; self.reset()

    def __call__(self):
        x = (self.x_prev
             + self.theta*(self.mu-self.x_prev)*self.dt
             + self.sigma*np.sqrt(self.dt)
             *np.random.normal(size=self.mu.shape))
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = (self.x0 if self.x0 is not None
                       else np.zeros_like(self.mu))


"""──────────────────────────────────────────────────────
    Action Scaling
──────────────────────────────────────────────────────"""
def get_cs(a): return cb + 600*a
def get_ks(a):
    ka = 5000*a if a > 0 else 2500*a
    return kb + ka


"""──────────────────────────────────────────────────────
    Noise Schedule — DDPG only
──────────────────────────────────────────────────────"""
def get_sigma(episode):
    if   episode < 100: return 0.5
    elif episode < 200: return 0.3
    elif episode < 500: return 0.15
    else:               return 0.05


"""──────────────────────────────────────────────────────
    MAIN TRAINING FUNCTION
──────────────────────────────────────────────────────"""
def execute_training(save_dir='./checkpoints'):
    import os
    os.makedirs(save_dir, exist_ok=True)

    print('='*55)
    print(f'  Algorithm : {ALGORITHM}')
    print(f'  Seed      : {seed}')
    print(f'  Episodes  : {NUM_EPISODES}')
    print(f'  Device    : {config.device}')
    print('='*55)

    if ALGORITHM == 'DDPG':
        trainer = DDPGTrainer(
            ckpt_Q_origin  = f'{save_dir}/DDPG_Q_origin.pt',
            ckpt_mu_origin = f'{save_dir}/DDPG_mu_origin.pt'
        )
        # DDPG needs external OU noise
        def get_noise(episode):
            sigma = get_sigma(episode)
            return OrnsteinUhlenbeckActionNoise(
                mu=np.zeros(config.output_dim),
                sigma=np.ones(config.output_dim)*sigma)

    elif ALGORITHM == 'SAC':
        trainer = SACTrainer(
            ckpt_Q1    = f'{save_dir}/SAC_Q1_origin.pt',
            ckpt_Q2    = f'{save_dir}/SAC_Q2_origin.pt',
            ckpt_actor = f'{save_dir}/SAC_actor.pt',
            hidden_dim     = 256,
            actor_lr       = 3e-4,
            critic_lr      = 3e-4,
            alpha_lr       = 3e-4,
            init_alpha     = 0.2,
            target_entropy = -float(config.output_dim)
        )
        # SAC needs no noise — pass dummy that returns zeros
        class DummyNoise:
            def __call__(self): return np.zeros(config.output_dim)
        def get_noise(episode): return DummyNoise()
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    buffer         = replayBuffer(buffer_size=100000)
    reward_records = []
    best_score     = -99999

    for episode in tqdm(range(NUM_EPISODES), desc=f'Training {ALGORITHM}'):
        global road_profile
        road_profile = list(
            RoadProfile().get_profile_by_class(
                "E", config.t_stop, config.dt)[1][1:]
        )

        noise = get_noise(episode)

        s_ode      = [road_profile[0], road_profile[0], 0, 0]
        s_ode_prev = s_ode[:]
        total_reward = 0

        for t_idx, t in enumerate(config.TIME):
            xb, xw, dxb, dxw = s_ode
            dxr      = (road_profile[t_idx]-road_profile[t_idx-1])/dt
            _, _, dxb_p, dxw_p = s_ode_prev
            dxr_prev = (road_profile[max(t_idx-1,0)]
                        -road_profile[max(t_idx-2,0)])/dt
            s_rl = [dxb, dxw, dxr, dxb_p, dxw_p, dxr_prev]

            # Both DDPG and SAC use same pick_sample interface
            # SAC ignores noise internally
            a      = trainer.pick_sample(s_rl, noise)
            a_cs, a_ks = a[0], a[1]

            yout = solve_ivp(
                odefun, [t, t+dt], s_ode,
                args=(m, get_cs(a_cs), kw, get_ks(a_ks)),
                dense_output=True
            )
            xb_n, xw_n, dxb_n, dxw_n = yout.y[:,-1]
            dxr_n = ((road_profile[t_idx+1]-road_profile[t_idx])/dt
                     if t_idx < len(TIME)-1 else 0.0)

            s_ode_n = [xb_n, xw_n, dxb_n, dxw_n]
            s_rl_n  = [dxb_n, dxw_n, dxr_n, dxb, dxw, dxr]

            d2xb_n = (-get_cs(a_cs)*(dxb_n-dxw_n)
                      -get_ks(a_ks)*(xb_n-xw_n))/m1
            r = -1/10*abs(dxb_n) - 1/100*abs(d2xb_n)
            total_reward += r

            done = (t_idx == len(TIME)-1)
            buffer.add([s_rl, [a_cs, a_ks], r, s_rl_n, float(done)])

            if buffer.length() >= config.bs:
                s, ac, rw, ns, d = buffer.sample(config.bs)
                trainer.optimize(s, ac, rw, ns, d)
                trainer.update_target()

            s_ode_prev = s_ode[:]
            s_ode      = s_ode_n

        reward_records.append(total_reward)
        np.save(f'Reward_{ALGORITHM}.npy', np.array(reward_records))

        if episode % 100 == 0:
            print(f'\n  Ep {episode:4d} | '
                  f'Reward: {total_reward:8.2f} | '
                  f'Best: {best_score:8.2f}')
            if ALGORITHM == 'SAC':
                print(f'  Alpha: {trainer.alpha.item():.4f}')

        if total_reward > best_score:
            best_score = total_reward
            trainer.save_checkpoints()
            print(f'  ★ Ep {episode} new best: {best_score:.3f}')


if __name__ == '__main__':
    execute_training('./checkpoints')