# import random
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from config import *
# from model import *


# """**********************************************************
#     Trainer class which:
#         1. Initialize Actor/Critic weights
#         2. Perform backpropagation given (s{t}, a{t}, r{t}, s{t+1})
#         3. Save checkpoints: 
#             ckpt_Q_origin -> Q-Network; 
#             ckpt_mu_origin -> Policy Network
# **********************************************************"""
# class Trainer:
#     def __init__(self, ckpt_Q_origin, ckpt_mu_origin):
#         self.ckpt_Q_origin = ckpt_Q_origin
#         self.ckpt_mu_origin = ckpt_mu_origin

#         """######################################################################
#             1. Initialize Critics (Q-Networks)
#                 Q_origin: instantly updated weights
#                 Q_target: delayed weights (updated periodically)
#                     * NOTE: Q_target are not back-propagated -> disable require_grads for efficiency
#             --> "Twin-delayed" to increase training stability & convergence
#         ######################################################################"""
#         self.Q_origin = QNet(input_dim+output_dim, Q_hidden_dim, 1).to(device)
#         self.Q_origin = self.Q_origin.to(device)
#         self.Q_origin.requires_grad_(True)

#         self.Q_target = QNet(input_dim+output_dim, Q_hidden_dim, 1).to(device)
#         self.Q_target.requires_grad_(False) # Target network's weights are copied but not back-propagated

#         """###################################################################### 
#             2. Initialize Actors (Policy Networks)
#                 Q_origin: instantly updated weights
#                 Q_target: delayed weights (updated periodically)
#                     * NOTE: Q_target are not back-propagated -> disable require_grads for efficiency
#             --> "Twin-delayed" to increase training stability & convergence
#         ######################################################################"""
#         self.mu_origin = PolicyNet(input_dim, mu_hidden_dim, output_dim).to(device)
#         self.mu_origin = self.mu_origin.to(device)
#         self.mu_origin.requires_grad_(True)

#         self.mu_target = PolicyNet(input_dim, mu_hidden_dim, output_dim).to(device)
#         self.mu_target.requires_grad_(False) # Target network's weights are copied but not back-propagated

#         print('Learning rate:', lr)
#         print('Device:', device)
#         print('Q-Network:', self.Q_origin)
#         print('Policy Network:', self.mu_origin)

#         """###################################################################### 
#             3. Initialize Optimizer
#         ######################################################################"""
#         self.optimizer_Qnet = torch.optim.Adam(self.Q_origin.parameters(), lr=lr)
#         self.optimizer_munet = torch.optim.Adam(self.mu_origin.parameters(), lr=lr/10)

#     """**********************************************************
#         Optimization framework for Twin-Delayed (TD) Critic & Actor
#         :states      - st
#         :actions     - at
#         :rewards     - rt
#         :next_states - s{t+1}
#         :dones       - bool

#         Purpose:
#             * Update weights of Q_orig and mu_orig using back-propagation
#             * Q_target & mu_target are NOT updated (soft delayed updated separately)
#     **********************************************************"""
#     def optimize(self, states, actions, rewards, next_states, dones):
#         """ 1. Tensor conversion of (st, at, rt, s{t+1}, done)"""
#         states = torch.tensor(np.array(states), dtype=torch.float).to(device)
#         actions = torch.tensor(np.array(actions), dtype=torch.float).to(device)
#         # actions = actions.unsqueeze(dim=1)
#         rewards = torch.tensor(rewards, dtype=torch.float).to(device)
#         rewards = rewards.unsqueeze(dim=1)
#         next_states = torch.tensor(next_states, dtype=torch.float).to(device)
#         dones = torch.tensor(dones, dtype=torch.float).to(device)
#         dones = dones.unsqueeze(dim=1)

#         """ 2. Critic Loss Optimization"""
#         self.optimizer_Qnet.zero_grad()
#         qvals = self.Q_origin(states, actions)                            # Q_orig(st, at)
#         a_next_tgt = self.mu_target(next_states)                          # a{t+1}_tgt
#         qvals_next_tgt = self.Q_target(next_states, a_next_tgt)           # Q_tgt(s{t+1}, a{t+1}_tgt)
#         qvals_tgt = rewards + (1.0 - dones) * gamma * qvals_next_tgt # Q_tgt(st, at) = rt + gamma*Q_tgt(s{t+1}, a{t+1})

#         """ 2.1. Update Critic -> Minimize MSE between
#                 `Q_orig(st, at)` and `rt + gamma*Q_tgt(s{t+1}, a{t+1})`
#                     where:
#                         Q_orig, Q_tgt    is given
#                         mu_orig, mu_tgt  is given
#                         st, at, rt, s{t+1} is given
#                         a{t+1} = mu_tgt(s{t+1})
#         """
#         loss_Q = F.mse_loss(qvals, qvals_tgt, reduction="none")
#         loss_Q.sum().backward()
#         self.optimizer_Qnet.step()


#         """ 3. Actor Loss Optimization"""
#         """ 3.1. Freeze Critic (Q_origin) before Actor (mu_origin) optimization"""
#         for p in self.Q_origin.parameters():
#             p.requires_grad = False

#         """ 3.2. Update Actor -> Maximize the reward"""
#         self.optimizer_munet.zero_grad()
#         a_next = self.mu_origin(states)            # a{t+1} = mu_orig(s{t})
#         q_tgt_max = self.Q_origin(states, a_next)  #
#         loss_actor = (-q_tgt_max) - 0.1*abs(a_next)
#         loss_actor.sum().backward()          # (q_tgt_max) or (-q_tgt_max) ?
#         self.optimizer_munet.step()

#         """ 3.1. Unfreeze Critic (Q_origin) """
#         for p in self.Q_origin.parameters():
#             p.requires_grad = True # enable grad again


#     """**********************************************************
#         Update lr of Q-Network and Policy Network to lr/10
#     **********************************************************"""
#     def reduce_lr(self):
#         for g in self.optimizer_Qnet.param_groups:
#             g['lr'] = g['lr'] / 10
#         for g in self.optimizer_munet.param_groups:
#             g['lr'] = g['lr'] / 10

#     """**********************************************************
#         Soft-update Delayed Target models:
#             Q_target = tau*Q_target + (1-tau)*Q_orig
#             mu_target = tau*mu_target + (1-tau)*mu_orig
#     **********************************************************"""
#     def update_target(self):
#         for var, var_target in zip(self.Q_origin.parameters(), self.Q_target.parameters()):
#             var_target.data = tau * var_target.data + (1.0 - tau) * var.data
#         for var, var_target in zip(self.mu_origin.parameters(), self.mu_target.parameters()):
#             var_target.data = tau * var_target.data + (1.0 - tau) * var.data

#     """**********************************************************
#         Pick up action with Ornstein-Uhlenbeck noise
#     **********************************************************"""
#     def pick_sample(self, s, ou_action_noise):
#         with torch.no_grad():
#             s = np.array(s)
#             s_batch = np.expand_dims(s, axis=0)
#             s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
#             action_det = self.mu_origin(s_batch)
#             action_det = action_det.squeeze(dim=1)
#             noise = ou_action_noise()
#             action = action_det.cpu().numpy() + noise
#             # action = np.clip(action, -1.0, 1.0)
#             # return float(action.item())
#             return action.astype(float)[0]
        
#     """**********************************************************
#         Save Actor-Critic weights as new checkpoints
#     **********************************************************"""
#     def save_checkpoints(self):
#         torch.save(self.Q_origin, self.ckpt_Q_origin)
#         torch.save(self.mu_origin, self.ckpt_mu_origin)
        


##########################ver2######################################
###################################################################

# import random
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from config import *
# from model import *


# """**********************************************************
#     DDPG Trainer — Original Implementation (UNCHANGED)
    
#     Uses:
#         - Single Critic (Q_origin + Q_target)
#         - Single Actor  (mu_origin + mu_target)
#         - Bellman target: r + gamma * Q_target(s', mu_target(s'))
# **********************************************************"""
# class DDPGTrainer:
#     def __init__(self, ckpt_Q_origin, ckpt_mu_origin):
#         self.ckpt_Q_origin  = ckpt_Q_origin
#         self.ckpt_mu_origin = ckpt_mu_origin

#         # ── Critic ───────────────────────────────────────────
#         self.Q_origin = QNet(input_dim+output_dim, Q_hidden_dim, 1).to(device)
#         self.Q_origin.requires_grad_(True)

#         self.Q_target = QNet(input_dim+output_dim, Q_hidden_dim, 1).to(device)
#         self.Q_target.requires_grad_(False)   # never backprop through target

#         # ── Actor ────────────────────────────────────────────
#         self.mu_origin = PolicyNet(input_dim, mu_hidden_dim, output_dim).to(device)
#         self.mu_origin.requires_grad_(True)

#         self.mu_target = PolicyNet(input_dim, mu_hidden_dim, output_dim).to(device)
#         self.mu_target.requires_grad_(False)

#         # ── Optimizers ───────────────────────────────────────
#         self.optimizer_Qnet  = torch.optim.Adam(self.Q_origin.parameters(),  lr=lr)
#         self.optimizer_munet = torch.optim.Adam(self.mu_origin.parameters(), lr=lr/10)

#         print('[DDPG] Learning rate:', lr)
#         print('[DDPG] Device:', device)
#         print('[DDPG] Q-Network:', self.Q_origin)
#         print('[DDPG] Policy Network:', self.mu_origin)

#     # ── Core Training Step ───────────────────────────────────
#     def optimize(self, states, actions, rewards, next_states, dones):
#         # 1. Convert to tensors
#         states      = torch.tensor(np.array(states),      dtype=torch.float).to(device)
#         actions     = torch.tensor(np.array(actions),     dtype=torch.float).to(device)
#         rewards     = torch.tensor(rewards,               dtype=torch.float).to(device).unsqueeze(1)
#         next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
#         dones       = torch.tensor(dones,                 dtype=torch.float).to(device).unsqueeze(1)

#         # 2. Critic update
#         self.optimizer_Qnet.zero_grad()
#         qvals          = self.Q_origin(states, actions)
#         a_next_tgt     = self.mu_target(next_states)
#         qvals_next_tgt = self.Q_target(next_states, a_next_tgt)        # single critic target
#         qvals_tgt      = rewards + (1.0 - dones) * gamma * qvals_next_tgt

#         loss_Q = F.mse_loss(qvals, qvals_tgt, reduction="none")
#         loss_Q.sum().backward()
#         self.optimizer_Qnet.step()

#         # 3. Actor update
#         for p in self.Q_origin.parameters():
#             p.requires_grad = False

#         self.optimizer_munet.zero_grad()
#         a_new        = self.mu_origin(states)
#         q_tgt_max    = self.Q_origin(states, a_new)
#         loss_actor   = (-q_tgt_max) - 0.1 * abs(a_new)
#         loss_actor.sum().backward()
#         self.optimizer_munet.step()

#         for p in self.Q_origin.parameters():
#             p.requires_grad = True

#     # ── Soft Update ──────────────────────────────────────────
#     def update_target(self):
#         for var, var_tgt in zip(self.Q_origin.parameters(),  self.Q_target.parameters()):
#             var_tgt.data = tau * var_tgt.data + (1.0 - tau) * var.data
#         for var, var_tgt in zip(self.mu_origin.parameters(), self.mu_target.parameters()):
#             var_tgt.data = tau * var_tgt.data + (1.0 - tau) * var.data

#     # ── Action Selection ─────────────────────────────────────
#     def pick_sample(self, s, ou_action_noise):
#         with torch.no_grad():
#             s_batch     = torch.tensor(np.expand_dims(np.array(s), 0), dtype=torch.float).to(device)
#             action_det  = self.mu_origin(s_batch).squeeze(1)
#             noise       = ou_action_noise()
#             action      = action_det.cpu().numpy() + noise
#             return action.astype(float)[0]

#     # ── Save ─────────────────────────────────────────────────
#     def save_checkpoints(self):
#         torch.save(self.Q_origin,  self.ckpt_Q_origin)
#         torch.save(self.mu_origin, self.ckpt_mu_origin)

#     def reduce_lr(self):
#         for g in self.optimizer_Qnet.param_groups:
#             g['lr'] /= 10
#         for g in self.optimizer_munet.param_groups:
#             g['lr'] /= 10


# """**********************************************************
#     TD3 Trainer — NEW Implementation
    
#     Key differences from DDPG:
#     ─────────────────────────────────────────────────────────
#     1. TWO Critics (Q1, Q2) instead of one
#        → During target computation, take min(Q1_target, Q2_target)
#        → This prevents Q-value overestimation (DDPG's main weakness)
    
#     2. Target Policy Smoothing
#        → Add small clipped noise to target actor's action
#        → Prevents Actor from exploiting sharp Q-value peaks
    
#     3. Delayed Actor Updates (optional — set actor_update_freq)
#        → Actor updates less frequently than Critic
#        → Critic becomes more accurate before Actor chases it
    
#     Architecture:
#         Q1_origin, Q1_target  ← first  critic pair  (same QNet class)
#         Q2_origin, Q2_target  ← second critic pair  (same QNet class)
#         mu_origin, mu_target  ← single actor pair   (same PolicyNet class)
# **********************************************************"""
# class TD3Trainer:
#     def __init__(self, ckpt_Q1_origin, ckpt_Q2_origin, ckpt_mu_origin,
#                  actor_update_freq=2,          # Actor updates every N critic steps
#                  target_noise=0.2,             # std of smoothing noise on target action
#                  target_noise_clip=0.5):       # clip smoothing noise to [-clip, +clip]

#         self.ckpt_Q1_origin  = ckpt_Q1_origin
#         self.ckpt_Q2_origin  = ckpt_Q2_origin
#         self.ckpt_mu_origin  = ckpt_mu_origin

#         self.actor_update_freq   = actor_update_freq
#         self.target_noise        = target_noise
#         self.target_noise_clip   = target_noise_clip
#         self.total_updates       = 0            # counts critic updates for delayed actor

#         # ── Critic 1 ─────────────────────────────────────────
#         # Identical architecture to DDPG critic
#         # Role: first independent Q-value estimate
#         self.Q1_origin = QNet(input_dim+output_dim, Q_hidden_dim, 1).to(device)
#         self.Q1_origin.requires_grad_(True)
#         self.Q1_target = QNet(input_dim+output_dim, Q_hidden_dim, 1).to(device)
#         self.Q1_target.requires_grad_(False)

#         # ── Critic 2 ─────────────────────────────────────────
#         # Identical architecture to Critic 1
#         # Role: second independent Q-value estimate
#         # KEY: same input, same architecture, different random init weights
#         # → produces different Q-value estimates
#         # → taking minimum of both is conservative and prevents overestimation
#         self.Q2_origin = QNet(input_dim+output_dim, Q_hidden_dim, 1).to(device)
#         self.Q2_origin.requires_grad_(True)
#         self.Q2_target = QNet(input_dim+output_dim, Q_hidden_dim, 1).to(device)
#         self.Q2_target.requires_grad_(False)

#         # ── Actor (single — same as DDPG) ────────────────────
#         self.mu_origin = PolicyNet(input_dim, mu_hidden_dim, output_dim).to(device)
#         self.mu_origin.requires_grad_(True)
#         self.mu_target = PolicyNet(input_dim, mu_hidden_dim, output_dim).to(device)
#         self.mu_target.requires_grad_(False)

#         # ── Optimizers ───────────────────────────────────────
#         # Both critics trained independently with same lr
#         self.optimizer_Q1net = torch.optim.Adam(self.Q1_origin.parameters(), lr=lr)
#         self.optimizer_Q2net = torch.optim.Adam(self.Q2_origin.parameters(), lr=lr)
#         self.optimizer_munet = torch.optim.Adam(self.mu_origin.parameters(), lr=lr/10)

#         print('[TD3] Learning rate:', lr)
#         print('[TD3] Device:', device)
#         print('[TD3] Actor update frequency: every', actor_update_freq, 'critic steps')
#         print('[TD3] Target noise:', target_noise, '| clip:', target_noise_clip)
#         print('[TD3] Q1-Network:', self.Q1_origin)
#         print('[TD3] Q2-Network:', self.Q2_origin)
#         print('[TD3] Policy Network:', self.mu_origin)

#     # ── Core Training Step ───────────────────────────────────
#     def optimize(self, states, actions, rewards, next_states, dones):
#         self.total_updates += 1

#         # 1. Convert to tensors (identical to DDPG)
#         states      = torch.tensor(np.array(states),      dtype=torch.float).to(device)
#         actions     = torch.tensor(np.array(actions),     dtype=torch.float).to(device)
#         rewards     = torch.tensor(rewards,               dtype=torch.float).to(device).unsqueeze(1)
#         next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
#         dones       = torch.tensor(dones,                 dtype=torch.float).to(device).unsqueeze(1)

#         # ────────────────────────────────────────────────────
#         # 2. TD3 TARGET COMPUTATION — the key difference from DDPG
#         # ────────────────────────────────────────────────────
#         with torch.no_grad():
#             # 2a. Target actor picks next action
#             a_next_tgt = self.mu_target(next_states)

#             # 2b. TARGET POLICY SMOOTHING — add clipped noise to target action
#             #     This prevents the Actor from exploiting narrow Q-value peaks
#             #     that the Critics might overfit to
#             noise = torch.randn_like(a_next_tgt) * self.target_noise
#             noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
#             a_next_tgt = (a_next_tgt + noise).clamp(-1.0, 1.0)  # keep in valid range

#             # 2c. CLIPPED DOUBLE Q — both target critics evaluate the noisy action
#             q1_next = self.Q1_target(next_states, a_next_tgt)
#             q2_next = self.Q2_target(next_states, a_next_tgt)

#             # 2d. Take MINIMUM of the two — conservative estimate
#             #     This is the core TD3 contribution:
#             #     DDPG uses:  target = r + gamma * Q_target(s', a')
#             #     TD3 uses:   target = r + gamma * MIN(Q1_target, Q2_target)(s', a')
#             #     Why min? Because overestimated Q → Actor chases wrong actions
#             q_next_min = torch.min(q1_next, q2_next)
#             qvals_tgt  = rewards + (1.0 - dones) * gamma * q_next_min

#         # ────────────────────────────────────────────────────
#         # 3. Update BOTH Critics independently
#         #    Each critic is trained separately with its own optimizer
#         #    Both use the SAME target (qvals_tgt computed above)
#         # ────────────────────────────────────────────────────

#         # Critic 1 update
#         self.optimizer_Q1net.zero_grad()
#         q1_vals  = self.Q1_origin(states, actions)
#         loss_Q1  = F.mse_loss(q1_vals, qvals_tgt, reduction="none")
#         loss_Q1.sum().backward()
#         self.optimizer_Q1net.step()

#         # Critic 2 update (independent — different weights, different gradients)
#         self.optimizer_Q2net.zero_grad()
#         q2_vals  = self.Q2_origin(states, actions)
#         loss_Q2  = F.mse_loss(q2_vals, qvals_tgt, reduction="none")
#         loss_Q2.sum().backward()
#         self.optimizer_Q2net.step()

#         # ────────────────────────────────────────────────────
#         # 4. DELAYED Actor update
#         #    Actor only updates every actor_update_freq critic steps
#         #    Default: every 2 steps → Critic is more accurate before Actor follows
#         # ────────────────────────────────────────────────────
#         if self.total_updates % self.actor_update_freq == 0:
#             # Freeze BOTH critics before Actor update
#             for p in self.Q1_origin.parameters():
#                 p.requires_grad = False
#             for p in self.Q2_origin.parameters():
#                 p.requires_grad = False

#             # Actor maximises Q1 (standard choice — either critic works)
#             self.optimizer_munet.zero_grad()
#             a_new      = self.mu_origin(states)
#             q_actor    = self.Q1_origin(states, a_new)
#             loss_actor = (-q_actor) - 0.1 * abs(a_new)
#             loss_actor.sum().backward()
#             self.optimizer_munet.step()

#             # Unfreeze both critics
#             for p in self.Q1_origin.parameters():
#                 p.requires_grad = True
#             for p in self.Q2_origin.parameters():
#                 p.requires_grad = True

#     # ── Soft Update — now updates 3 pairs instead of 2 ──────
#     def update_target(self):
#         # Q1 pair
#         for var, var_tgt in zip(self.Q1_origin.parameters(), self.Q1_target.parameters()):
#             var_tgt.data = tau * var_tgt.data + (1.0 - tau) * var.data
#         # Q2 pair
#         for var, var_tgt in zip(self.Q2_origin.parameters(), self.Q2_target.parameters()):
#             var_tgt.data = tau * var_tgt.data + (1.0 - tau) * var.data
#         # Actor pair
#         for var, var_tgt in zip(self.mu_origin.parameters(), self.mu_target.parameters()):
#             var_tgt.data = tau * var_tgt.data + (1.0 - tau) * var.data

#     # ── Action Selection — identical to DDPG ────────────────
#     def pick_sample(self, s, ou_action_noise):
#         with torch.no_grad():
#             s_batch    = torch.tensor(np.expand_dims(np.array(s), 0), dtype=torch.float).to(device)
#             action_det = self.mu_origin(s_batch).squeeze(1)
#             noise      = ou_action_noise()
#             action     = action_det.cpu().numpy() + noise
#             return action.astype(float)[0]

#     # ── Save — saves all three origin networks ───────────────
#     def save_checkpoints(self):
#         torch.save(self.Q1_origin, self.ckpt_Q1_origin)
#         torch.save(self.Q2_origin, self.ckpt_Q2_origin)
#         torch.save(self.mu_origin, self.ckpt_mu_origin)

#     def reduce_lr(self):
#         for g in self.optimizer_Q1net.param_groups:
#             g['lr'] /= 10
#         for g in self.optimizer_Q2net.param_groups:
#             g['lr'] /= 10
#         for g in self.optimizer_munet.param_groups:
#             g['lr'] /= 10




##########  trainer.py ##################

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *
from model import *


"""**********************************************************
    DDPGTrainer — Original, kept unchanged for comparison.
    Target networks initialised from online networks (fix).
**********************************************************"""
class DDPGTrainer:
    def __init__(self, ckpt_Q_origin, ckpt_mu_origin):
        self.ckpt_Q_origin  = ckpt_Q_origin
        self.ckpt_mu_origin = ckpt_mu_origin

        self.Q_origin = QNet(input_dim+output_dim, Q_hidden_dim, 1).to(device)
        self.Q_origin.requires_grad_(True)
        self.Q_target = QNet(input_dim+output_dim, Q_hidden_dim, 1).to(device)
        self.Q_target.requires_grad_(False)
        self.Q_target.load_state_dict(self.Q_origin.state_dict())

        self.mu_origin = PolicyNet(input_dim, mu_hidden_dim, output_dim).to(device)
        self.mu_origin.requires_grad_(True)
        self.mu_target = PolicyNet(input_dim, mu_hidden_dim, output_dim).to(device)
        self.mu_target.requires_grad_(False)
        self.mu_target.load_state_dict(self.mu_origin.state_dict())

        self.opt_Q  = torch.optim.Adam(self.Q_origin.parameters(),  lr=lr)
        self.opt_mu = torch.optim.Adam(self.mu_origin.parameters(), lr=lr/10)
        print(f'[DDPG] device={device} | critic_lr={lr} | actor_lr={lr/10}')

    def optimize(self, states, actions, rewards, next_states, dones):
        S  = torch.tensor(np.array(states),      dtype=torch.float).to(device)
        A  = torch.tensor(np.array(actions),     dtype=torch.float).to(device)
        R  = torch.tensor(rewards, dtype=torch.float).to(device).unsqueeze(1)
        S_ = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
        D  = torch.tensor(dones,  dtype=torch.float).to(device).unsqueeze(1)

        # Critic update
        self.opt_Q.zero_grad()
        with torch.no_grad():
            a_next   = self.mu_target(S_)
            q_target = R + (1.0 - D) * gamma * self.Q_target(S_, a_next)
        loss_Q = F.mse_loss(self.Q_origin(S, A), q_target, reduction="none")
        loss_Q.sum().backward()
        self.opt_Q.step()

        # Actor update
        for p in self.Q_origin.parameters(): p.requires_grad = False
        self.opt_mu.zero_grad()
        a_cur = self.mu_origin(S)
        loss_actor = (-self.Q_origin(S, a_cur)) - 0.1*abs(a_cur)
        loss_actor.sum().backward()
        self.opt_mu.step()
        for p in self.Q_origin.parameters(): p.requires_grad = True

    def update_target(self):
        for p, pt in zip(self.Q_origin.parameters(),  self.Q_target.parameters()):
            pt.data = tau*pt.data + (1.0-tau)*p.data
        for p, pt in zip(self.mu_origin.parameters(), self.mu_target.parameters()):
            pt.data = tau*pt.data + (1.0-tau)*p.data

    def pick_sample(self, s, ou_noise):
        with torch.no_grad():
            sb = torch.tensor(np.expand_dims(np.array(s),0),
                              dtype=torch.float).to(device)
            a  = self.mu_origin(sb).squeeze(1).cpu().numpy()
            return (a + ou_noise()).astype(float)[0]

    def save_checkpoints(self):
        torch.save(self.Q_origin,  self.ckpt_Q_origin)
        torch.save(self.mu_origin, self.ckpt_mu_origin)


"""**********************************************************
    SACTrainer — Soft Actor-Critic

    Why SAC beats DDPG on this problem:
    ─────────────────────────────────────────────────────────
    DDPG uses a deterministic policy — it always outputs
    the exact same action for the same state. This makes it
    prone to getting stuck in local optima, especially in
    stiff physical systems like the quarter-car ODE.

    SAC uses a stochastic policy and maximises:
        J(π) = Σ E[r(s,a) + α·H(π(·|s))]
    where H is the entropy of the policy distribution.

    The entropy bonus α·H(π) rewards the agent for being
    uncertain — it stays exploratory throughout training,
    not just at the start. This fundamentally different
    objective helps SAC find better policies on physical
    systems where DDPG plateaus.

    Key components:
        1. Two critics (like TD3) — clipped double Q
        2. Stochastic actor (SACPolicyNet)
        3. Automatic entropy tuning (α adapts during training)
        4. No external exploration noise needed
    ─────────────────────────────────────────────────────────
    Architecture:
        Q1_origin, Q1_target  ← critic 1 pair  (QNet 8→256→256→1)
        Q2_origin, Q2_target  ← critic 2 pair  (QNet 8→256→256→1)
        actor                 ← SACPolicyNet    (6→256→256→[mean,std])
        log_alpha             ← learnable entropy temperature
**********************************************************"""
class SACTrainer:
    def __init__(self, ckpt_Q1, ckpt_Q2, ckpt_actor,
                 hidden_dim=256,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 alpha_lr=3e-4,
                 init_alpha=0.2,
                 target_entropy=None):

        self.ckpt_Q1    = ckpt_Q1
        self.ckpt_Q2    = ckpt_Q2
        self.ckpt_actor = ckpt_actor

        # ── Two Critics (clipped double Q, same as TD3) ───
        # Larger hidden dim (256) than DDPG (32) because SAC
        # needs to model a more complex stochastic value function
        self.Q1_origin = QNet(input_dim+output_dim, hidden_dim, 1).to(device)
        self.Q1_target = QNet(input_dim+output_dim, hidden_dim, 1).to(device)
        self.Q1_target.requires_grad_(False)
        self.Q1_target.load_state_dict(self.Q1_origin.state_dict())

        self.Q2_origin = QNet(input_dim+output_dim, hidden_dim, 1).to(device)
        self.Q2_target = QNet(input_dim+output_dim, hidden_dim, 1).to(device)
        self.Q2_target.requires_grad_(False)
        self.Q2_target.load_state_dict(self.Q2_origin.state_dict())

        # ── Stochastic Actor ─────────────────────────────
        self.actor = SACPolicyNet(input_dim, hidden_dim, output_dim).to(device)

        # ── Automatic Entropy Tuning ─────────────────────
        # α controls how much entropy is rewarded.
        # Instead of fixing α, we learn it automatically.
        # Target entropy is set to -output_dim by default
        # (heuristic from SAC paper: Haarnoja et al. 2018)
        self.target_entropy = (target_entropy if target_entropy is not None
                               else -float(output_dim))
        self.log_alpha = torch.tensor(
            np.log(init_alpha), dtype=torch.float,
            requires_grad=True, device=device
        )

        # ── Optimizers ───────────────────────────────────
        self.opt_Q1    = torch.optim.Adam(self.Q1_origin.parameters(), lr=critic_lr)
        self.opt_Q2    = torch.optim.Adam(self.Q2_origin.parameters(), lr=critic_lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(),     lr=actor_lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha],            lr=alpha_lr)

        print(f'[SAC] device={device}')
        print(f'[SAC] critic_lr={critic_lr} | actor_lr={actor_lr} | alpha_lr={alpha_lr}')
        print(f'[SAC] init_alpha={init_alpha} | target_entropy={self.target_entropy}')
        print(f'[SAC] hidden_dim={hidden_dim}')

    @property
    def alpha(self):
        """Current entropy temperature (always positive via exp)."""
        return self.log_alpha.exp()

    def optimize(self, states, actions, rewards, next_states, dones):
        S  = torch.tensor(np.array(states),      dtype=torch.float).to(device)
        A  = torch.tensor(np.array(actions),     dtype=torch.float).to(device)
        R  = torch.tensor(rewards, dtype=torch.float).to(device).unsqueeze(1)
        S_ = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
        D  = torch.tensor(dones,  dtype=torch.float).to(device).unsqueeze(1)

        # ────────────────────────────────────────────────
        # Step 1: Compute SAC Bellman target
        #
        # SAC target differs from DDPG in one key way:
        #   DDPG: Q_tgt = r + γ·Q_target(s', μ(s'))
        #   SAC:  Q_tgt = r + γ·[Q_target(s',a') - α·log π(a'|s')]
        #                                           ↑ entropy bonus
        # The entropy term α·log π(a'|s') subtracts uncertainty —
        # it rewards keeping future options open.
        # ────────────────────────────────────────────────
        with torch.no_grad():
            a_next, log_prob_next, _ = self.actor.sample(S_)
            q1_next = self.Q1_target(S_, a_next)
            q2_next = self.Q2_target(S_, a_next)
            q_next  = torch.min(q1_next, q2_next)           # clipped double Q
            q_target = R + (1.0-D) * gamma * (q_next - self.alpha * log_prob_next)

        # ── Update Critic 1 ───────────────────────────────
        self.opt_Q1.zero_grad()
        loss_Q1 = F.mse_loss(self.Q1_origin(S, A), q_target, reduction="none")
        loss_Q1.sum().backward()
        self.opt_Q1.step()

        # ── Update Critic 2 ───────────────────────────────
        self.opt_Q2.zero_grad()
        loss_Q2 = F.mse_loss(self.Q2_origin(S, A), q_target, reduction="none")
        loss_Q2.sum().backward()
        self.opt_Q2.step()

        # ────────────────────────────────────────────────
        # Step 2: Update Actor
        #
        # SAC actor loss:
        #   L(π) = E[α·log π(a|s) - Q(s,a)]
        # Minimising this = maximising Q while maximising entropy.
        # This is opposite of DDPG where actor just maximises Q.
        # ────────────────────────────────────────────────
        self.opt_actor.zero_grad()
        a_cur, log_prob_cur, _ = self.actor.sample(S)
        q1_cur = self.Q1_origin(S, a_cur)
        q2_cur = self.Q2_origin(S, a_cur)
        q_cur  = torch.min(q1_cur, q2_cur)
        loss_actor = (self.alpha.detach() * log_prob_cur - q_cur).mean()
        loss_actor.backward()
        self.opt_actor.step()

        # ────────────────────────────────────────────────
        # Step 3: Update Alpha (automatic entropy tuning)
        #
        # α is adjusted so that actual entropy stays close
        # to target_entropy. If policy is too deterministic
        # (low entropy), α increases to encourage exploration.
        # If policy is too random, α decreases.
        # DDPG has no equivalent — it uses fixed noise schedule.
        # ────────────────────────────────────────────────
        self.opt_alpha.zero_grad()
        loss_alpha = -(self.log_alpha *
                       (log_prob_cur + self.target_entropy).detach()).mean()
        loss_alpha.backward()
        self.opt_alpha.step()

    def update_target(self):
        """Soft update both critic targets — called every step."""
        for p, pt in zip(self.Q1_origin.parameters(), self.Q1_target.parameters()):
            pt.data = tau*pt.data + (1.0-tau)*p.data
        for p, pt in zip(self.Q2_origin.parameters(), self.Q2_target.parameters()):
            pt.data = tau*pt.data + (1.0-tau)*p.data

    def pick_sample(self, s, *args):
        """
        SAC does NOT use external OU noise.
        The stochastic policy explores naturally.
        The *args absorbs the ou_noise argument passed
        by the training loop so no loop changes needed.
        During training: samples from distribution.
        """
        with torch.no_grad():
            sb = torch.tensor(np.expand_dims(np.array(s),0),
                              dtype=torch.float).to(device)
            action, _, _ = self.actor.sample(sb)
            return action.squeeze(0).cpu().numpy().astype(float)

    def pick_sample_deterministic(self, s):
        """Used during evaluation — returns mean action only."""
        with torch.no_grad():
            sb = torch.tensor(np.expand_dims(np.array(s),0),
                              dtype=torch.float).to(device)
            _, _, mean_action = self.actor.sample(sb)
            return mean_action.squeeze(0).cpu().numpy().astype(float)

    def save_checkpoints(self):
        torch.save(self.Q1_origin, self.ckpt_Q1)
        torch.save(self.Q2_origin, self.ckpt_Q2)
        torch.save(self.actor,     self.ckpt_actor)