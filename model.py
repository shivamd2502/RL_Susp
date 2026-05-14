# import random
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from config import *

# """************************************
#     Q-Network: CRITIC
#         function: (st,at) -> Q-value = Q(st, at)
# ************************************"""
# class QNet(nn.Module):
#     def __init__(self, input_dim=4+2, hidden_dim=64, output_dim=1):
#         super().__init__()
#         self.hidden1 = nn.Linear(input_dim, hidden_dim)
#         self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
#         self.output = nn.Linear(hidden_dim, output_dim)

#     def forward(self, s, a):
#         outs = torch.cat([s, a], dim=-1)
#         outs = F.relu(self.hidden1(outs))
#         outs = F.relu(self.hidden2(outs))
#         outs = self.output(outs)
#         # outs = 1000*outs
#         return outs

# """************************************
#     Policy Network: ACTOR
#         function:  st -> at
# ************************************"""
# class PolicyNet(nn.Module):
#     def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
#         super().__init__()
#         self.hidden1 = nn.Linear(input_dim, hidden_dim)
#         self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
#         self.output = nn.Linear(hidden_dim, output_dim)

#     def forward(self, s):
#         outs = F.relu(self.hidden1(s))
#         outs = F.relu(self.hidden2(outs))
#         outs = self.output(outs)
#         outs = torch.tanh(outs)    # range [-1, 1]
#         # outs = outs * 500   # desire output is [1000,2000]
#         return outs




########### model.py  ##########
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *

"""************************************
    Q-Network: CRITIC
    Shared by DDPG and SAC.
    Input:  state + action  (6+2 = 8)
    Output: scalar Q-value
************************************"""
class QNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=1):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output  = nn.Linear(hidden_dim, output_dim)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.output(x)


"""************************************
    DDPG Policy Network: ACTOR
    Deterministic: state → action ∈ [-1,1]
************************************"""
class PolicyNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=2):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output  = nn.Linear(hidden_dim, output_dim)

    def forward(self, s):
        x = F.relu(self.hidden1(s))
        x = F.relu(self.hidden2(x))
        return torch.tanh(self.output(x))   # bounded [-1, 1]


"""************************************
    SAC Policy Network: STOCHASTIC ACTOR
    
    Key difference from DDPG PolicyNet:
        DDPG → outputs one deterministic action
        SAC  → outputs (mean, log_std) of a
               Gaussian distribution, then
               samples from it during training.
               At evaluation, uses mean only.
    
    Why this matters:
        Stochastic policy naturally explores
        without needing external OU noise.
        Entropy bonus in reward encourages
        the policy to stay diverse, helping
        it escape local optima that DDPG
        gets stuck in.
    
    Architecture:
        Shared trunk (6→256→256)
        → mean head    (256→2)  tanh bounded
        → log_std head (256→2)  clamped [-20, 2]
************************************"""
LOG_STD_MIN = -20
LOG_STD_MAX =  2

class SACPolicyNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=2):
        super().__init__()
        # Shared feature extractor — larger than DDPG (256 vs 16)
        # SAC needs more capacity to model action distributions
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)

        # Two separate output heads
        self.mean_head    = nn.Linear(hidden_dim, output_dim)
        self.log_std_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, s):
        """
        Returns (mean, log_std) of action distribution.
        Used during training to compute entropy.
        """
        x       = F.relu(self.hidden1(s))
        x       = F.relu(self.hidden2(x))
        mean    = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, s):
        """
        Samples action using reparameterisation trick.
        Returns (action, log_prob, mean).

        Reparameterisation:
            z      ~ N(0, I)
            action  = tanh(mean + std * z)
            log_prob accounts for tanh squashing via
            the change-of-variables formula.

        During training: returns sampled action (exploration)
        During evaluation: caller uses mean directly
        """
        mean, log_std = self.forward(s)
        std = log_std.exp()

        # Sample from Gaussian using reparameterisation
        z      = torch.randn_like(mean)
        x_t    = mean + std * z           # pre-squash sample

        # Squash to [-1, 1] with tanh
        action = torch.tanh(x_t)

        # Log probability with tanh correction
        # log π(a|s) = log N(x_t) - Σ log(1 - tanh²(x_t))
        log_prob = (
            torch.distributions.Normal(mean, std).log_prob(x_t)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)

        # Mean action (used deterministically at evaluation)
        mean_action = torch.tanh(mean)

        return action, log_prob, mean_action