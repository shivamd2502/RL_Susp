###############################ver1####################################
#######################################################################


# """
# evaluation.py
# ─────────────────────────────────────────────────────────────────────
# Change 3: Multi-Road Class Evaluation

# Tests BOTH trained models (DDPG and TD3) across FOUR road classes:
#     Class B → smooth highway
#     Class C → average road
#     Class D → poor road
#     Class E → very rough (same as training)

# For each road class, computes:
#     → % velocity reduction  (mean and Q3)
#     → % acceleration reduction (mean and Q3)

# Produces:
#     1. Summary comparison table (all algorithms × all road classes)
#     2. Individual plots for each road class
#     3. Saves all results to results_summary.npy
# ─────────────────────────────────────────────────────────────────────
# """

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from scipy.integrate import solve_ivp
# from config import *
# from road_generator import *
# from model import *

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# """──────────────────────────────────────────────────────
#     CONFIGURATION — change these paths to your saved models
# ──────────────────────────────────────────────────────"""
# DDPG_MU_PATH = './checkpoints/DDPG_mu_origin.pt'
# TD3_MU_PATH  = './checkpoints/TD3_mu_origin.pt'

# # Road classes to test — Change 3 is here
# # Original paper only tested Class E
# # We test B, C, D, E for generalisation analysis
# ROAD_CLASSES = ['B', 'C', 'D', 'E']

# # Number of test runs per road class (averaged for reliability)
# NUM_TEST_RUNS = 3


# """──────────────────────────────────────────────────────
#     OU Noise class (zero sigma = deterministic evaluation)
# ──────────────────────────────────────────────────────"""
# class OrnsteinUhlenbeckActionNoise:
#     def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
#         self.theta = theta;  self.mu = mu
#         self.sigma = sigma;  self.dt = dt
#         self.x0    = x0;     self.reset()

#     def __call__(self):
#         x = (self.x_prev
#              + self.theta*(self.mu - self.x_prev)*self.dt
#              + self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape))
#         self.x_prev = x
#         return x

#     def reset(self):
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# """──────────────────────────────────────────────────────
#     Action scaling — identical to execute_training.py
# ──────────────────────────────────────────────────────"""
# def get_cs(a):
#     return cb + 600*a

# def get_ks(a):
#     ka = 5000*a if a > 0 else 2500*a
#     return kb + ka


# """──────────────────────────────────────────────────────
#     ODE function — identical to execute_training.py
# ──────────────────────────────────────────────────────"""
# road_profile = None   # set before each simulation run

# def odefun(t, y0, m, cs, kw, ks):
#     m1 = m[0];  m2 = m[1]
#     t_idx = min(round(t/dt), len(TIME)-1)
#     xr    = road_profile[t_idx]
#     xb  = y0[0];  xw  = y0[1]
#     dxb = y0[2];  dxw = y0[3]
#     d2xb = -ks/m1*(xb-xw) - cs/m1*(dxb-dxw)
#     d2xw =  ks/m2*(xb-xw) + cs/m2*(dxb-dxw) + kw/m2*(xr-xw)
#     return [dxb, dxw, d2xb, d2xw]


# """──────────────────────────────────────────────────────
#     Action picker — same as original evaluation.ipynb
# ──────────────────────────────────────────────────────"""
# def pick_sample(mu_model, s, ou_noise):
#     with torch.no_grad():
#         s_batch    = torch.tensor(np.expand_dims(np.array(s), 0),
#                                   dtype=torch.float).to(device)
#         action_det = mu_model(s_batch).squeeze(1)
#         noise      = ou_noise()
#         action     = action_det.cpu().numpy() + noise
#         return action.astype(float)[0]


# """──────────────────────────────────────────────────────
#     run_passive()
#     Simulates passive suspension (fixed cb, kb) on given road
#     Returns: xb_list, dxb_list, d2xb_list
# ──────────────────────────────────────────────────────"""
# def run_passive(road):
#     global road_profile
#     road_profile = road

#     s_ode    = [road[0], road[0], 0, 0]
#     xb_list  = [];  dxb_list = [];  d2xb_list = []

#     for t_idx, t in enumerate(TIME):
#         xb, xw, dxb, dxw = np.array(s_ode)[0:4]
#         d2xb = (-cb*(dxb-dxw) - kb*(xb-xw)) / m1

#         xb_list.append(xb)
#         dxb_list.append(dxb)
#         d2xb_list.append(d2xb)

#         yout  = solve_ivp(odefun, [t, t+dt], s_ode,
#                           args=([m1,m2], cb, kw, kb), dense_output=True)
#         s_ode = list(yout.y[:,-1])

#     return xb_list, dxb_list, d2xb_list


# """──────────────────────────────────────────────────────
#     run_rl()
#     Simulates RL-controlled suspension on given road
#     Returns: xb_list, dxb_list, d2xb_list, cs_list, ks_list
# ──────────────────────────────────────────────────────"""
# def run_rl(mu_model, road):
#     global road_profile
#     road_profile = road

#     ou_noise = OrnsteinUhlenbeckActionNoise(
#         mu=np.zeros(output_dim),
#         sigma=np.ones(output_dim)*0.0   # NO noise during evaluation
#     )

#     s_ode      = [road[0], road[0], 0, 0]
#     s_ode_prev = s_ode

#     xb_list  = [];  dxb_list = [];  d2xb_list = []
#     cs_list  = [];  ks_list  = []

#     for t_idx, t in enumerate(TIME):
#         xb, xw, dxb, dxw = np.array(s_ode)[0:4]
#         dxr      = (road[t_idx] - road[max(t_idx-1,0)]) / dt
#         xb_p, xw_p, dxb_p, dxw_p = np.array(s_ode_prev)[0:4]
#         dxr_prev = (road[max(t_idx-1,0)] - road[max(t_idx-2,0)]) / dt

#         s_rl    = [dxb, dxw, dxr, dxb_p, dxw_p, dxr_prev]
#         a       = pick_sample(mu_model, s_rl, ou_noise)
#         a_cs, a_ks = a

#         d2xb = (-get_cs(a_cs)*(dxb-dxw) - get_ks(a_ks)*(xb-xw)) / m1

#         xb_list.append(xb)
#         dxb_list.append(dxb)
#         d2xb_list.append(d2xb)
#         cs_list.append(get_cs(a_cs))
#         ks_list.append(get_ks(a_ks))

#         yout  = solve_ivp(odefun, [t, t+dt], s_ode,
#                           args=([m1,m2], get_cs(a_cs), kw, get_ks(a_ks)),
#                           dense_output=True)
#         s_ode_prev = s_ode
#         s_ode      = list(yout.y[:,-1])

#     return xb_list, dxb_list, d2xb_list, cs_list, ks_list


# """──────────────────────────────────────────────────────
#     compute_metrics()
#     Computes all 4 performance metrics used in paper Table II
# ──────────────────────────────────────────────────────"""
# def compute_metrics(passive_dxb, passive_d2xb, rl_dxb, rl_d2xb):
#     p_dxb  = np.abs(passive_dxb)
#     r_dxb  = np.abs(rl_dxb)
#     p_d2xb = np.abs(passive_d2xb)
#     r_d2xb = np.abs(rl_d2xb)

#     mean_vel_change = round((r_dxb.mean()  - p_dxb.mean())  / p_dxb.mean()  * 100, 5)
#     q3_vel_change   = round((np.quantile(r_dxb, 0.75)  - np.quantile(p_dxb, 0.75))
#                              / np.quantile(p_dxb, 0.75) * 100, 5)
#     mean_acc_change = round((r_d2xb.mean() - p_d2xb.mean()) / p_d2xb.mean() * 100, 5)
#     q3_acc_change   = round((np.quantile(r_d2xb, 0.75) - np.quantile(p_d2xb, 0.75))
#                              / np.quantile(p_d2xb, 0.75) * 100, 5)

#     return {
#         'mean_vel_%':  mean_vel_change,
#         'q3_vel_%':    q3_vel_change,
#         'mean_acc_%':  mean_acc_change,
#         'q3_acc_%':    q3_acc_change,
#     }


# """──────────────────────────────────────────────────────
#     plot_road_comparison()
#     Generates comparison plots for one road class
#     Shows passive vs DDPG vs TD3 side by side
# ──────────────────────────────────────────────────────"""
# def plot_road_comparison(road_class, road,
#                          passive_xb,  passive_dxb,  passive_d2xb,
#                          ddpg_xb,     ddpg_dxb,     ddpg_d2xb,
#                          td3_xb,      td3_dxb,      td3_d2xb,
#                          ddpg_cs, ddpg_ks, td3_cs, td3_ks):

#     fig = plt.figure(figsize=(18, 14))
#     fig.suptitle(f'ISO 8608 Class-{road_class} Road Profile: '
#                  f'Passive vs DDPG vs TD3', fontsize=16, fontweight='bold')

#     gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

#     time_axis = np.array(TIME) * 100   # convert to 10^-2 s for x-axis label

#     # ── Plot 1: Road Profile ─────────────────────────────────
#     ax1 = fig.add_subplot(gs[0, :])
#     ax1.plot(road[1:len(passive_xb)], color='black', linewidth=1.5)
#     ax1.set_title(f'Road Excitation ($x_r$) — Class {road_class}', fontsize=13)
#     ax1.set_xlabel('Time ($10^{-2}$ s)', fontsize=11)
#     ax1.set_ylabel('Excitation (m)', fontsize=11)
#     ax1.grid(True, linestyle='--', alpha=0.5)

#     # ── Plot 2: Body Displacement ────────────────────────────
#     ax2 = fig.add_subplot(gs[1, 0])
#     ax2.plot(passive_xb[1:],  '-.', color='gray',   label='Passive',  linewidth=1.2)
#     ax2.plot(ddpg_xb[1:],     '--', color='blue',   label='DDPG',     linewidth=1.2)
#     ax2.plot(td3_xb[1:],           color='red',    label='TD3',      linewidth=1.5)
#     ax2.set_title('Body Displacement ($x_b$)', fontsize=13)
#     ax2.set_xlabel('Time ($10^{-2}$ s)', fontsize=11)
#     ax2.set_ylabel('Displacement (m)', fontsize=11)
#     ax2.legend(fontsize=10)
#     ax2.grid(True, linestyle='--', alpha=0.5)

#     # ── Plot 3: Body Velocity ────────────────────────────────
#     ax3 = fig.add_subplot(gs[1, 1])
#     ax3.plot(passive_dxb[1:],  '-.', color='gray',   label='Passive',  linewidth=1.2)
#     ax3.plot(ddpg_dxb[1:],     '--', color='blue',   label='DDPG',     linewidth=1.2)
#     ax3.plot(td3_dxb[1:],           color='red',    label='TD3',      linewidth=1.5)
#     ax3.set_title('Body Velocity ($\\dot{x}_b$)', fontsize=13)
#     ax3.set_xlabel('Time ($10^{-2}$ s)', fontsize=11)
#     ax3.set_ylabel('Velocity (m/s)', fontsize=11)
#     ax3.legend(fontsize=10)
#     ax3.grid(True, linestyle='--', alpha=0.5)

#     # ── Plot 4: Body Acceleration ────────────────────────────
#     ax4 = fig.add_subplot(gs[2, 0])
#     ax4.plot(passive_d2xb[1:],  '-.', color='gray',   label='Passive',  linewidth=1.2)
#     ax4.plot(ddpg_d2xb[1:],     '--', color='blue',   label='DDPG',     linewidth=1.2)
#     ax4.plot(td3_d2xb[1:],           color='red',    label='TD3',      linewidth=1.5)
#     ax4.set_title('Body Acceleration ($\\ddot{x}_b$)', fontsize=13)
#     ax4.set_xlabel('Time ($10^{-2}$ s)', fontsize=11)
#     ax4.set_ylabel('Acceleration (m/s²)', fontsize=11)
#     ax4.legend(fontsize=10)
#     ax4.grid(True, linestyle='--', alpha=0.5)

#     # ── Plot 5: RL Controlled Damping and Stiffness ──────────
#     ax5 = fig.add_subplot(gs[2, 1])
#     ax5.plot(ddpg_cs[1:], '--', color='blue',  label='DDPG $C_a$',  linewidth=1.0)
#     ax5.plot(td3_cs[1:],       color='red',   label='TD3 $C_a$',   linewidth=1.0)
#     ax5.set_title('Controlled Damping ($C_a$)', fontsize=13)
#     ax5.set_xlabel('Time ($10^{-2}$ s)', fontsize=11)
#     ax5.set_ylabel('Damping (Ns/m)', fontsize=11)
#     ax5.legend(fontsize=10)
#     ax5.grid(True, linestyle='--', alpha=0.5)

#     plt.savefig(f'comparison_class_{road_class}.png', dpi=150, bbox_inches='tight')
#     print(f'  → Saved: comparison_class_{road_class}.png')
#     plt.show()


# """──────────────────────────────────────────────────────
#     print_summary_table()
#     Prints final comparison table — this becomes Table II
#     in your report, extended across road classes
# ──────────────────────────────────────────────────────"""
# def print_summary_table(all_results):
#     print('\n')
#     print('=' * 80)
#     print('  RESULTS SUMMARY TABLE')
#     print('  % change relative to passive suspension (negative = improvement)')
#     print('=' * 80)

#     header = f"{'Road':<8} {'Algorithm':<10} {'Mean Vel%':>10} {'Q3 Vel%':>10} "
#     header += f"{'Mean Acc%':>10} {'Q3 Acc%':>10}"
#     print(header)
#     print('-' * 80)

#     for road_class in ROAD_CLASSES:
#         for algo in ['DDPG', 'TD3']:
#             m = all_results[road_class][algo]
#             row = (f"  {road_class:<6} {algo:<10} "
#                    f"{m['mean_vel_%']:>10.2f} "
#                    f"{m['q3_vel_%']:>10.2f} "
#                    f"{m['mean_acc_%']:>10.2f} "
#                    f"{m['q3_acc_%']:>10.2f}")
#             print(row)
#         print('-' * 40)

#     print('=' * 80)
#     print()


# """──────────────────────────────────────────────────────
#     MAIN EVALUATION LOOP
# ──────────────────────────────────────────────────────"""
# def main():
#     # ── Load trained models ──────────────────────────────────
#     print('Loading trained models...')
#     try:
#         ddpg_mu = torch.load(DDPG_MU_PATH, map_location=device, weights_only=False)
#         ddpg_mu.eval()
#         print(f'  DDPG Actor loaded from: {DDPG_MU_PATH}')
#     except FileNotFoundError:
#         print(f'  WARNING: DDPG model not found at {DDPG_MU_PATH}')
#         print('  Run execute_training.py with ALGORITHM="DDPG" first.')
#         ddpg_mu = None

#     try:
#         td3_mu = torch.load(TD3_MU_PATH, map_location=device, weights_only=False)
#         td3_mu.eval()
#         print(f'  TD3 Actor loaded from: {TD3_MU_PATH}')
#     except FileNotFoundError:
#         print(f'  WARNING: TD3 model not found at {TD3_MU_PATH}')
#         print('  Run execute_training.py with ALGORITHM="TD3" first.')
#         td3_mu = None

#     if ddpg_mu is None and td3_mu is None:
#         print('No models found. Please train at least one model first.')
#         return

#     # ── Run evaluation for each road class ──────────────────
#     all_results = {}

#     for road_class in ROAD_CLASSES:
#         print(f'\n{"="*50}')
#         print(f'  Testing on ISO 8608 Class-{road_class} Road')
#         print(f'{"="*50}')

#         # Accumulate results over multiple runs for reliability
#         p_dxb_runs  = [];  p_d2xb_runs  = []
#         d_dxb_runs  = [];  d_d2xb_runs  = []
#         t_dxb_runs  = [];  t_d2xb_runs  = []

#         for run in range(NUM_TEST_RUNS):
#             # Generate fresh road for each run
#             road = list(
#                 RoadProfile().get_profile_by_class(road_class, t_stop, dt)[1][1:]
#             )

#             # Run passive
#             p_xb, p_dxb, p_d2xb = run_passive(road)
#             p_dxb_runs.append(p_dxb);  p_d2xb_runs.append(p_d2xb)

#             # Run DDPG
#             if ddpg_mu is not None:
#                 d_xb, d_dxb, d_d2xb, d_cs, d_ks = run_rl(ddpg_mu, road)
#                 d_dxb_runs.append(d_dxb);  d_d2xb_runs.append(d_d2xb)

#             # Run TD3
#             if td3_mu is not None:
#                 t_xb, t_dxb, t_d2xb, t_cs, t_ks = run_rl(td3_mu, road)
#                 t_dxb_runs.append(t_dxb);  t_d2xb_runs.append(t_d2xb)

#         # Average across runs
#         p_dxb_avg  = np.array(p_dxb_runs).mean(axis=0)
#         p_d2xb_avg = np.array(p_d2xb_runs).mean(axis=0)

#         all_results[road_class] = {}

#         if ddpg_mu is not None:
#             d_dxb_avg  = np.array(d_dxb_runs).mean(axis=0)
#             d_d2xb_avg = np.array(d_d2xb_runs).mean(axis=0)
#             metrics_ddpg = compute_metrics(p_dxb_avg, p_d2xb_avg, d_dxb_avg, d_d2xb_avg)
#             all_results[road_class]['DDPG'] = metrics_ddpg
#             print(f'  DDPG Results (Class {road_class}):')
#             for k, v in metrics_ddpg.items():
#                 print(f'    {k}: {v}%')

#         if td3_mu is not None:
#             t_dxb_avg  = np.array(t_dxb_runs).mean(axis=0)
#             t_d2xb_avg = np.array(t_d2xb_runs).mean(axis=0)
#             metrics_td3  = compute_metrics(p_dxb_avg, p_d2xb_avg, t_dxb_avg, t_d2xb_avg)
#             all_results[road_class]['TD3']  = metrics_td3
#             print(f'  TD3 Results (Class {road_class}):')
#             for k, v in metrics_td3.items():
#                 print(f'    {k}: {v}%')

#         # Generate comparison plot for this road class
#         # (uses last run's data for plotting)
#         if ddpg_mu is not None and td3_mu is not None:
#             plot_road_comparison(
#                 road_class, road,
#                 p_xb,  p_dxb_avg,  p_d2xb_avg,
#                 d_xb,  d_dxb_avg,  d_d2xb_avg,
#                 t_xb,  t_dxb_avg,  t_d2xb_avg,
#                 d_cs,  d_ks,
#                 t_cs,  t_ks
#             )

#     # ── Print final summary table ────────────────────────────
#     print_summary_table(all_results)

#     # ── Save all numerical results ───────────────────────────
#     np.save('results_summary.npy', all_results)
#     print('All results saved to: results_summary.npy')

#     # ── Reward curve comparison plot ─────────────────────────
#     try:
#         ddpg_rewards = np.load('Reward_DDPG.npy')
#         td3_rewards  = np.load('Reward_TD3.npy')

#         plt.figure(figsize=(12, 5))
#         plt.title('Training Reward Curve: DDPG vs TD3', fontsize=14)
#         plt.plot(ddpg_rewards, color='blue',  alpha=0.4, label='DDPG (raw)')
#         plt.plot(td3_rewards,  color='red',   alpha=0.4, label='TD3  (raw)')

#         # Moving average (window=20) for cleaner visualisation
#         window = 20
#         ddpg_ma = np.convolve(ddpg_rewards, np.ones(window)/window, mode='valid')
#         td3_ma  = np.convolve(td3_rewards,  np.ones(window)/window, mode='valid')
#         plt.plot(ddpg_ma, color='blue', linewidth=2, label=f'DDPG (MA-{window})')
#         plt.plot(td3_ma,  color='red',  linewidth=2, label=f'TD3  (MA-{window})')

#         plt.xlabel('Episode', fontsize=12)
#         plt.ylabel('Total Reward', fontsize=12)
#         plt.legend(fontsize=11)
#         plt.grid(True, linestyle='--', alpha=0.5)
#         plt.tight_layout()
#         plt.savefig('reward_curve_comparison.png', dpi=150)
#         print('Reward curve saved to: reward_curve_comparison.png')
#         plt.show()

#     except FileNotFoundError:
#         print('Reward files not found — skipping reward curve plot.')
#         print('(Train both DDPG and TD3 to generate this plot)')


# if __name__ == '__main__':
#     main()

"""
evaluation.py — DDPG vs SAC comparison
Tests both models across ISO 8608 classes B, C, D, E.
All three systems (passive, DDPG, SAC) run on the
same road instance per run for fair comparison.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from config import *
from road_generator import *
from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DDPG_MU_PATH  = './checkpoints/DDPG_mu_origin.pt'
SAC_ACTOR_PATH = './checkpoints/SAC_actor.pt'

ROAD_CLASSES  = ['B', 'C', 'D', 'E']
NUM_TEST_RUNS = 10
EVAL_SEED     = 123
np.random.seed(EVAL_SEED)


def get_cs(a): return cb + 600*a
def get_ks(a):
    ka = 5000*a if a > 0 else 2500*a
    return kb + ka


def make_odefun(road):
    def odefun(t, y0, m_, cs, kw_, ks):
        m1_=m_[0]; m2_=m_[1]
        idx = min(round(t/dt), len(TIME)-1)
        xr  = road[idx]
        xb=y0[0]; xw=y0[1]; dxb=y0[2]; dxw=y0[3]
        d2xb = -ks/m1_*(xb-xw) - cs/m1_*(dxb-dxw)
        d2xw =  ks/m2_*(xb-xw) + cs/m2_*(dxb-dxw) + kw_/m2_*(xr-xw)
        return [dxb, dxw, d2xb, d2xw]
    return odefun


def run_passive(road):
    fn    = make_odefun(road)
    s_ode = [road[0], road[0], 0, 0]
    xb_l, dxb_l, d2xb_l = [], [], []
    for t_idx, t in enumerate(TIME):
        xb,xw,dxb,dxw = s_ode
        d2xb = (-cb*(dxb-dxw)-kb*(xb-xw))/m1
        xb_l.append(xb); dxb_l.append(dxb); d2xb_l.append(d2xb)
        yout  = solve_ivp(fn,[t,t+dt],s_ode,
                          args=([m1,m2],cb,kw,kb),dense_output=True)
        s_ode = list(yout.y[:,-1])
    return np.array(xb_l), np.array(dxb_l), np.array(d2xb_l)


def run_ddpg(mu_model, road):
    """DDPG uses deterministic PolicyNet — no noise at evaluation."""
    fn         = make_odefun(road)
    s_ode      = [road[0], road[0], 0, 0]
    s_ode_prev = s_ode[:]
    xb_l, dxb_l, d2xb_l, cs_l = [], [], [], []

    for t_idx, t in enumerate(TIME):
        xb,xw,dxb,dxw = s_ode
        dxr      = (road[t_idx]-road[max(t_idx-1,0)])/dt
        _,_,dxb_p,dxw_p = s_ode_prev
        dxr_prev = (road[max(t_idx-1,0)]-road[max(t_idx-2,0)])/dt
        s_rl = [dxb,dxw,dxr,dxb_p,dxw_p,dxr_prev]

        with torch.no_grad():
            sb = torch.tensor(np.expand_dims(np.array(s_rl),0),
                              dtype=torch.float).to(device)
            a  = mu_model(sb).squeeze(1).cpu().numpy()[0]
        a_cs, a_ks = a[0], a[1]

        d2xb = (-get_cs(a_cs)*(dxb-dxw)-get_ks(a_ks)*(xb-xw))/m1
        xb_l.append(xb); dxb_l.append(dxb)
        d2xb_l.append(d2xb); cs_l.append(get_cs(a_cs))

        yout  = solve_ivp(fn,[t,t+dt],s_ode,
                          args=([m1,m2],get_cs(a_cs),kw,get_ks(a_ks)),
                          dense_output=True)
        s_ode_prev = s_ode[:]
        s_ode      = list(yout.y[:,-1])

    return np.array(xb_l),np.array(dxb_l),np.array(d2xb_l),np.array(cs_l)


def run_sac(actor_model, road):
    """
    SAC uses mean action (deterministic) at evaluation.
    This gives the best learned policy without exploration noise.
    """
    fn         = make_odefun(road)
    s_ode      = [road[0], road[0], 0, 0]
    s_ode_prev = s_ode[:]
    xb_l, dxb_l, d2xb_l, cs_l = [], [], [], []

    for t_idx, t in enumerate(TIME):
        xb,xw,dxb,dxw = s_ode
        dxr      = (road[t_idx]-road[max(t_idx-1,0)])/dt
        _,_,dxb_p,dxw_p = s_ode_prev
        dxr_prev = (road[max(t_idx-1,0)]-road[max(t_idx-2,0)])/dt
        s_rl = [dxb,dxw,dxr,dxb_p,dxw_p,dxr_prev]

        with torch.no_grad():
            sb = torch.tensor(np.expand_dims(np.array(s_rl),0),
                              dtype=torch.float).to(device)
            # Use mean action — deterministic at eval
            _, _, mean_action = actor_model.sample(sb)
            a = mean_action.squeeze(0).cpu().numpy()
        a_cs, a_ks = a[0], a[1]

        d2xb = (-get_cs(a_cs)*(dxb-dxw)-get_ks(a_ks)*(xb-xw))/m1
        xb_l.append(xb); dxb_l.append(dxb)
        d2xb_l.append(d2xb); cs_l.append(get_cs(a_cs))

        yout  = solve_ivp(fn,[t,t+dt],s_ode,
                          args=([m1,m2],get_cs(a_cs),kw,get_ks(a_ks)),
                          dense_output=True)
        s_ode_prev = s_ode[:]
        s_ode      = list(yout.y[:,-1])

    return np.array(xb_l),np.array(dxb_l),np.array(d2xb_l),np.array(cs_l)


def compute_metrics(p_dxb, p_d2xb, r_dxb, r_d2xb):
    def pct(r,p): return round((r-p)/abs(p)*100, 2)
    pv=np.abs(p_dxb); rv=np.abs(r_dxb)
    pa=np.abs(p_d2xb); ra=np.abs(r_d2xb)
    return {
        'mean_vel_%': pct(rv.mean(),             pv.mean()),
        'q3_vel_%':   pct(np.quantile(rv,0.75),  np.quantile(pv,0.75)),
        'mean_acc_%': pct(ra.mean(),             pa.mean()),
        'q3_acc_%':   pct(np.quantile(ra,0.75),  np.quantile(pa,0.75)),
    }


def plot_comparison(rc, road,
                    p_xb,  p_dxb,  p_d2xb,
                    d_xb,  d_dxb,  d_d2xb, d_cs,
                    s_xb,  s_dxb,  s_d2xb, s_cs):

    fig = plt.figure(figsize=(18,12))
    fig.suptitle(f'ISO 8608 Class-{rc}  |  Passive vs DDPG vs SAC',
                 fontsize=15, fontweight='bold')
    gs = gridspec.GridSpec(3,2,figure=fig,hspace=0.42,wspace=0.32)

    a0 = fig.add_subplot(gs[0,:])
    a0.plot(road[1:len(p_xb)],color='black',lw=1.2)
    a0.set(title=f'Road Excitation $x_r$ — Class {rc}',
           xlabel='Time (×10⁻² s)',ylabel='Height (m)')
    a0.grid(ls='--',alpha=0.4)

    cfgs = [
        (fig.add_subplot(gs[1,0]), p_xb,   d_xb,   s_xb,   'Body Displacement $x_b$ (m)'),
        (fig.add_subplot(gs[1,1]), p_dxb,  d_dxb,  s_dxb,  'Body Velocity $\\dot{x}_b$ (m/s)'),
        (fig.add_subplot(gs[2,0]), p_d2xb, d_d2xb, s_d2xb, 'Body Acceleration $\\ddot{x}_b$ (m/s²)'),
    ]
    for ax,p,d,s,title in cfgs:
        ax.plot(p[1:],'-.', color='gray', lw=1.2, label='Passive')
        ax.plot(d[1:],'--', color='blue', lw=1.2, label='DDPG')
        ax.plot(s[1:],      color='green',lw=1.8, label='SAC')
        ax.set(title=title, xlabel='Time (×10⁻² s)')
        ax.legend(fontsize=9); ax.grid(ls='--',alpha=0.4)

    ax4 = fig.add_subplot(gs[2,1])
    ax4.plot(d_cs[1:],'--',color='blue', lw=1.0,label='DDPG $C_a$')
    ax4.plot(s_cs[1:],     color='green',lw=1.0,label='SAC $C_a$')
    ax4.set(title='Controlled Damping $C_a$ (Ns/m)',
            xlabel='Time (×10⁻² s)')
    ax4.legend(fontsize=9); ax4.grid(ls='--',alpha=0.4)

    fname = f'comparison_class_{rc}.png'
    plt.savefig(fname,dpi=150,bbox_inches='tight')
    print(f'  → Saved: {fname}')
    plt.close()


def print_table(all_results):
    print('\n'+'='*74)
    print('  RESULTS: % change vs passive  (negative = improvement)')
    print('='*74)
    print(f"{'Road':<6}{'Algorithm':<10}"
          f"{'MeanVel%':>10}{'Q3Vel%':>9}"
          f"{'MeanAcc%':>10}{'Q3Acc%':>9}")
    print('-'*74)
    for rc in ROAD_CLASSES:
        for algo in ['DDPG','SAC']:
            if algo not in all_results.get(rc,{}): continue
            m = all_results[rc][algo]
            print(f"  {rc:<4}{algo:<10}"
                  f"{m['mean_vel_%']:>10.2f}"
                  f"{m['q3_vel_%']:>9.2f}"
                  f"{m['mean_acc_%']:>10.2f}"
                  f"{m['q3_acc_%']:>9.2f}")
        print('-'*38)
    print('='*74+'\n')


def main():
    print('Loading models...')
    ddpg_mu, sac_actor = None, None

    try:
        ddpg_mu = torch.load(DDPG_MU_PATH, map_location=device,
                             weights_only=False)
        ddpg_mu.eval()
        print(f'  DDPG loaded: {DDPG_MU_PATH}')
    except FileNotFoundError:
        print(f'  DDPG not found')

    try:
        sac_actor = torch.load(SAC_ACTOR_PATH, map_location=device,
                               weights_only=False)
        sac_actor.eval()
        print(f'  SAC  loaded: {SAC_ACTOR_PATH}')
    except FileNotFoundError:
        print(f'  SAC not found')

    if ddpg_mu is None and sac_actor is None:
        print('No models. Train first.'); return

    all_results = {}

    for rc in ROAD_CLASSES:
        print(f'\n{"="*50}')
        print(f'  Class-{rc}  ({NUM_TEST_RUNS} runs, same road per run)')
        print(f'{"="*50}')

        p_dxb_r,p_d2xb_r = [],[]
        d_dxb_r,d_d2xb_r = [],[]
        s_dxb_r,s_d2xb_r = [],[]
        last = {}

        for run in range(NUM_TEST_RUNS):
            # All systems share the same road
            road = list(RoadProfile().get_profile_by_class(
                rc, t_stop, dt)[1][1:])

            p_xb,p_dxb,p_d2xb = run_passive(road)
            p_dxb_r.append(p_dxb); p_d2xb_r.append(p_d2xb)

            if ddpg_mu is not None:
                d_xb,d_dxb,d_d2xb,d_cs = run_ddpg(ddpg_mu,road)
                d_dxb_r.append(d_dxb); d_d2xb_r.append(d_d2xb)
                last.update(d_xb=d_xb,d_dxb=d_dxb,
                            d_d2xb=d_d2xb,d_cs=d_cs)

            if sac_actor is not None:
                s_xb,s_dxb,s_d2xb,s_cs = run_sac(sac_actor,road)
                s_dxb_r.append(s_dxb); s_d2xb_r.append(s_d2xb)
                last.update(s_xb=s_xb,s_dxb=s_dxb,
                            s_d2xb=s_d2xb,s_cs=s_cs,
                            road=road,p_xb=p_xb,
                            p_dxb=p_dxb,p_d2xb=p_d2xb)

        p_dxb_avg  = np.mean(p_dxb_r,  axis=0)
        p_d2xb_avg = np.mean(p_d2xb_r, axis=0)
        all_results[rc] = {}

        if ddpg_mu is not None and d_dxb_r:
            d_avg  = np.mean(d_dxb_r,  axis=0)
            da_avg = np.mean(d_d2xb_r, axis=0)
            md = compute_metrics(p_dxb_avg,p_d2xb_avg,d_avg,da_avg)
            all_results[rc]['DDPG'] = md
            print(f'  DDPG — Class {rc}:')
            for k,v in md.items(): print(f'    {k}: {v}%')

        if sac_actor is not None and s_dxb_r:
            s_avg  = np.mean(s_dxb_r,  axis=0)
            sa_avg = np.mean(s_d2xb_r, axis=0)
            ms = compute_metrics(p_dxb_avg,p_d2xb_avg,s_avg,sa_avg)
            all_results[rc]['SAC'] = ms
            print(f'  SAC  — Class {rc}:')
            for k,v in ms.items(): print(f'    {k}: {v}%')

        if ddpg_mu is not None and sac_actor is not None and last:
            plot_comparison(rc,last['road'],
                last['p_xb'],last['p_dxb'],last['p_d2xb'],
                last['d_xb'],last['d_dxb'],last['d_d2xb'],last['d_cs'],
                last['s_xb'],last['s_dxb'],last['s_d2xb'],last['s_cs'])

    print_table(all_results)
    np.save('results_summary.npy', all_results)
    print('Saved → results_summary.npy')

    # Reward curves
    try:
        dr = np.load('Reward_DDPG.npy')
        sr = np.load('Reward_SAC.npy')
        w  = 20
        plt.figure(figsize=(12,5))
        plt.title('Training Reward: DDPG vs SAC', fontsize=13)
        plt.plot(dr, color='blue', alpha=0.2, label='DDPG raw')
        plt.plot(sr, color='green',alpha=0.2, label='SAC raw')
        plt.plot(np.convolve(dr,np.ones(w)/w,'valid'),
                 color='blue', lw=2, label=f'DDPG MA-{w}')
        plt.plot(np.convolve(sr,np.ones(w)/w,'valid'),
                 color='green',lw=2, label=f'SAC MA-{w}')
        plt.xlabel('Episode'); plt.ylabel('Total Reward')
        plt.legend(fontsize=11); plt.grid(ls='--',alpha=0.5)
        plt.tight_layout()
        plt.savefig('reward_curve_comparison.png',dpi=150)
        print('Reward curve → reward_curve_comparison.png')
        plt.close()
    except FileNotFoundError:
        print('Reward files not found — train both first.')


if __name__ == '__main__':
    main()