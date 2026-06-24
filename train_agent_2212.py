"""
train_agent_2212.py
=================
PPO agent for LunarLander-v3 — tuned for avg 300+.

Key changes from v2:
  - HIDDEN_SIZE = 256      : more capacity for fine-grained control
  - Linear LR decay        : replaces ReduceLROnPlateau which collapsed LR too early
  - N_STEPS = 8192         : larger rollouts = more stable gradients at high performance
  - Entropy annealing      : 0.01 → 0.001 linearly, so agent can exploit late in training
  - MAX_TIMESTEPS = 5M     : agent was still improving at 3M end
  - SOLVE_REWARD = 325     : don't stop until genuinely hitting 325 greedy avg
  - my_policy.py unchanged : save file uses 256 hidden, update HIDDEN_SIZE there too
"""

import os
import time
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical

# ============================================================
# 1.  Hyper-parameters
# ============================================================
HIDDEN_SIZE     = 256      # ← increased from 128; more capacity for fine control
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.15     # tighter clip for stable high-perf training
C_VALUE         = 0.5
C_ENTROPY_START = 0.01     # ← annealed down to 0.001 over training
C_ENTROPY_END   = 0.001
LR_START        = 3e-4     # ← linear decay to LR_END (no plateau scheduler)
LR_END          = 1e-5
N_STEPS         = 8192     # ← doubled; more stable gradient estimates
BATCH_SIZE      = 256      # ← scaled with N_STEPS
K_EPOCHS        = 10
MAX_TIMESTEPS   = 5_000_000  # ← more budget
SOLVE_REWARD    = 325.0
EVAL_WINDOW     = 100
LOG_INTERVAL    = 20
SAVE_THRESHOLD  = 200.0    # only start greedy-eval saving after this


# ============================================================
# 2.  Actor Network
# ============================================================
class Actor(nn.Module):
    """
    Architecture:
        Input(8)
          -> Linear(8->256) -> LayerNorm(256) -> GELU
          -> Linear(256->256) -> LayerNorm(256) -> GELU
          -> Linear(256->4)

    Parameter count:
        Linear0.weight   : (256,  8) =  2048
        Linear0.bias     : (256,)    =   256
        LayerNorm0.weight: (256,)    =   256
        LayerNorm0.bias  : (256,)    =   256
        Linear1.weight   : (256,256) = 65536
        Linear1.bias     : (256,)    =   256
        LayerNorm1.weight: (256,)    =   256
        LayerNorm1.bias  : (256,)    =   256
        Linear2.weight   : (  4,256) =  1024
        Linear2.bias     : (  4,)    =     4
        ─────────────────────────────────────
        Total                        = 70148
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4,
                 hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linears[:-1]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(linears[-1].weight, gain=0.01)
        nn.init.constant_(linears[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def to_numpy_flat(self) -> np.ndarray:
        return np.concatenate([
            p.detach().cpu().numpy().flatten()
            for p in self.parameters()
        ])


# ============================================================
# 3.  Critic Network
# ============================================================
class Critic(nn.Module):
    def __init__(self, state_dim: int = 8, hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linears[:-1]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(linears[-1].weight, gain=1.0)
        nn.init.constant_(linears[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================
# 4.  PPO Agent
# ============================================================
class PPOAgent:
    def __init__(self, state_dim: int = 8, action_dim: int = 4,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.actor  = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=LR_START, eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR_START, eps=1e-5)

    def set_lr(self, lr: float):
        """Manually set LR for both optimizers (used by linear decay)."""
        for opt in [self.actor_opt, self.critic_opt]:
            for pg in opt.param_groups:
                pg["lr"] = lr

    @torch.no_grad()
    def select_action(self, state: np.ndarray):
        s      = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.actor(s)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), self.critic(s).item()

    def update(self, rollout: dict, c_entropy: float) -> dict:
        states        = rollout["states"]
        actions       = rollout["actions"]
        log_probs_old = rollout["log_probs_old"]
        returns       = rollout["returns"]
        advantages    = rollout["advantages"]
        old_values    = rollout["old_values"]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        p_sum = v_sum = e_sum = steps = 0
        n = states.size(0)

        for _ in range(K_EPOCHS):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, BATCH_SIZE):
                mb = idx[start: start + BATCH_SIZE]

                # Actor update
                logits    = self.actor(states[mb])
                dist      = Categorical(logits=logits)
                log_p_new = dist.log_prob(actions[mb])
                entropy   = dist.entropy()

                ratio  = torch.exp(log_p_new - log_probs_old[mb])
                mb_adv = advantages[mb]
                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                p_loss = -torch.min(surr1, surr2).mean()
                e_loss = -entropy.mean()

                self.actor_opt.zero_grad()
                (p_loss + c_entropy * e_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

                # Critic update
                values = self.critic(states[mb])
                v_clip = old_values[mb] + torch.clamp(
                    values - old_values[mb], -CLIP_EPS, CLIP_EPS)
                v_loss = 0.5 * torch.max(
                    (values - returns[mb]).pow(2),
                    (v_clip  - returns[mb]).pow(2)
                ).mean()

                self.critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()

                p_sum += p_loss.item()
                v_sum += v_loss.item()
                e_sum += e_loss.item()
                steps += 1

        return {
            "policy_loss": p_sum / steps,
            "value_loss":  v_sum / steps,
            "entropy":    -e_sum / steps,
        }


# ============================================================
# 5.  GAE
# ============================================================
def compute_gae(rewards, values, dones, last_value,
                gamma=GAMMA, lam=GAE_LAMBDA):
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae        = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        delta  = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
        gae    = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ============================================================
# 6.  Greedy evaluation (matches evaluate_agent.py exactly)
# ============================================================
@torch.no_grad()
def greedy_eval(actor: Actor, episodes: int = 20, device: str = "cpu") -> float:
    dev   = torch.device(device)
    total = 0.0
    for _ in range(episodes):
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        obs, _ = env.reset()
        done = False; ep_r = 0.0
        while not done:
            s      = torch.FloatTensor(obs).unsqueeze(0).to(dev)
            action = int(torch.argmax(actor(s), dim=-1).item())
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r; done = term or trunc
        env.close()
        total += ep_r
    return total / episodes


# ============================================================
# 7.  Training loop
# ============================================================
def train_and_save(filename: str):
    env   = gym.make("LunarLander-v3", render_mode="rgb_array")
    agent = PPOAgent()

    total_params = sum(p.numel() for p in agent.actor.parameters())

    print("=" * 70)
    print("PPO Agent v3 — LunarLander-v3  (targeting avg 300+)")
    print(f"  Hidden size   : {HIDDEN_SIZE}  |  Actor params: {total_params:,}")
    print(f"  LR schedule   : linear {LR_START:.0e} → {LR_END:.0e} over {MAX_TIMESTEPS/1e6:.0f}M steps")
    print(f"  Entropy       : annealed {C_ENTROPY_START} → {C_ENTROPY_END}")
    print(f"  N_STEPS       : {N_STEPS}  |  Batch: {BATCH_SIZE}  |  K_epochs: {K_EPOCHS}")
    print(f"  Clip ε        : {CLIP_EPS}  |  Gamma: {GAMMA}")
    print(f"  Budget        : {MAX_TIMESTEPS:,} steps")
    print(f"  Solve target  : greedy avg {SOLVE_REWARD:.0f}")
    # print(f"  Saving to     : {filename}  (best_policy.npy untouched)")
    print(f"  Saving to     : {filename}  (best_policy_2212.npy untouched)")
    print("=" * 70)

    total_steps   = 0
    episode_count = 0
    update_count  = 0
    best_avg      = -np.inf
    best_flat     = None
    reward_window = collections.deque(maxlen=EVAL_WINDOW)
    start_time    = time.time()

    s_buf = []; a_buf  = []; lp_buf = []
    r_buf = []; v_buf  = []; d_buf  = []

    state, _ = env.reset()
    ep_reward = 0.0

    while total_steps < MAX_TIMESTEPS:
        # ── Linear LR and entropy annealing ──────────────────────────────
        progress   = total_steps / MAX_TIMESTEPS          # 0.0 → 1.0
        current_lr = LR_START + (LR_END - LR_START) * progress
        c_entropy  = C_ENTROPY_START + (C_ENTROPY_END - C_ENTROPY_START) * progress
        agent.set_lr(current_lr)

        s_buf.clear(); a_buf.clear(); lp_buf.clear()
        r_buf.clear(); v_buf.clear(); d_buf.clear()

        # ── Phase 1: Collect N_STEPS transitions ──────────────────────────
        for _ in range(N_STEPS):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            s_buf.append(state);     a_buf.append(action)
            lp_buf.append(log_prob); r_buf.append(reward)
            v_buf.append(value);     d_buf.append(float(done))

            ep_reward   += reward
            total_steps += 1
            state        = next_state

            if done:
                episode_count += 1
                reward_window.append(ep_reward)

                if episode_count % LOG_INTERVAL == 0:
                    avg_reward = np.mean(reward_window)
                    elapsed    = time.time() - start_time
                    sps        = total_steps / elapsed
                    eta_min    = (MAX_TIMESTEPS - total_steps) / sps / 60
                    print(
                        f"Ep {episode_count:5d} | "
                        f"Steps {total_steps:8d} | "
                        f"Reward {ep_reward:8.2f} | "
                        f"Avg({EVAL_WINDOW}): {avg_reward:7.2f} | "
                        f"Best: {best_avg:7.2f} | "
                        f"LR: {current_lr:.1e} | "
                        f"Ent: {c_entropy:.4f} | "
                        f"ETA: {eta_min:.1f}m",
                        flush=True
                    )

                ep_reward = 0.0
                state, _ = env.reset()

        # ── Phase 2: GAE + PPO update ─────────────────────────────────────
        _, _, last_val = agent.select_action(state)
        advantages, returns = compute_gae(r_buf, v_buf, d_buf, last_val)

        dev     = agent.device
        rollout = {
            "states":        torch.from_numpy(np.array(s_buf,  np.float32)).to(dev),
            "actions":       torch.from_numpy(np.array(a_buf,  np.int64  )).to(dev),
            "log_probs_old": torch.from_numpy(np.array(lp_buf, np.float32)).to(dev),
            "old_values":    torch.from_numpy(np.array(v_buf,  np.float32)).to(dev),
            "returns":       torch.from_numpy(returns).to(dev),
            "advantages":    torch.from_numpy(advantages).to(dev),
        }
        agent.update(rollout, c_entropy)
        update_count += 1

        # ── Phase 3: Greedy eval every 5 updates ─────────────────────────
        avg_now = float(np.mean(reward_window)) if reward_window else -999.0
        if update_count % 5 == 0 and avg_now >= SAVE_THRESHOLD:
            greedy_avg = greedy_eval(agent.actor, episodes=20, device=str(dev))
            print(f"  [Greedy eval] {greedy_avg:.2f} over 20 eps  "
                  f"(train avg: {avg_now:.2f} | LR: {current_lr:.1e} | Ent: {c_entropy:.4f})",
                  flush=True)

            if greedy_avg > best_avg:
                best_avg  = greedy_avg
                best_flat = agent.actor.to_numpy_flat()
                np.save(filename, best_flat)
                print(f"  ✓ New best saved → {filename}  "
                      f"(greedy avg = {best_avg:.2f})", flush=True)

            if greedy_avg >= SOLVE_REWARD and episode_count >= EVAL_WINDOW:
                print(f"\n✓ Solved at episode {episode_count} "
                      f"(greedy avg = {greedy_avg:.2f})!")
                env.close()
                return

    env.close()
    if best_flat is not None:
        np.save(filename, best_flat)
    print(f"\nTraining complete.")
    print(f"Best greedy avg = {best_avg:.2f}")
    print(f"Best actor saved → {filename}")


# ============================================================
# 8.  Pure-NumPy policy  ← UPDATE my_policy.py HIDDEN_SIZE to 256
# ============================================================
def policy_action(params: np.ndarray, observation: np.ndarray) -> int:
    H   = HIDDEN_SIZE
    idx = 0

    W0    = params[idx:idx+H*8].reshape(H, 8); idx += H*8
    b0    = params[idx:idx+H];                 idx += H
    ln0_w = params[idx:idx+H];                 idx += H
    ln0_b = params[idx:idx+H];                 idx += H

    W1    = params[idx:idx+H*H].reshape(H, H); idx += H*H
    b1    = params[idx:idx+H];                 idx += H
    ln1_w = params[idx:idx+H];                 idx += H
    ln1_b = params[idx:idx+H];                 idx += H

    Wa    = params[idx:idx+4*H].reshape(4, H); idx += 4*H
    ba    = params[idx:idx+4]

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))

    def ln(x, w, b):
        m = x.mean(); v = x.var()
        return (x - m) / np.sqrt(v + 1e-5) * w + b

    h = gelu(ln(observation @ W0.T + b0, ln0_w, ln0_b))
    h = gelu(ln(h @ W1.T + b1,          ln1_w, ln1_b))
    return int(np.argmax(h @ Wa.T + ba))


def evaluate_policy_numpy(params: np.ndarray, episodes: int = 5,
                           render: bool = False) -> float:
    total = 0.0
    for _ in range(episodes):
        env = gym.make("LunarLander-v3",
                       render_mode="human" if render else "rgb_array")
        obs, _ = env.reset()
        done = False; ep_r = 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy_action(params, obs))
            ep_r += r; done = term or trunc
        env.close()
        total += ep_r
    return total / episodes


# ============================================================
# 9.  Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--play",     action="store_true")
    # parser.add_argument("--filename", type=str, default="policy_v3.npy")
    parser.add_argument("--filename", type=str, default="policy_2212.npy")
    args = parser.parse_args()

    if args.train:
        train_and_save(args.filename)
    elif args.play:
        if not os.path.exists(args.filename):
            print(f"File not found: {args.filename}")
        else:
            params = np.load(args.filename)
            avg = evaluate_policy_numpy(params, episodes=5, render=True)
            print(f"Average reward over 5 rendered episodes: {avg:.2f}")
    else:
        print("Use --train or --play.")
        # print("Example: python train_agent_v3.py --train --filename policy_v3.npy")
        print("Example: python train_agent_2212.py --train --filename policy_2212.npy")
