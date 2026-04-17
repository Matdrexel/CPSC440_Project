"""
ppo_agent.py — PPO agent with shared trunk + separate policy & value heads.

Network architecture:
    Input (17,)
        → Linear(17 → 128) → ReLU
        → Linear(128 → 128) → ReLU      [shared trunk]
              ↙                    ↘
    Linear(128 → n_actions)    Linear(128 → 1)
      [policy head]              [value head]

Action masking:
    Illegal actions receive logit = -1e9 before softmax, making their
    probability effectively 0. The entropy bonus is computed only over
    legal actions so it doesn't reward exploring illegal moves.

PPO update:
    L = L_policy + c_value * L_value - c_entropy * L_entropy
    L_policy  = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
    L_value   = 0.5 * MSE(V(s), returns)
    L_entropy = mean entropy of masked action distribution
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCriticNet(nn.Module):
    """Shared-trunk network with separate policy and value heads."""

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head  = nn.Linear(hidden_size, 1)

        # Initialise policy head with small weights for near-uniform initial policy
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight,  gain=1.0)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            obs:  (batch, obs_size) float32
            mask: (batch, n_actions) bool — True for legal actions

        Returns:
            dist:  Categorical distribution over legal actions
            value: (batch, 1) estimated state value
        """
        features = self.trunk(obs)

        # Apply mask: set illegal action logits to -inf
        logits = self.policy_head(features)
        logits = logits.masked_fill(~mask, -1e9)

        dist  = Categorical(logits=logits)
        value = self.value_head(features)
        return dist, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Value-only forward pass (used during rollout collection)."""
        features = self.trunk(obs)
        return self.value_head(features)


class PPOAgent:
    """
    PPO agent for heads-up poker self-play.

    Hyperparameters:
        clip_eps    — PPO clipping epsilon (default 0.2)
        c_value     — value loss coefficient (default 0.5)
        c_entropy   — entropy bonus coefficient (default 0.01)
        lr          — Adam learning rate (default 3e-4)
        n_epochs    — number of update epochs per rollout (default 4)
        batch_size  — mini-batch size for each epoch (default 64)
        max_grad_norm — gradient clipping norm (default 0.5)
    """

    def __init__(self,
                 obs_size:      int,
                 n_actions:     int,
                 hidden_size:   int   = 128,
                 lr:            float = 3e-4,
                 clip_eps:      float = 0.2,
                 c_value:       float = 0.5,
                 c_entropy:     float = 0.01,
                 n_epochs:      int   = 4,
                 batch_size:    int   = 64,
                 max_grad_norm: float = 0.5,
                 device:        str   = "cpu"):

        self.clip_eps      = clip_eps
        self.c_value       = c_value
        self.c_entropy     = c_entropy
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.max_grad_norm = max_grad_norm
        self.device        = device

        self.net = ActorCriticNet(obs_size, n_actions, hidden_size).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Inference (called during rollout collection)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray) -> tuple[int, float, float]:
        """
        Sample an action given an observation and legal action mask.

        Returns:
            action   — sampled action index
            log_prob — log π(action | obs) under current policy
            value    — V(obs) from critic
        """
        obs_t  = torch.tensor(obs,  dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_t = torch.tensor(mask, dtype=torch.bool   ).unsqueeze(0).to(self.device)

        dist, value = self.net(obs_t, mask_t)
        action      = dist.sample()
        log_prob    = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        """Estimate V(obs) without sampling an action."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.net.get_value(obs_t).item()

    # ------------------------------------------------------------------
    # PPO update (called after collecting a full rollout buffer)
    # ------------------------------------------------------------------

    def update(self, buffer) -> dict[str, float]:
        """
        Run n_epochs of PPO updates over the rollout buffer.

        Args:
            buffer — RolloutBuffer with compute_advantages() already called

        Returns:
            dict of mean losses for logging:
                policy_loss, value_loss, entropy, total_loss, approx_kl
        """
        metrics = dict(policy_loss=[], value_loss=[],
                       entropy=[], total_loss=[], approx_kl=[])

        for _ in range(self.n_epochs):
            for obs_b, act_b, old_log_prob_b, returns_b, adv_b, mask_b \
                    in buffer.get_batches(self.batch_size):

                # Forward pass
                dist, values = self.net(obs_b, mask_b)
                new_log_prob = dist.log_prob(act_b)
                entropy      = dist.entropy().mean()

                # Probability ratio r_t = π_new / π_old
                ratio = torch.exp(new_log_prob - old_log_prob_b)

                # Clipped surrogate policy loss
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((values.squeeze(-1) - returns_b) ** 2).mean()

                # Combined loss
                loss = policy_loss + self.c_value * value_loss - self.c_entropy * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Logging
                with torch.no_grad():
                    approx_kl = ((old_log_prob_b - new_log_prob) ** 2).mean() * 0.5

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["total_loss"].append(loss.item())
                metrics["approx_kl"].append(approx_kl.item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "net_state":       self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["net_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])