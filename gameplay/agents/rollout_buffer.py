"""
rollout_buffer.py — Stores self-play trajectories and computes GAE advantages.

Each hand produces a sequence of (obs, action, log_prob, reward, value, mask)
tuples — one per decision point. Since rewards are sparse (only non-zero at
hand end), GAE correctly propagates the terminal signal back through all steps.

GAE (Generalised Advantage Estimation):
    delta_t   = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t       = sum_{l=0}^{T} (gamma * lambda)^l * delta_t+l

    Returns   = A_t + V(s_t)
"""

import numpy as np
import torch


class RolloutBuffer:
    """
    Accumulates experience from one or more complete hands, then provides
    mini-batches for PPO updates.

    All tensors are shaped (n_steps, ...) where n_steps is the total number
    of decision points collected before calling compute_advantages().
    """

    def __init__(self,
                 obs_size: int,
                 n_actions: int,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = "cpu"):
        self.obs_size   = obs_size
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = device
        self._reset_lists()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def add(self,
            obs:      np.ndarray,   # (obs_size,)
            action:   int,
            log_prob: float,        # log π(a|s) at collection time
            reward:   float,        # r_t (in BB, zero for non-terminal steps)
            value:    float,        # V(s_t) from critic
            done:     bool,
            mask:     np.ndarray):  # (n_actions,) bool
        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._rewards.append(reward)
        self._values.append(value)
        self._dones.append(done)
        self._masks.append(mask)

    def size(self) -> int:
        return len(self._actions)

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def compute_advantages(self, last_value: float = 0.0):
        """
        Compute GAE advantages and returns in-place.
        Call once after collecting a full batch of experience, passing
        last_value=0 if the final step was terminal (done=True).
        """
        n = self.size()
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        values = np.array(self._values, dtype=np.float32)
        rewards = np.array(self._rewards, dtype=np.float32)
        dones = np.array(self._dones, dtype=np.float32)

        # Bootstrap from last_value for incomplete episodes
        next_value = last_value
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae   = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            next_value = values[t]

        self._advantages = advantages
        self._returns    = advantages + values   # V_target for critic loss

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def get_batches(self, batch_size: int):
        """
        Yield randomised mini-batches as torch tensors.
        Must call compute_advantages() first.
        """
        n = self.size()
        indices = np.random.permutation(n)

        # Convert everything to tensors once
        obs      = torch.tensor(np.array(self._obs),      dtype=torch.float32).to(self.device)
        actions  = torch.tensor(self._actions,             dtype=torch.long   ).to(self.device)
        log_probs= torch.tensor(self._log_probs,           dtype=torch.float32).to(self.device)
        returns  = torch.tensor(self._returns,             dtype=torch.float32).to(self.device)
        masks    = torch.tensor(np.array(self._masks),     dtype=torch.bool   ).to(self.device)

        # Normalise advantages over the whole batch (reduces variance)
        adv = torch.tensor(self._advantages, dtype=torch.float32).to(self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield (obs[idx], actions[idx], log_probs[idx],
                   returns[idx], adv[idx], masks[idx])

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def clear(self):
        """Reset buffer after a PPO update."""
        self._reset_lists()
        self._advantages = None
        self._returns    = None

    def _reset_lists(self):
        self._obs       = []
        self._actions   = []
        self._log_probs = []
        self._rewards   = []
        self._values    = []
        self._dones     = []
        self._masks     = []
        self._advantages = None
        self._returns    = None