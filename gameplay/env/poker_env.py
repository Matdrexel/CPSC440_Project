"""
poker_env.py — Gym-style wrapper around GameEngine + StateEncoder.

Presents a single unified interface for the PPO training loop.
Since we use self-play, the environment manages both seats internally
and always returns the observation / mask from the perspective of
whoever is currently acting.

Observation:  float32 numpy array, shape (17,)
Action:       int in [0, n_actions)
Reward:       float, net chips / big_blind  (zero-sum, only non-zero at hand end)
Done:         bool, True when the hand is over
"""

import numpy as np
from env.game_engine import GameEngine, Action
from env.state_encoder import StateEncoder


class PokerEnv:
    def __init__(self,
                 starting_stack: int = 1000,
                 small_blind: int = 5,
                 big_blind: int = 10,
                 raise_sizes: list[float] = None):
        self.big_blind = big_blind
        self.engine = GameEngine(
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            raise_sizes=raise_sizes or [0.5, 1.0, 2.0],
        )
        self.encoder = StateEncoder()
        self.n_actions = 2 + len(self.engine.raise_sizes)
        self.obs_size  = self.encoder.obs_size
        self.state     = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Start a new hand.
        Returns (obs, action_mask) for the first acting player.
        """
        self.state = self.engine.reset()
        return self._observe()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, np.ndarray]:
        """
        Apply action for the current acting player.

        Returns:
            obs         — observation for the NEXT acting player (float32, shape (17,))
            reward      — reward in BB for the player who just acted (0 until hand ends)
            done        — True if the hand is over
            action_mask — legal action mask for the next acting player
        """
        acting_player = self.state.acting_player
        self.state = self.engine.step(action)

        if self.state.hand_over:
            # Return terminal reward normalised by big blind
            reward = self.state.rewards[acting_player] / self.big_blind
            # Observation and mask are meaningless at terminal step —
            # return zeros so the caller doesn't have to special-case shape
            obs  = np.zeros(self.obs_size, dtype=np.float32)
            mask = np.ones(self.n_actions, dtype=bool)
            return obs, reward, True, mask

        reward = 0.0
        obs, mask = self._observe()
        return obs, reward, False, mask

    def current_player(self) -> int:
        """Which player is currently acting (0 or 1)."""
        return self.state.acting_player

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        player = self.state.acting_player
        obs  = self.encoder.encode(self.state, player)
        mask = self.engine.legal_actions_mask()
        return obs, mask