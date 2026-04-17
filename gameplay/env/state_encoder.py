"""
state_encoder.py — Converts a GameState into a flat numpy vector for DQN/PPO.

Vector layout (17 elements total):
  [0:2]   hole cards          — card index / 51, shape (2,)
  [2:7]   board cards         — card index / 51, or -1.0 if undealt, shape (5,)
  [7:11]  street one-hot      — [preflop, flop, turn, river], shape (4,)
  [11]    position            — 0.0=BB (OOP), 1.0=BTN (IP)
  [12]    my stack            — stack / starting_stack
  [13]    opponent stack      — stack / starting_stack
  [14]    pot                 — pot / starting_stack
  [15]    my bet this street  — bet / starting_stack
  [16]    opp bet this street — bet / starting_stack

All chip values normalised by starting_stack so values stay in [0, 1].
Card values normalised to [0, 1] by dividing by 51.
Undealt board card slots use -1.0 as a sentinel (distinguishable from any
real card value which is >= 0).

Usage:
    encoder = StateEncoder(num_raise_sizes=3)
    obs = encoder.encode(state, player_id=0)   # numpy array shape (17,)
    obs_size = encoder.obs_size                 # 17
"""

import numpy as np
from env.game_engine import GameState, Street


class StateEncoder:
    """
    Encodes a GameState into a flat float32 numpy observation vector
    from the perspective of a given player.
    """

    # Fixed board slots regardless of how many cards are dealt
    BOARD_SLOTS = 5

    def __init__(self):
        # 2 hole + 5 board + 4 street + 1 position + 5 chip features
        self.obs_size = 2 + self.BOARD_SLOTS + 4 + 1 + 5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, state: GameState, player_id: int) -> np.ndarray:
        """
        Encode `state` from the perspective of `player_id` (0 or 1).
        Returns a float32 numpy array of shape (obs_size,) = (17,).
        """
        obs = np.empty(self.obs_size, dtype=np.float32)
        idx = 0

        # --- Hole cards (2) -------------------------------------------
        for card in state.hole_cards[player_id]:
            obs[idx] = card / 51.0
            idx += 1

        # --- Board cards (5, padded with -1 for undealt slots) --------
        for i in range(self.BOARD_SLOTS):
            if i < len(state.board):
                obs[idx] = state.board[i] / 51.0
            else:
                obs[idx] = -1.0
            idx += 1

        # --- Street one-hot (4) ----------------------------------------
        for s in range(4):
            obs[idx] = 1.0 if state.street == s else 0.0
            idx += 1

        # --- Position (1) ---------------------------------------------
        # Player 0 = BTN/SB = in position postflop → 1.0
        # Player 1 = BB     = out of position      → 0.0
        obs[idx] = 1.0 if player_id == 0 else 0.0
        idx += 1

        # --- Chip features (5) ----------------------------------------
        norm = float(state.starting_stack)
        opponent_id = 1 - player_id

        obs[idx] = state.stacks[player_id] / norm;     idx += 1
        obs[idx] = state.stacks[opponent_id] / norm;   idx += 1
        obs[idx] = state.pot / norm;                   idx += 1
        obs[idx] = state.bets[player_id] / norm;       idx += 1
        obs[idx] = state.bets[opponent_id] / norm;     idx += 1

        assert idx == self.obs_size, f"Encoder wrote {idx} values, expected {self.obs_size}"
        return obs

    def encode_both(self, state: GameState) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: encode from both players' perspectives at once."""
        return self.encode(state, 0), self.encode(state, 1)

    def feature_names(self) -> list[str]:
        """Human-readable names for each index — useful for debugging."""
        names = []
        names += [f"hole_{i}" for i in range(2)]
        names += [f"board_{i}" for i in range(self.BOARD_SLOTS)]
        names += ["street_preflop", "street_flop", "street_turn", "street_river"]
        names += ["position"]
        names += ["my_stack", "opp_stack", "pot", "my_bet", "opp_bet"]
        return names