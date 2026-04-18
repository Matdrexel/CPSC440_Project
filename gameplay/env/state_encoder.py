"""
state_encoder.py — Converts a GameState into a flat numpy vector for DQN/PPO.

Vector layout (18 elements total):
  [0:2]   hole cards          — card index / 51, shape (2,)
  [2:7]   board cards         — card index / 51, or -1.0 if undealt, shape (5,)
  [7:11]  street one-hot      — [preflop, flop, turn, river], shape (4,)
  [11]    position            — 0.0=BB (OOP), 1.0=BTN (IP)
  [12]    my stack            — stack / starting_stack
  [13]    opponent stack      — stack / starting_stack
  [14]    pot                 — pot / starting_stack
  [15]    my bet this street  — bet / starting_stack
  [16]    opp bet this street — bet / starting_stack
  [17]    MC equity           — P(hero wins at showdown) via Monte Carlo, in [0, 1]

All chip values normalised by starting_stack so values stay in [0, 1].
Card values normalised to [0, 1] by dividing by 51.
Undealt board card slots use -1.0 as a sentinel (distinguishable from any
real card value which is >= 0).

MC equity uses your custom deck (infinite, with replacement) and hand
evaluator (straights > flushes, flush five, flush house, five of a kind)
so the equity values correctly reflect the unique rules of infinite holdem.

Usage:
    encoder = StateEncoder(mc_samples=1000)
    obs = encoder.encode(state, player_id=0)   # numpy array shape (18,)
    obs_size = encoder.obs_size                # 18
"""

import numpy as np
from env.game_engine import GameState, Street
from env.deck import deal_card
from env.hand_evaluator import check_hand


# ---------------------------------------------------------------------------
# Monte Carlo equity
# ---------------------------------------------------------------------------

def mc_equity(hole_cards: list[int],
              board: list[int],
              n_samples: int = 100) -> float:
    """
    Estimate P(hero wins at showdown) by Monte Carlo simulation.

    Uses the infinite deck (sampling with replacement) and your custom hand
    evaluator, so equity correctly accounts for:
      - Duplicate cards being possible (infinite deck)
      - Straights outranking flushes
      - Flush five, flush house, five of a kind hand types

    Args:
        hole_cards: hero's 2 hole cards (card indices 0-51)
        board:      community cards dealt so far (0-5 cards)
        n_samples:  number of Monte Carlo rollouts

    Returns:
        float in [0, 1] — fraction of simulations hero won outright.
        Ties are counted as 0.5 wins (equity-correct).
    """
    board_needed = 5 - len(board)
    wins = 0.0

    for _ in range(n_samples):
        # Deal villain 2 hole cards from the infinite deck (with replacement)
        villain_hole = [deal_card(), deal_card()]

        # Complete the board with the remaining community cards
        runout    = [deal_card() for _ in range(board_needed)]
        full_board = board + runout

        hero_hand    = check_hand(hole_cards   + full_board)
        villain_hand = check_hand(villain_hole + full_board)

        if hero_hand > villain_hand:
            wins += 1.0
        elif hero_hand == villain_hand:
            wins += 0.5   # chop — equity-correct split

    return wins / n_samples


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------

class StateEncoder:
    """
    Encodes a GameState into a flat float32 numpy observation vector
    from the perspective of a given player.
    """

    BOARD_SLOTS = 5

    def __init__(self, mc_samples: int = 100):
        """
        Args:
            mc_samples: Monte Carlo rollouts per encode() call for equity
                        estimation. 1000 gives ~1% std dev, adds ~1ms/step.
        """
        self.mc_samples = mc_samples
        # 2 hole + 5 board + 4 street + 1 position + 5 chip + 1 equity
        self.obs_size = 2 + self.BOARD_SLOTS + 4 + 1 + 5 + 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, state: GameState, player_id: int) -> np.ndarray:
        """
        Encode `state` from the perspective of `player_id` (0 or 1).
        Returns a float32 numpy array of shape (obs_size,) = (18,).
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
        norm        = float(state.starting_stack)
        opponent_id = 1 - player_id

        obs[idx] = state.stacks[player_id]   / norm;  idx += 1
        obs[idx] = state.stacks[opponent_id] / norm;  idx += 1
        obs[idx] = state.pot                 / norm;  idx += 1
        obs[idx] = state.bets[player_id]     / norm;  idx += 1
        obs[idx] = state.bets[opponent_id]   / norm;  idx += 1

        # --- Monte Carlo equity (1) -----------------------------------
        obs[idx] = mc_equity(
            state.hole_cards[player_id],
            state.board,
            self.mc_samples,
        )
        idx += 1

        assert idx == self.obs_size, \
            f"Encoder wrote {idx} values, expected {self.obs_size}"
        return obs

    def encode_both(self, state: GameState) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: encode from both players' perspectives at once."""
        return self.encode(state, 0), self.encode(state, 1)

    def feature_names(self) -> list[str]:
        """Human-readable names for each index — useful for debugging."""
        names  = [f"hole_{i}" for i in range(2)]
        names += [f"board_{i}" for i in range(self.BOARD_SLOTS)]
        names += ["street_preflop", "street_flop", "street_turn", "street_river"]
        names += ["position"]
        names += ["my_stack", "opp_stack", "pot", "my_bet", "opp_bet"]
        names += ["mc_equity"]
        return names