"""
bc_dataset.py — Configurable CSV parser and PyTorch Dataset for BC pretraining.

Your CSV has one row per decision point. You describe its columns via a
CSV_CONFIG dict and this module handles the rest: parsing, encoding each
row into a (obs, action_mask, action) tuple, and serving mini-batches.

─────────────────────────────────────────────────────────────────────────
QUICK START
─────────────────────────────────────────────────────────────────────────

from bc_dataset import CSVConfig, PokerCSVDataset

config = CSVConfig(
    hole_card_1   = "hole1",          # card index 0-51
    hole_card_2   = "hole2",
    board_cards   = ["board1", "board2", "board3", "board4", "board5"],
    street        = "street",         # int 0-3, or "preflop"/"flop"/"turn"/"river"
    position      = "position",       # 0=BB/OOP, 1=BTN/IP
    stack_hero    = "stack_hero",     # hero's stack (chips)
    stack_villain = "stack_villain",
    pot           = "pot",
    bet_hero      = "bet_hero",       # hero's bet this street
    bet_villain   = "bet_villain",
    action        = "action",         # 0=fold,1=call,2+=raise
    starting_stack = "starting_stack",# scalar OR column name
    raise_sizes   = [0.5, 1.0, 2.0], # must match engine config
    # Optional overrides:
    street_map    = None,             # e.g. {"preflop":0,"flop":1,"turn":2,"river":3}
    action_map    = None,             # e.g. {"fold":0,"call":1,"raise":2}
    separator     = ",",
    missing_board = -1.0,             # sentinel value for undealt board slots
)

dataset = PokerCSVDataset("hands.csv", config)
loader  = dataset.dataloader(batch_size=256, shuffle=True)
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

STREET_STRINGS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

@dataclass
class CSVConfig:
    # ── Required column names ──────────────────────────────────────────
    hole_card_1:    str   = "hole_card_1"   # card index 0-51
    hole_card_2:    str   = "hole_card_2"
    street:         str   = "street"        # 0-3 or string
    position:       str   = "position"      # 0=OOP/BB, 1=IP/BTN
    stack_hero:     str   = "stack_hero"
    stack_villain:  str   = "stack_villain"
    pot:            str   = "pot"
    bet_hero:       str   = "bet_hero"
    bet_villain:    str   = "bet_villain"
    action:         str   = "action"        # 0=fold,1=call,2+=raise

    # ── Board cards (up to 5 column names; use None for absent streets) -
    board_cards: list[str | None] = field(
        default_factory=lambda: ["board_1", "board_2", "board_3", "board_4", "board_5"]
    )

    # ── Starting stack: scalar or column name ──────────────────────────
    starting_stack: Union[int, float, str] = 1000

    # ── Raise sizes must match the engine / agent config ───────────────
    raise_sizes: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])

    # ── Optional string → int mappings ─────────────────────────────────
    # If your CSV has "preflop"/"flop"/"turn"/"river" set street_map=None
    # (auto-detected). Supply a custom dict only if you use different strings.
    street_map:  dict | None = None
    # If your CSV has "fold"/"call"/"raise" set action_map={"fold":0,"call":1,"raise":2}
    action_map:  dict | None = None

    # ── Parsing options ─────────────────────────────────────────────────
    separator:     str   = ","
    missing_board: float = -1.0   # obs value for undealt board card slots

    # ── Validation ──────────────────────────────────────────────────────
    def __post_init__(self):
        if len(self.board_cards) > 5:
            raise ValueError("board_cards can have at most 5 entries.")
        n_actions = 2 + len(self.raise_sizes)
        if n_actions < 2:
            raise ValueError("raise_sizes must be a non-empty list.")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PokerCSVDataset(Dataset):
    """
    Reads a CSV file and converts each row into a training sample:
        obs         — float32 numpy array (17,)
        action_mask — bool numpy array    (n_actions,)
        action      — int label           scalar

    All raises in the CSV are treated as legal for the mask (since the data
    comes from a human player who could have raised). If you want stricter
    masks you can override _build_mask().
    """

    # Observation layout (matches StateEncoder exactly)
    OBS_SIZE    = 17
    BOARD_SLOTS = 5

    def __init__(self, csv_path: str, config: CSVConfig):
        self.config    = config
        self.n_actions = 2 + len(config.raise_sizes)
        self._load(csv_path)

    # ------------------------------------------------------------------
    # Loading & preprocessing
    # ------------------------------------------------------------------

    def _load(self, csv_path: str):
        cfg = self.config
        df  = pd.read_csv(csv_path, sep=cfg.separator)

        # ── Resolve starting_stack ─────────────────────────────────────
        if isinstance(cfg.starting_stack, str):
            starting_stacks = df[cfg.starting_stack].values.astype(float)
        else:
            starting_stacks = np.full(len(df), float(cfg.starting_stack))

        # ── Resolve street ─────────────────────────────────────────────
        raw_street = df[cfg.street]
        if cfg.street_map is not None:
            streets = raw_street.map(cfg.street_map).values.astype(int)
        elif raw_street.dtype == object:
            streets = raw_street.str.lower().map(STREET_STRINGS).values.astype(int)
        else:
            streets = raw_street.values.astype(int)

        # ── Resolve action ─────────────────────────────────────────────
        raw_action = df[cfg.action]
        if cfg.action_map is not None:
            actions = raw_action.map(cfg.action_map).values.astype(int)
        elif raw_action.dtype == object:
            raise ValueError(
                "action column contains strings but no action_map was provided. "
                "Supply action_map={'fold':0,'call':1,'raise':2} (or similar) in CSVConfig."
            )
        else:
            actions = raw_action.values.astype(int)

        # ── Board cards ────────────────────────────────────────────────
        # Build (N, 5) array; columns absent from CSV or beyond street become -1
        board_arr = np.full((len(df), self.BOARD_SLOTS), cfg.missing_board, dtype=np.float32)
        for slot, col in enumerate(cfg.board_cards):
            if col is not None and col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(-1).values
                # -1 or NaN means undealt; normalise valid cards to [0,1]
                valid = vals >= 0
                board_arr[valid, slot] = vals[valid] / 51.0
                board_arr[~valid, slot] = cfg.missing_board

        # ── Scalar chip columns ────────────────────────────────────────
        stacks_hero    = df[cfg.stack_hero].values.astype(float)
        stacks_villain = df[cfg.stack_villain].values.astype(float)
        pots           = df[cfg.pot].values.astype(float)
        bets_hero      = df[cfg.bet_hero].values.astype(float)
        bets_villain   = df[cfg.bet_villain].values.astype(float)
        positions      = df[cfg.position].values.astype(float)
        hole1          = df[cfg.hole_card_1].values.astype(float) / 51.0
        hole2          = df[cfg.hole_card_2].values.astype(float) / 51.0

        # ── Assemble observation matrix (N, 17) ────────────────────────
        N    = len(df)
        norm = starting_stacks  # (N,) broadcast divisor

        # Street one-hot (N, 4)
        street_onehot = np.zeros((N, 4), dtype=np.float32)
        for i, s in enumerate(streets):
            if 0 <= s <= 3:
                street_onehot[i, s] = 1.0

        obs = np.column_stack([
            hole1,                        # [0]
            hole2,                        # [1]
            board_arr,                    # [2:7]
            street_onehot,                # [7:11]
            positions,                    # [11]
            stacks_hero    / norm,        # [12]
            stacks_villain / norm,        # [13]
            pots           / norm,        # [14]
            bets_hero      / norm,        # [15]
            bets_villain   / norm,        # [16]
        ]).astype(np.float32)

        assert obs.shape[1] == self.OBS_SIZE, \
            f"Built obs with {obs.shape[1]} features, expected {self.OBS_SIZE}"

        # ── Store ──────────────────────────────────────────────────────
        self.obs     = obs       # (N, 17)
        self.actions = actions   # (N,)   int
        self.masks   = self._build_masks(actions, N)  # (N, n_actions) bool

        print(f"Loaded {N} decision points from CSV.")
        print(f"  Action distribution: { {a: int((actions==a).sum()) for a in range(self.n_actions)} }")

    # ------------------------------------------------------------------
    # Action mask construction
    # ------------------------------------------------------------------

    def _build_masks(self, actions: np.ndarray, N: int) -> np.ndarray:
        """
        Default: fold and call are always legal; raise actions are legal
        if the recorded action was a raise (we can't know for sure which
        sizes were available, so we mark all raise slots legal when any
        raise occurred).

        Override this method in a subclass for stricter masking.
        """
        masks = np.zeros((N, self.n_actions), dtype=bool)
        masks[:, 0] = True   # fold always legal
        masks[:, 1] = True   # call always legal
        raise_occurred = actions >= 2
        masks[raise_occurred, 2:] = True   # all raise sizes legal if player raised
        return masks

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.obs[idx],   dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.bool),
            torch.tensor(self.actions[idx], dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def dataloader(self, batch_size: int = 256, shuffle: bool = True,
                   num_workers: int = 0) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers)

    @property
    def obs_size(self) -> int:
        return self.OBS_SIZE

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @n_actions.setter
    def n_actions(self, v: int):
        self._n_actions = v