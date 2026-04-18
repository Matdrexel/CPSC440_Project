"""
bc_trainer.py — Behavioural cloning pretraining for the PPO agent.

Trains the ActorCriticNet to imitate actions from the CSV dataset using
cross-entropy loss on the policy head. The value head is ignored during BC
(it will be trained from scratch during PPO self-play).

After pretraining, the weights are saved in two forms:
  1. A full checkpoint loadable by PPOAgent.load() — used to warm-start PPO.
  2. A frozen "reference policy" checkpoint loaded by BCPolicy — used as the
     KL anchor during PPO fine-tuning.

Both files are identical; they are saved separately so the reference policy
can be kept frozen while the PPO agent's weights diverge.

Run standalone:
    python bc_trainer.py --csv hands.csv --epochs 20 --batch_size 256
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split

from agents.bc_dataset import CSVConfig, PokerCSVDataset
from agents.ppo_agent import PPOAgent


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BCTrainer:
    """
    Pretrains a PPOAgent's network on demonstration data via cross-entropy.

    Args:
        agent      — PPOAgent whose network will be trained in-place
        dataset    — PokerCSVDataset
        val_frac   — fraction of data held out for validation (default 0.1)
        device     — "cpu" or "cuda"
    """

    def __init__(self,
                 agent:    PPOAgent,
                 dataset:  PokerCSVDataset,
                 val_frac: float = 0.1,
                 device:   str   = "cpu"):
        self.agent   = agent
        self.device  = device

        # Train / validation split
        n_val   = max(1, int(len(dataset) * val_frac))
        n_train = len(dataset) - n_val
        self.train_set, self.val_set = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"BC dataset: {n_train} train / {n_val} val samples")

        self.loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self,
              epochs:      int   = 10,
              batch_size:  int   = 256,
              lr:          float = 1e-3,
              patience:    int   = 5,
              save_dir:    str   = "checkpoints") -> PPOAgent:
        """
        Run BC pretraining.

        Args:
            epochs      — maximum number of full passes over the training set
            batch_size  — mini-batch size
            lr          — learning rate (separate from PPO lr; typically higher)
            patience    — early stopping: halt if val loss doesn't improve for
                          this many epochs
            save_dir    — where to write bc_pretrained.pt and bc_reference.pt

        Returns the trained agent (same object, mutated in-place).
        """
        os.makedirs(save_dir, exist_ok=True)

        optimizer  = torch.optim.Adam(self.agent.net.parameters(), lr=lr)
        train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(
            self.val_set,   batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        print(f"\nStarting BC pretraining — {epochs} epochs, batch_size={batch_size}, lr={lr}")
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val Acc':>9} {'Time':>7}")
        print("-" * 52)

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ── Training pass ──────────────────────────────────────────
            self.agent.net.train()
            train_losses = []
            for obs_b, mask_b, act_b in train_loader:
                obs_b  = obs_b.to(self.device)
                mask_b = mask_b.to(self.device)
                act_b  = act_b.to(self.device)

                dist, _ = self.agent.net(obs_b, mask_b)
                # dist.logits are already masked; cross-entropy over full logit vector
                loss = self.loss_fn(dist.logits, act_b)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.net.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # ── Validation pass ────────────────────────────────────────
            val_loss, val_acc = self._evaluate(val_loader)
            mean_train_loss   = float(np.mean(train_losses))
            elapsed           = time.time() - t0

            print(f"{epoch:>6}  {mean_train_loss:>12.4f}  {val_loss:>10.4f}"
                  f"  {val_acc:>8.1%}  {elapsed:>6.1f}s")

            # ── Early stopping ─────────────────────────────────────────
            if val_loss < best_val_loss - 1e-4:
                best_val_loss     = val_loss
                epochs_no_improve = 0
                best_state        = {k: v.clone() for k, v in
                                     self.agent.net.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                    break

        # Restore best weights
        if best_state is not None:
            self.agent.net.load_state_dict(best_state)
            print(f"Restored best weights (val_loss={best_val_loss:.4f})")

        # ── Save both checkpoints ──────────────────────────────────────
        warm_start_path = os.path.join(save_dir, "bc_pretrained.pt")
        reference_path  = os.path.join(save_dir, "bc_reference.pt")

        self.agent.save(warm_start_path)
        self.agent.save(reference_path)

        print(f"\nSaved warm-start weights : {warm_start_path}")
        print(f"Saved reference policy   : {reference_path}")
        print("(Both files are identical at this point — "
              "bc_pretrained.pt will be updated by PPO; bc_reference.pt stays frozen.)")

        return self.agent

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, loader) -> tuple[float, float]:
        """Returns (mean_loss, accuracy) on the provided DataLoader."""
        self.agent.net.eval()
        losses, correct, total = [], 0, 0
        for obs_b, mask_b, act_b in loader:
            obs_b  = obs_b.to(self.device)
            mask_b = mask_b.to(self.device)
            act_b  = act_b.to(self.device)

            dist, _ = self.agent.net(obs_b, mask_b)
            loss    = self.loss_fn(dist.logits, act_b)
            losses.append(loss.item())

            preds    = dist.logits.argmax(dim=-1)
            correct += (preds == act_b).sum().item()
            total   += len(act_b)

        return float(np.mean(losses)), correct / total


# ---------------------------------------------------------------------------
# Reference policy (frozen BC snapshot used during PPO KL penalty)
# ---------------------------------------------------------------------------

class BCPolicy:
    """
    Frozen copy of the BC-pretrained network. Used during PPO fine-tuning
    to compute KL( π_bc || π_current ) as a regularisation term.

    The network weights are permanently frozen — gradients never flow through
    this object.
    """

    def __init__(self, checkpoint_path: str, obs_size: int, n_actions: int,
                 hidden_size: int = 128, device: str = "cpu"):
        from ppo_agent import ActorCriticNet
        self.device = device
        self.net = ActorCriticNet(obs_size, n_actions, hidden_size).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.net.load_state_dict(checkpoint["net_state"])

        # Permanently freeze
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

        print(f"Loaded frozen BC reference policy from {checkpoint_path}")

    @torch.no_grad()
    def log_probs(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Return log π_bc(a | obs) for all actions.
        Shape: (batch, n_actions)
        """
        dist, _ = self.net(obs, mask)
        # log-softmax over masked logits
        return torch.log_softmax(dist.logits, dim=-1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Behavioural cloning pretraining")

    # Data
    p.add_argument("--csv",          type=str,   required=True,
                   help="Path to CSV file of decision points")
    p.add_argument("--separator",    type=str,   default=",")
    p.add_argument("--val_frac",     type=float, default=0.1)

    # CSV column names (override defaults to match your file)
    p.add_argument("--col_hole1",       type=str, default="hole_card_1")
    p.add_argument("--col_hole2",       type=str, default="hole_card_2")
    p.add_argument("--col_board",       type=str, nargs="*",
                   default=["board_1","board_2","board_3","board_4","board_5"])
    p.add_argument("--col_street",      type=str, default="street")
    p.add_argument("--col_position",    type=str, default="position")
    p.add_argument("--col_stack_hero",  type=str, default="stack_hero")
    p.add_argument("--col_stack_villain",type=str,default="stack_villain")
    p.add_argument("--col_pot",         type=str, default="pot")
    p.add_argument("--col_bet_hero",    type=str, default="bet_hero")
    p.add_argument("--col_bet_villain", type=str, default="bet_villain")
    p.add_argument("--col_action",      type=str, default="action")
    p.add_argument("--starting_stack",  type=int, default=1000,
                   help="Scalar starting stack, or pass a column name via --col_starting_stack")
    p.add_argument("--col_starting_stack", type=str, default=None)
    p.add_argument("--raise_sizes",     type=float, nargs="+", default=[0.5, 1.0, 2.0])

    # Training
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=5)
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--save_dir",    type=str,   default="checkpoints")
    p.add_argument("--cpu",         action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    starting_stack = args.col_starting_stack if args.col_starting_stack else args.starting_stack

    config = CSVConfig(
        hole_card_1    = args.col_hole1,
        hole_card_2    = args.col_hole2,
        board_cards    = args.col_board,
        street         = args.col_street,
        position       = args.col_position,
        stack_hero     = args.col_stack_hero,
        stack_villain  = args.col_stack_villain,
        pot            = args.col_pot,
        bet_hero       = args.col_bet_hero,
        bet_villain    = args.col_bet_villain,
        action         = args.col_action,
        starting_stack = starting_stack,
        raise_sizes    = args.raise_sizes,
        separator      = args.separator,
    )

    dataset = PokerCSVDataset(args.csv, config)
    agent   = PPOAgent(
        obs_size    = dataset.obs_size,
        n_actions   = dataset.n_actions,
        hidden_size = args.hidden_size,
        device      = device,
    )

    trainer = BCTrainer(agent, dataset, val_frac=args.val_frac, device=device)
    trainer.train(
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        patience   = args.patience,
        save_dir   = args.save_dir,
    )