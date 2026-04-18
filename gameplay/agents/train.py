"""
train.py — Self-play PPO training loop for heads-up poker.

Training flow per iteration:
  1. Collect `steps_per_update` decision points via self-play
     (same agent plays both seats simultaneously)
  2. Compute GAE advantages on the collected rollout
  3. Run PPO update for n_epochs over the rollout
  4. Log metrics, optionally save checkpoint

Since it's self-play, a single PPOAgent instance acts for both players.
Each player's perspective is correctly encoded by StateEncoder before
being passed to the agent, so the agent learns a position-aware policy.

Rewards are in units of BB (big blinds) — a reward of +1.0 means the
acting player won 1 big blind on that hand.

Run:
    python train.py
    python train.py --total_steps 2000000 --lr 1e-4 --save_every 50
"""

import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from env.poker_env import PokerEnv
from agents.ppo_agent import PPOAgent
from agents.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(env: PokerEnv,
                    agent: PPOAgent,
                    buffer: RolloutBuffer,
                    steps_per_update: int) -> dict:
    """
    Play hands until we have collected `steps_per_update` decision points.
    Since this is self-play, the same agent acts for both players.

    Returns summary stats: hands_played, mean_reward_per_hand
    """
    buffer.clear()

    steps_collected = 0
    hands_played    = 0
    hand_rewards    = []   # total reward (in BB) per hand, from player 0's perspective

    obs, mask = env.reset()

    while steps_collected < steps_per_update:
        # Agent acts for whoever is currently sitting in the hot seat
        action, log_prob, value = agent.act(obs, mask)
        next_obs, reward, done, next_mask = env.step(action)

        buffer.add(
            obs      = obs,
            action   = action,
            log_prob = log_prob,
            reward   = reward,
            value    = value,
            done     = done,
            mask     = mask,
        )
        steps_collected += 1

        if done:
            hands_played += 1
            hand_rewards.append(env.state.rewards[0] / env.big_blind)
            obs, mask = env.reset()
        else:
            obs, mask = next_obs, next_mask

    # Bootstrap value for the last incomplete trajectory
    last_value = 0.0 if done else agent.get_value(obs)
    buffer.compute_advantages(last_value=last_value)

    return {
        "hands_played":        hands_played,
        "mean_reward_bb":      float(np.mean(hand_rewards)) if hand_rewards else 0.0,
        "steps_collected":     steps_collected,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # --- Setup -----------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Training on {device}")

    env = PokerEnv(
        starting_stack = args.starting_stack,
        small_blind    = args.small_blind,
        big_blind      = args.big_blind,
        raise_sizes    = args.raise_sizes,
    )

    # --- BC reference policy (optional) ----------------------------------
    bc_policy = None
    if args.bc_reference is not None:
        from bc_trainer import BCPolicy
        bc_policy = BCPolicy(
            checkpoint_path = args.bc_reference,
            obs_size        = env.obs_size,
            n_actions       = env.n_actions,
            hidden_size     = args.hidden_size,
            device          = device,
        )

    agent = PPOAgent(
        obs_size           = env.obs_size,
        n_actions          = env.n_actions,
        hidden_size        = args.hidden_size,
        lr                 = args.lr,
        clip_eps           = args.clip_eps,
        c_value            = args.c_value,
        c_entropy          = args.c_entropy,
        n_epochs           = args.n_epochs,
        batch_size         = args.batch_size,
        max_grad_norm      = args.max_grad_norm,
        device             = device,
        bc_policy          = bc_policy,
        bc_kl_coeff        = args.bc_kl_coeff,
        bc_kl_decay_steps  = args.bc_kl_decay_steps,
    )

    # Load BC pretrained weights if provided
    if args.bc_checkpoint is not None:
        agent.load(args.bc_checkpoint)
        print(f"Loaded BC pretrained weights from {args.bc_checkpoint}")

    buffer = RolloutBuffer(
        obs_size   = env.obs_size,
        n_actions  = env.n_actions,
        gamma      = args.gamma,
        gae_lambda = args.gae_lambda,
        device     = device,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    reward_window = deque(maxlen=100)
    loss_window   = deque(maxlen=20)

    total_steps  = 0
    update_count = 0
    start_time   = time.time()

    bc_status = ""
    if args.bc_checkpoint:
        bc_status = f"  bc_checkpoint={args.bc_checkpoint}, kl_coeff={args.bc_kl_coeff}, decay_steps={args.bc_kl_decay_steps}"

    print(f"\nStarting self-play PPO training")
    print(f"  obs_size={env.obs_size}  n_actions={env.n_actions}")
    print(f"  steps_per_update={args.steps_per_update}  total_steps={args.total_steps}")
    print(f"  lr={args.lr}  clip_eps={args.clip_eps}  n_epochs={args.n_epochs}")
    if bc_status:
        print(bc_status)
    print()

    # --- Main loop -------------------------------------------------------
    while total_steps < args.total_steps:
        rollout_stats = collect_rollout(env, agent, buffer, args.steps_per_update)
        total_steps  += rollout_stats["steps_collected"]
        update_count += 1

        reward_window.append(rollout_stats["mean_reward_bb"])
        update_metrics = agent.update(buffer)
        loss_window.append(update_metrics["total_loss"])

        if update_count % args.log_every == 0:
            elapsed       = time.time() - start_time
            steps_per_sec = total_steps / elapsed
            smooth_reward = np.mean(reward_window)
            smooth_loss   = np.mean(loss_window)
            beta          = update_metrics.get("bc_beta", 0.0)
            bc_kl         = update_metrics.get("bc_kl_loss", 0.0)

            bc_str = f" | bc_beta: {beta:.3f} | bc_kl: {bc_kl:.4f}" if bc_policy else ""
            print(
                f"[update {update_count:5d} | steps {total_steps:8d}] "
                f"reward/hand: {smooth_reward:+.3f} BB | "
                f"loss: {smooth_loss:.4f} | "
                f"policy: {update_metrics['policy_loss']:.4f} | "
                f"value: {update_metrics['value_loss']:.4f} | "
                f"entropy: {update_metrics['entropy']:.4f} | "
                f"kl: {update_metrics['approx_kl']:.4f}"
                f"{bc_str} | "
                f"{steps_per_sec:.0f} steps/s"
            )

        if update_count % args.save_every == 0:
            path = os.path.join(args.save_dir, f"checkpoint_{update_count:05d}.pt")
            agent.save(path)
            print(f"  → saved checkpoint: {path}")

    final_path = os.path.join(args.save_dir, "final.pt")
    agent.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Self-play PPO poker training")

    # Environment
    p.add_argument("--starting_stack", type=int,   default=1000)
    p.add_argument("--small_blind",    type=int,   default=5)
    p.add_argument("--big_blind",      type=int,   default=10)
    p.add_argument("--raise_sizes",    type=float, nargs="+", default=[0.5, 1.0, 2.0])

    # Training
    p.add_argument("--total_steps",      type=int,   default=1_000_000)
    p.add_argument("--steps_per_update", type=int,   default=2048)
    p.add_argument("--gamma",            type=float, default=0.99)
    p.add_argument("--gae_lambda",       type=float, default=0.95)

    # PPO
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--clip_eps",      type=float, default=0.2)
    p.add_argument("--c_value",       type=float, default=0.5)
    p.add_argument("--c_entropy",     type=float, default=0.05)
    p.add_argument("--n_epochs",      type=int,   default=4)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    # Network
    p.add_argument("--hidden_size", type=int, default=256)

    # BC regularisation
    p.add_argument("--bc_checkpoint",     type=str,   default=None,
                   help="Path to bc_pretrained.pt from bc_trainer.py (warm-starts PPO weights)")
    p.add_argument("--bc_reference",      type=str,   default=None,
                   help="Path to bc_reference.pt — frozen policy used for KL penalty. "
                        "Defaults to bc_checkpoint if not set separately.")
    p.add_argument("--bc_kl_coeff",       type=float, default=1.0,
                   help="Initial KL penalty coefficient beta_0")
    p.add_argument("--bc_kl_decay_steps", type=int,   default=500_000,
                   help="Steps over which KL penalty decays linearly to zero")

    # Misc
    p.add_argument("--save_dir",   type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=50,
                   help="Save checkpoint every N updates")
    p.add_argument("--log_every",  type=int, default=10,
                   help="Print log every N updates")
    p.add_argument("--cpu",        action="store_true",
                   help="Force CPU even if CUDA is available")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Default bc_reference to bc_checkpoint if not separately specified
    if args.bc_checkpoint is not None and args.bc_reference is None:
        args.bc_reference = args.bc_checkpoint
    train(args)