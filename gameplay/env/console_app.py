import numpy as np
import torch

from env.poker_env import PokerEnv
from env.deck import hand_name
from agents.ppo_agent import PPOAgent


# --------------------------------------------------
# Helper: pretty print actions
# --------------------------------------------------

def action_to_string(action: int, raise_sizes):
    if action == 0:
        return "Fold"
    elif action == 1:
        return "Call/Check"
    else:
        size = raise_sizes[action - 2]
        return f"Raise ({size}x pot)"


# --------------------------------------------------
# Helper: print state (basic)
# --------------------------------------------------

def print_state(env: PokerEnv):
    state = env.state

    print("\n" + "=" * 40)
    print(f"Pot: {state.pot}")
    print(f"Stacks: P0={state.stacks[0]} | P1={state.stacks[1]}")
    print(f"Board: {hand_name(state.board)}")
    print(f"Acting Player: {state.acting_player}")
    print("=" * 40)


# --------------------------------------------------
# Human action selection
# --------------------------------------------------

def get_human_action(mask, raise_sizes, state, player_num):
    legal_actions = [i for i, m in enumerate(mask) if m]

    print(f"\nYour hand: {hand_name(state.hole_cards[player_num])}")
    print("\nYour options:")
    for a in legal_actions:
        print(f"{a}: {action_to_string(a, raise_sizes)}")

    while True:
        try:
            choice = int(input("Choose action: "))
            if choice in legal_actions:
                return choice
        except:
            pass
        print("Invalid action. Try again.")


# --------------------------------------------------
# Main play loop
# --------------------------------------------------

def play(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Setup env ---
    env = PokerEnv()
    raise_sizes = env.engine.raise_sizes

    # --- Setup agent ---
    agent = PPOAgent(
        obs_size=env.obs_size,
        n_actions=env.n_actions,
        device=device
    )
    agent.load(model_path)
    agent.net.eval()

    print("\nLoaded model:", model_path)

    # Choose seat
    human_player = int(input("Play as player 0 or 1? (0/1): "))

    while True:
        obs, mask = env.reset()
        done = False

        print("\n=== New Hand ===")

        while not done:
            print_state(env)
            current_player = env.current_player()

            if current_player == human_player:
                action = get_human_action(mask, raise_sizes, env.state, human_player)
                print(f"You chose: {action_to_string(action, raise_sizes)}")
            else:
                action, _, _ = agent.act(obs, mask)
                print(f"Bot chose: {action_to_string(action, raise_sizes)}")

            obs, reward, done, mask = env.step(action)

        # Hand over
        print("\n=== Hand Over ===")
        print(f"\nBoard: {hand_name(env.state.board)}")
        print(f"\nBots hand: {hand_name(env.state.hole_cards[1 - human_player])}")
        print(f"Final rewards (BB): {env.state.rewards}")

        again = input("Play another hand? (y/n): ")
        if again.lower() != "y":
            break


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    play("checkpoints/final.pt")  # <-- change path if needed