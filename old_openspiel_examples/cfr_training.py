"""
cfr_training.py - Train a poker AI using Counterfactual Regret Minimization

CFR is the algorithm family that powered Libratus and Pluribus.
This file demonstrates:
- What CFR is and how it works
- Training CFR on Kuhn Poker
- Evaluating the learned strategy
- Visualizing the strategy
"""

import pyspiel
import numpy as np
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability


def explain_cfr():
    """
    Explain what CFR is and why it matters.
    """
    print("=" * 60)
    print("WHAT IS CFR (COUNTERFACTUAL REGRET MINIMIZATION)?")
    print("=" * 60)
    
    print("""
CFR is the algorithm family that created superhuman poker AI:
- Libratus (beat top pros in heads-up NLHE, 2017)
- Pluribus (beat pros in 6-player NLHE, 2019)

How it works (simplified):
1. Start with a random strategy
2. Play many simulated games against yourself (self-play)
3. Track "regret" - how much better you could have done
4. Update strategy to minimize future regret
5. Over time, converge to Nash Equilibrium

Nash Equilibrium means:
- Neither player can improve by changing their strategy alone
- The best response to a Nash strategy is another Nash strategy
- In poker: you can't be exploited if you play Nash

Why it works for poker:
- CFR handles imperfect information (hidden cards)
- Doesn't need to see opponent's cards to learn
- Converges to optimal strategy through self-play
""")


def train_cfr_on_kuhn_poker(iterations=1000):
    """
    Train CFR on Kuhn Poker and analyze the results.
    """
    print("\n" + "=" * 60)
    print(f"TRAINING CFR ON KUHN POKER ({iterations} iterations)")
    print("=" * 60)
    
    # Load game
    game = pyspiel.load_game("kuhn_poker")
    
    # Create CFR solver
    cfr_solver = cfr.CFRSolver(game)
    
    # Train
    print("\nTraining...")
    for i in range(iterations):
        cfr_solver.evaluate_and_update_policy()
        
        if (i + 1) % (iterations // 5) == 0:
            # Calculate exploitability (how far from Nash equilibrium)
            avg_policy = cfr_solver.average_policy()
            exp = exploitability.exploitability(game, avg_policy)
            print(f"  Iteration {i + 1}: exploitability = {exp:.6f}")
    
    print("\nTraining complete!")
    
    # Get the learned strategy
    avg_policy = cfr_solver.average_policy()
    
    return game, avg_policy


def analyze_strategy(game, policy):
    """
    Analyze and display the learned strategy.
    """
    print("\n" + "=" * 60)
    print("ANALYZING LEARNED STRATEGY")
    print("=" * 60)
    
    # Calculate final exploitability
    exp = exploitability.exploitability(game, policy)
    print(f"\nFinal exploitability: {exp:.6f}")
    print("(Lower = closer to Nash equilibrium, 0 = perfect)")
    
    print("\n--- Learned Strategy by Information State ---")
    print("\nFormat: [Pass probability, Bet probability]")
    print("(Information state shows: card held + action history)")
    
    # Key decision points in Kuhn Poker
    info_states = [
        # Player 0's information states (first to act)
        ("0", "Player 0 has Jack, acting first"),
        ("1", "Player 0 has Queen, acting first"),
        ("2", "Player 0 has King, acting first"),
        ("0pb", "Player 0 has Jack, passed then faced bet"),
        ("1pb", "Player 0 has Queen, passed then faced bet"),
        ("2pb", "Player 0 has King, passed then faced bet"),
        # Player 1's information states (second to act)
        ("0p", "Player 1 has Jack, opponent passed"),
        ("1p", "Player 1 has Queen, opponent passed"),
        ("2p", "Player 1 has King, opponent passed"),
        ("0b", "Player 1 has Jack, opponent bet"),
        ("1b", "Player 1 has Queen, opponent bet"),
        ("2b", "Player 1 has King, opponent bet"),
    ]
    
    print("\n" + "-" * 50)
    for info_state, description in info_states:
        try:
            probs = policy.action_probabilities(
                pyspiel.InfoStateNode(game, info_state)
            )
            print(f"\n{description}")
            print(f"  Info state: '{info_state}'")
            print(f"  Pass: {probs.get(0, 0):.3f}, Bet: {probs.get(1, 0):.3f}")
        except:
            pass
    
    print("\n" + "-" * 50)
    print("""
Optimal Kuhn Poker strategy highlights:
- With King: Always bet when possible (strong hand)
- With Jack: Usually pass, occasionally bluff (~1/3)
- With Queen: Mixed strategy (sometimes bet, sometimes check)
""")


def compare_strategies(game, policy):
    """
    Compare CFR strategy against random play.
    """
    print("\n" + "=" * 60)
    print("CFR STRATEGY VS RANDOM PLAY")
    print("=" * 60)
    
    num_games = 10000
    cfr_total = 0
    random_total = 0
    
    for _ in range(num_games):
        state = game.new_initial_state()
        
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
            else:
                current_player = state.current_player()
                legal_actions = state.legal_actions()
                
                if current_player == 0:
                    # CFR player
                    info_state = state.information_state_string(current_player)
                    try:
                        probs = policy.action_probabilities(
                            pyspiel.InfoStateNode(game, info_state)
                        )
                        action_probs = [probs.get(a, 0) for a in legal_actions]
                        action_probs = np.array(action_probs)
                        action_probs /= action_probs.sum()  # Normalize
                        action = np.random.choice(legal_actions, p=action_probs)
                    except:
                        action = np.random.choice(legal_actions)
                else:
                    # Random player
                    action = np.random.choice(legal_actions)
            
            state.apply_action(action)
        
        returns = state.returns()
        cfr_total += returns[0]
        random_total += returns[1]
    
    print(f"\nResults over {num_games} games:")
    print(f"CFR player (P0) average: {cfr_total / num_games:+.4f}")
    print(f"Random player (P1) average: {random_total / num_games:+.4f}")
    print("\nCFR should significantly outperform random play!")


def demonstrate_self_play(game, policy, num_games=5):
    """
    Show some example self-play games with the learned strategy.
    """
    print("\n" + "=" * 60)
    print(f"EXAMPLE SELF-PLAY GAMES")
    print("=" * 60)
    
    card_names = {0: "Jack", 1: "Queen", 2: "King"}
    action_names = {0: "Pass", 1: "Bet"}
    
    for game_num in range(num_games):
        print(f"\n--- Game {game_num + 1} ---")
        state = game.new_initial_state()
        p0_card = None
        p1_card = None
        
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
                
                if p0_card is None:
                    p0_card = action
                    print(f"Player 0 dealt: {card_names[action]}")
                else:
                    p1_card = action
                    print(f"Player 1 dealt: {card_names[action]}")
            else:
                current_player = state.current_player()
                legal_actions = state.legal_actions()
                info_state = state.information_state_string(current_player)
                
                try:
                    probs = policy.action_probabilities(
                        pyspiel.InfoStateNode(game, info_state)
                    )
                    action_probs = [probs.get(a, 0) for a in legal_actions]
                    action_probs = np.array(action_probs)
                    action_probs /= action_probs.sum()
                    action = np.random.choice(legal_actions, p=action_probs)
                except:
                    action = np.random.choice(legal_actions)
                
                print(f"Player {current_player}: {action_names[action]}")
            
            state.apply_action(action)
        
        returns = state.returns()
        winner = "Player 0" if returns[0] > 0 else "Player 1"
        print(f"Result: {winner} wins {abs(returns[0])}")


if __name__ == "__main__":
    # Explain CFR
    explain_cfr()
    
    # Train CFR
    game, policy = train_cfr_on_kuhn_poker(iterations=1000)
    
    # Analyze the learned strategy
    analyze_strategy(game, policy)
    
    # Compare against random
    compare_strategies(game, policy)
    
    # Show some example games
    demonstrate_self_play(game, policy, num_games=3)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
To apply CFR to larger poker variants:

1. Leduc Poker (in OpenSpiel):
   game = pyspiel.load_game("leduc_poker")
   
2. Full NLHE requires:
   - Card abstraction (group similar hands)
   - Action abstraction (limit bet sizes)
   - Distributed computing (very large game tree)
   
3. Check out these resources:
   - PokerKit for NLHE game simulation
   - The Pluribus paper for advanced techniques
   - PioSolver for commercial solver
""")
