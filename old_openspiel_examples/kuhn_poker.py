"""
kuhn_poker.py - Introduction to OpenSpiel with Kuhn Poker

Kuhn Poker is a simplified poker game perfect for learning AI algorithms:
- 3-card deck: Jack, Queen, King
- 2 players, each gets 1 card
- Single betting round
- Simple enough to solve completely, complex enough to be interesting

This file demonstrates:
- Loading a game in OpenSpiel
- Understanding game states
- Playing random games
- Exploring the game tree
"""

import pyspiel
import numpy as np


def explore_kuhn_poker():
    """
    Learn the basics of OpenSpiel with Kuhn Poker.
    """
    print("=" * 60)
    print("EXPLORING KUHN POKER IN OPENSPIEL")
    print("=" * 60)
    
    # Load the game
    game = pyspiel.load_game("kuhn_poker")
    
    print("\n--- Game Information ---")
    print(f"Game: {game}")
    print(f"Number of players: {game.num_players()}")
    print(f"Max game length: {game.max_game_length()}")
    print(f"Number of distinct actions: {game.num_distinct_actions()}")
    
    # In Kuhn Poker:
    # Action 0 = Pass (Check/Fold)
    # Action 1 = Bet (or Call)
    
    print("\n--- Action Meanings ---")
    print("Action 0: Pass (check if no bet, fold if facing bet)")
    print("Action 1: Bet (or call if facing bet)")
    
    return game


def play_random_game(game):
    """
    Play a random game of Kuhn Poker.
    """
    print("\n" + "=" * 60)
    print("PLAYING A RANDOM GAME")
    print("=" * 60)
    
    # Create initial state
    state = game.new_initial_state()
    
    print("\n--- Initial State ---")
    print(f"State: {state}")
    print(f"Is terminal: {state.is_terminal()}")
    print(f"Is chance node: {state.is_chance_node()}")
    
    step = 0
    while not state.is_terminal():
        step += 1
        print(f"\n--- Step {step} ---")
        
        if state.is_chance_node():
            # Chance node = dealing cards
            outcomes = state.chance_outcomes()
            print(f"Chance node - dealing card")
            print(f"Possible outcomes: {outcomes}")
            
            # Sample from chance outcomes
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            print(f"Dealt: {action} (0=Jack, 1=Queen, 2=King)")
            
        else:
            # Player decision node
            current_player = state.current_player()
            legal_actions = state.legal_actions()
            
            print(f"Player {current_player}'s turn")
            print(f"Information state: {state.information_state_string(current_player)}")
            print(f"Legal actions: {legal_actions}")
            
            # Choose random action
            action = np.random.choice(legal_actions)
            action_name = "Pass" if action == 0 else "Bet"
            print(f"Action chosen: {action} ({action_name})")
        
        # Apply the action
        state.apply_action(action)
    
    # Game is over
    print("\n--- Game Over ---")
    print(f"Final state: {state}")
    returns = state.returns()
    print(f"Returns: Player 0 = {returns[0]}, Player 1 = {returns[1]}")
    
    if returns[0] > returns[1]:
        print("Player 0 wins!")
    elif returns[1] > returns[0]:
        print("Player 1 wins!")
    else:
        print("Tie!")
    
    return returns


def understand_information_states(game):
    """
    Understand how information states work in imperfect information games.
    """
    print("\n" + "=" * 60)
    print("UNDERSTANDING INFORMATION STATES")
    print("=" * 60)
    
    print("""
In poker, players don't see each other's cards. This creates
"information states" - what a player knows from their perspective.

Two game states can look the same to a player (same information state)
even though the hidden cards are different.

In Kuhn Poker information states look like:
- "0" = Player has Jack (card 0)
- "1" = Player has Queen (card 1)  
- "2" = Player has King (card 2)
- "0pb" = Player has Jack, opponent passed, player bet
- "1b" = Player has Queen, opponent bet first
""")
    
    # Let's enumerate some states
    state = game.new_initial_state()
    
    # Deal Jack to P0, Queen to P1
    state.apply_action(0)  # P0 gets Jack
    state.apply_action(1)  # P1 gets Queen
    
    print("\n--- Example Game State ---")
    print(f"P0's information state: {state.information_state_string(0)}")
    print(f"P1's information state: {state.information_state_string(1)}")
    print("(P0 only knows they have Jack, P1 only knows they have Queen)")
    
    # P0 passes
    state.apply_action(0)  # Pass
    print("\nAfter P0 passes:")
    print(f"P1's information state: {state.information_state_string(1)}")
    
    # P1 bets
    state.apply_action(1)  # Bet
    print("\nAfter P1 bets:")
    print(f"P0's information state: {state.information_state_string(0)}")


def simulate_many_games(game, num_games=1000):
    """
    Simulate many random games and collect statistics.
    """
    print("\n" + "=" * 60)
    print(f"SIMULATING {num_games} RANDOM GAMES")
    print("=" * 60)
    
    p0_total = 0
    p1_total = 0
    
    for i in range(num_games):
        state = game.new_initial_state()
        
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
            else:
                legal_actions = state.legal_actions()
                action = np.random.choice(legal_actions)
            
            state.apply_action(action)
        
        returns = state.returns()
        p0_total += returns[0]
        p1_total += returns[1]
    
    print(f"\nResults after {num_games} games:")
    print(f"Player 0 average: {p0_total / num_games:.3f}")
    print(f"Player 1 average: {p1_total / num_games:.3f}")
    print("\n(With random play, both should be close to 0)")
    print("(In optimal play, Player 0 has slight disadvantage: ~-0.056)")


def list_available_poker_games():
    """
    Show what poker games are available in OpenSpiel.
    """
    print("\n" + "=" * 60)
    print("AVAILABLE POKER GAMES IN OPENSPIEL")
    print("=" * 60)
    
    poker_games = [
        ("kuhn_poker", "Simplest poker - 3 cards, 1 betting round"),
        ("leduc_poker", "Slightly larger - 6 cards (2 suits × 3 ranks)"),
        ("liars_dice", "Not poker, but similar imperfect info game"),
        ("tiny_hanabi", "Cooperative card game"),
    ]
    
    print("\nPoker-like games in OpenSpiel:")
    for game_name, description in poker_games:
        try:
            game = pyspiel.load_game(game_name)
            print(f"\n  {game_name}:")
            print(f"    {description}")
            print(f"    Players: {game.num_players()}")
            print(f"    Actions: {game.num_distinct_actions()}")
        except:
            print(f"\n  {game_name}: Not available in this version")
    
    print("\n⚠️  Note: OpenSpiel does NOT have full No-Limit Texas Hold'em")
    print("   For NLHE, use PokerKit instead")


if __name__ == "__main__":
    # Run all demonstrations
    game = explore_kuhn_poker()
    play_random_game(game)
    understand_information_states(game)
    simulate_many_games(game, num_games=1000)
    list_available_poker_games()
