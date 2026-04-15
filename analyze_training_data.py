"""
analyze_training_data.py - Analyze and Visualize Infinite Hold'em Training Data

This script provides:
1. Summary statistics
2. Action distribution analysis
3. Hand replay visualization
4. Reward analysis
5. Duplicate card analysis (unique to infinite variant)
6. Decision pattern analysis by street/position
7. Export utilities for ML

Usage:
    python analyze_training_data.py path/to/training_data.json
"""

import json
import sys
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Optional: for visualizations
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: Install matplotlib for visualizations (pip install matplotlib)")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# Data Loading
# =============================================================================

def load_data(filepath: str) -> List[Dict]:
    """Load training data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


# =============================================================================
# Summary Statistics
# =============================================================================

def print_summary_stats(data: List[Dict]):
    """Print overall summary statistics."""
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    num_examples = len(data)
    hand_ids = set(d['hand_id'] for d in data)
    num_hands = len(hand_ids)
    
    print(f"\nTotal decision examples: {num_examples}")
    print(f"Total hands played: {num_hands}")
    print(f"Average decisions per hand: {num_examples / num_hands:.1f}")
    
    # Hands with duplicates
    dup_hands = set(d['hand_id'] for d in data if d.get('has_duplicates', False))
    print(f"\nHands with duplicate cards: {len(dup_hands)}/{num_hands} ({100*len(dup_hands)/num_hands:.1f}%)")
    
    # Action distribution
    actions = [d['action'] for d in data]
    action_counts = Counter(actions)
    print(f"\nOverall action distribution:")
    for action in ['fold', 'check', 'call', 'raise']:
        count = action_counts.get(action, 0)
        pct = 100 * count / len(actions)
        print(f"  {action}: {count} ({pct:.1f}%)")
    
    # Reward stats
    rewards = [d['reward'] for d in data]
    print(f"\nReward statistics:")
    print(f"  Min: {min(rewards)}")
    print(f"  Max: {max(rewards)}")
    print(f"  Mean: {sum(rewards)/len(rewards):.2f}")
    
    # Verify zero-sum
    hand_reward_sums = defaultdict(float)
    for d in data:
        hand_reward_sums[d['hand_id']] += d['reward']
    
    non_zero_hands = sum(1 for s in hand_reward_sums.values() if abs(s) > 0.01)
    print(f"\nZero-sum validation:")
    print(f"  Hands with non-zero reward sum: {non_zero_hands}/{num_hands}")
    if non_zero_hands == 0:
        print("  ✓ All hands are zero-sum (pot settlement correct)")


# =============================================================================
# Hand Reconstruction & Replay
# =============================================================================

@dataclass
class HandSummary:
    """Summary of a complete hand."""
    hand_id: int
    players: Dict[int, Dict]  # position -> {hole_cards, final_reward}
    board: List[str]
    actions: List[Dict]
    has_duplicates: bool
    winner: Optional[int]


def reconstruct_hands(data: List[Dict]) -> Dict[int, HandSummary]:
    """Reconstruct complete hands from decision data."""
    hands = defaultdict(lambda: {
        'actions': [],
        'players': {},
        'board': [],
        'has_duplicates': False
    })
    
    for d in data:
        hid = d['hand_id']
        pos = d['position']
        
        # Track player info
        if pos not in hands[hid]['players']:
            hands[hid]['players'][pos] = {
                'hole_cards': d['hole_cards'],
                'reward': d['reward']
            }
        
        # Track actions
        hands[hid]['actions'].append({
            'step': d['step_in_hand'],
            'position': pos,
            'street': d['street'],
            'action': d['action'],
            'raise_amount': d.get('raise_amount'),
            'pot': d['pot_size'],
            'to_call': d['to_call']
        })
        
        # Track board (use the longest board seen)
        if len(d['board_cards']) > len(hands[hid]['board']):
            hands[hid]['board'] = d['board_cards']
        
        hands[hid]['has_duplicates'] = hands[hid]['has_duplicates'] or d.get('has_duplicates', False)
    
    # Convert to HandSummary objects
    summaries = {}
    for hid, h in hands.items():
        # Determine winner
        winner = None
        for pos, player in h['players'].items():
            if player['reward'] > 0:
                winner = pos
                break
        
        summaries[hid] = HandSummary(
            hand_id=hid,
            players=h['players'],
            board=h['board'],
            actions=sorted(h['actions'], key=lambda x: x['step']),
            has_duplicates=h['has_duplicates'],
            winner=winner
        )
    
    return summaries


def print_hand_replay(hand: HandSummary, verbose: bool = True):
    """Print a detailed replay of a hand."""
    print(f"\n{'='*60}")
    print(f"HAND #{hand.hand_id}" + (" [HAS DUPLICATES]" if hand.has_duplicates else ""))
    print("=" * 60)
    
    # Players
    for pos, player in sorted(hand.players.items()):
        pos_name = "SB" if pos == 0 else "BB"
        cards = ' '.join(player['hole_cards'])
        reward = player['reward']
        win_marker = " 🏆" if reward > 0 else ""
        print(f"Player {pos} ({pos_name}): [{cards}] → {reward:+d}{win_marker}")
    
    if verbose:
        print(f"\nBoard: {' '.join(hand.board) if hand.board else '(no showdown)'}")
        
        # Group actions by street
        current_street = None
        for action in hand.actions:
            if action['street'] != current_street:
                current_street = action['street']
                print(f"\n--- {current_street.upper()} ---")
            
            pos_name = "SB" if action['position'] == 0 else "BB"
            action_str = action['action']
            if action['raise_amount']:
                action_str += f" {action['raise_amount']}"
            
            print(f"  P{action['position']} ({pos_name}): {action_str:12} (pot: {action['pot']})")
    
    print()


def replay_hands(data: List[Dict], hand_ids: List[int] = None, max_hands: int = 5):
    """Replay specific hands or random sample."""
    hands = reconstruct_hands(data)
    
    if hand_ids is None:
        # Show first few hands
        hand_ids = sorted(hands.keys())[:max_hands]
    
    print(f"\nReplaying {len(hand_ids)} hands...")
    
    for hid in hand_ids:
        if hid in hands:
            print_hand_replay(hands[hid])


def replay_hands_with_duplicates(data: List[Dict], max_hands: int = 5):
    """Replay hands that contain duplicate cards."""
    hands = reconstruct_hands(data)
    dup_hands = [h for h in hands.values() if h.has_duplicates]
    
    print(f"\n{'='*60}")
    print(f"HANDS WITH DUPLICATE CARDS ({len(dup_hands)} total)")
    print("=" * 60)
    
    for hand in dup_hands[:max_hands]:
        print_hand_replay(hand)


# =============================================================================
# Action Analysis
# =============================================================================

def analyze_actions_by_street(data: List[Dict]):
    """Analyze action distribution by street."""
    print("\n" + "=" * 60)
    print("ACTION DISTRIBUTION BY STREET")
    print("=" * 60)
    
    street_actions = defaultdict(list)
    for d in data:
        street_actions[d['street']].append(d['action'])
    
    streets = ['preflop', 'flop', 'turn', 'river']
    
    for street in streets:
        if street not in street_actions:
            continue
        actions = street_actions[street]
        counts = Counter(actions)
        total = len(actions)
        
        print(f"\n{street.upper()} ({total} decisions):")
        for action in ['fold', 'check', 'call', 'raise']:
            count = counts.get(action, 0)
            pct = 100 * count / total if total > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  {action:6}: {count:4} ({pct:5.1f}%) {bar}")


def analyze_actions_by_position(data: List[Dict]):
    """Analyze action distribution by position."""
    print("\n" + "=" * 60)
    print("ACTION DISTRIBUTION BY POSITION")
    print("=" * 60)
    
    pos_actions = defaultdict(list)
    for d in data:
        pos_actions[d['position']].append(d['action'])
    
    for pos in [0, 1]:
        pos_name = "Small Blind (P0)" if pos == 0 else "Big Blind (P1)"
        actions = pos_actions[pos]
        counts = Counter(actions)
        total = len(actions)
        
        print(f"\n{pos_name} ({total} decisions):")
        for action in ['fold', 'check', 'call', 'raise']:
            count = counts.get(action, 0)
            pct = 100 * count / total if total > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  {action:6}: {count:4} ({pct:5.1f}%) {bar}")


def analyze_raise_sizes(data: List[Dict]):
    """Analyze raise sizing patterns."""
    print("\n" + "=" * 60)
    print("RAISE SIZE ANALYSIS")
    print("=" * 60)
    
    raises = [d for d in data if d['action'] == 'raise' and d.get('raise_amount')]
    
    if not raises:
        print("No raises found in data.")
        return
    
    # By street
    street_raises = defaultdict(list)
    for d in raises:
        pot = d['pot_size']
        raise_amt = d['raise_amount']
        # Calculate raise as % of pot
        pot_pct = (raise_amt / pot * 100) if pot > 0 else 0
        street_raises[d['street']].append({
            'amount': raise_amt,
            'pot_pct': pot_pct,
            'pot': pot
        })
    
    for street in ['preflop', 'flop', 'turn', 'river']:
        if street not in street_raises:
            continue
        
        r = street_raises[street]
        amounts = [x['amount'] for x in r]
        pot_pcts = [x['pot_pct'] for x in r]
        
        print(f"\n{street.upper()} ({len(r)} raises):")
        print(f"  Raise amount: min={min(amounts)}, max={max(amounts)}, avg={sum(amounts)/len(amounts):.0f}")
        print(f"  As % of pot:  min={min(pot_pcts):.0f}%, max={max(pot_pcts):.0f}%, avg={sum(pot_pcts)/len(pot_pcts):.0f}%")


# =============================================================================
# Reward Analysis
# =============================================================================

def analyze_rewards(data: List[Dict]):
    """Analyze reward distributions."""
    print("\n" + "=" * 60)
    print("REWARD ANALYSIS")
    print("=" * 60)
    
    # Group by hand to get final results
    hands = reconstruct_hands(data)
    
    # Winners vs losers
    winners = []
    losers = []
    
    for hand in hands.values():
        for pos, player in hand.players.items():
            if player['reward'] > 0:
                winners.append(player['reward'])
            elif player['reward'] < 0:
                losers.append(player['reward'])
    
    print(f"\nWinning hands: {len(winners)}")
    if winners:
        print(f"  Average win: +{sum(winners)/len(winners):.0f}")
        print(f"  Biggest win: +{max(winners)}")
    
    print(f"\nLosing hands: {len(losers)}")
    if losers:
        print(f"  Average loss: {sum(losers)/len(losers):.0f}")
        print(f"  Biggest loss: {min(losers)}")
    
    # Reward by action taken
    print("\nAverage reward by action:")
    action_rewards = defaultdict(list)
    for d in data:
        action_rewards[d['action']].append(d['reward'])
    
    for action in ['fold', 'check', 'call', 'raise']:
        rewards = action_rewards.get(action, [])
        if rewards:
            avg = sum(rewards) / len(rewards)
            print(f"  {action}: {avg:+.1f}")


# =============================================================================
# Duplicate Card Analysis (Infinite Variant Specific)
# =============================================================================

def analyze_duplicates(data: List[Dict]):
    """Analyze hands with duplicate cards - unique to infinite variant."""
    print("\n" + "=" * 60)
    print("DUPLICATE CARD ANALYSIS (INFINITE VARIANT)")
    print("=" * 60)
    
    hands = reconstruct_hands(data)
    
    dup_hands = [h for h in hands.values() if h.has_duplicates]
    normal_hands = [h for h in hands.values() if not h.has_duplicates]
    
    print(f"\nHands with duplicates: {len(dup_hands)}")
    print(f"Hands without duplicates: {len(normal_hands)}")
    
    if not dup_hands:
        print("\nNo duplicate hands to analyze.")
        return
    
    # Compare outcomes
    def get_stats(hand_list):
        wins = []
        for h in hand_list:
            for p in h.players.values():
                if p['reward'] > 0:
                    wins.append(p['reward'])
        return wins
    
    dup_wins = get_stats(dup_hands)
    normal_wins = get_stats(normal_hands)
    
    print(f"\nAverage winning amount:")
    if dup_wins:
        print(f"  Hands with duplicates: +{sum(dup_wins)/len(dup_wins):.0f}")
    if normal_wins:
        print(f"  Normal hands: +{sum(normal_wins)/len(normal_wins):.0f}")
    
    # Show example duplicates
    print("\nExample hands with duplicates:")
    for hand in dup_hands[:3]:
        cards_seen = []
        for p in hand.players.values():
            cards_seen.extend(p['hole_cards'])
        cards_seen.extend(hand.board)
        
        # Find actual duplicates
        card_counts = Counter(cards_seen)
        dups = [c for c, n in card_counts.items() if n > 1]
        
        print(f"\n  Hand #{hand.hand_id}:")
        print(f"    Duplicate cards: {dups}")
        for pos, p in hand.players.items():
            print(f"    P{pos}: {p['hole_cards']} → {p['reward']:+d}")
        print(f"    Board: {hand.board}")


# =============================================================================
# ML Feature Extraction
# =============================================================================

def extract_ml_features(data: List[Dict]) -> Dict:
    """
    Extract features suitable for ML training.
    
    Returns dict with:
    - X: feature matrix
    - y_action: action labels
    - y_reward: rewards
    - feature_names: list of feature names
    """
    if not HAS_NUMPY:
        print("NumPy required for ML feature extraction")
        return None
    
    print("\n" + "=" * 60)
    print("ML FEATURE EXTRACTION")
    print("=" * 60)
    
    features = []
    actions = []
    rewards = []
    
    action_to_idx = {'fold': 0, 'check': 1, 'call': 2, 'raise': 3}
    street_to_idx = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    
    for d in data:
        # Build feature vector
        feat = []
        
        # Card counts (52 + 52 = 104 features)
        feat.extend(d['hole_card_counts'])
        feat.extend(d['board_card_counts'])
        
        # Game state features
        feat.append(d['pot_size'] / 10000)  # Normalized
        feat.append(d['stack'] / 10000)
        feat.append(d['opponent_stack'] / 10000)
        feat.append(d['to_call'] / 1000)
        feat.append(d['position'])
        feat.append(street_to_idx[d['street']])
        feat.append(1 if d.get('has_duplicates') else 0)
        
        features.append(feat)
        actions.append(action_to_idx.get(d['action'], -1))
        rewards.append(d['reward'])
    
    X = np.array(features)
    y_action = np.array(actions)
    y_reward = np.array(rewards)
    
    # Feature names
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['c', 'd', 'h', 's']
    card_names = [f"{r}{s}" for r in ranks for s in suits]
    
    feature_names = (
        [f"hole_{c}" for c in card_names] +
        [f"board_{c}" for c in card_names] +
        ['pot_norm', 'stack_norm', 'opp_stack_norm', 'to_call_norm', 
         'position', 'street', 'has_duplicates']
    )
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Actions shape: {y_action.shape}")
    print(f"Rewards shape: {y_reward.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    return {
        'X': X,
        'y_action': y_action,
        'y_reward': y_reward,
        'feature_names': feature_names,
        'action_map': {v: k for k, v in action_to_idx.items()}
    }


# =============================================================================
# Visualizations
# =============================================================================

def plot_action_distribution(data: List[Dict], save_path: str = None):
    """Plot action distribution pie chart."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    actions = [d['action'] for d in data]
    counts = Counter(actions)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall pie chart
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title('Overall Action Distribution')
    
    # By street bar chart
    street_actions = defaultdict(lambda: Counter())
    for d in data:
        street_actions[d['street']][d['action']] += 1
    
    streets = ['preflop', 'flop', 'turn', 'river']
    x = range(len(streets))
    width = 0.2
    
    for i, action in enumerate(['fold', 'check', 'call', 'raise']):
        values = [street_actions[s].get(action, 0) for s in streets]
        axes[1].bar([xi + i*width for xi in x], values, width, label=action, color=colors[i])
    
    axes[1].set_xlabel('Street')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Actions by Street')
    axes[1].set_xticks([xi + 1.5*width for xi in x])
    axes[1].set_xticklabels(streets)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_reward_distribution(data: List[Dict], save_path: str = None):
    """Plot reward distributions."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    rewards = [d['reward'] for d in data]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(rewards, bins=50, color='#45b7d1', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', label='Break even')
    axes[0].set_xlabel('Reward')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Reward Distribution')
    axes[0].legend()
    
    # Reward by action
    action_rewards = defaultdict(list)
    for d in data:
        action_rewards[d['action']].append(d['reward'])
    
    actions = ['fold', 'check', 'call', 'raise']
    avg_rewards = [np.mean(action_rewards[a]) if action_rewards[a] else 0 for a in actions]
    colors = ['#ff6b6b' if r < 0 else '#4ecdc4' for r in avg_rewards]
    
    axes[1].bar(actions, avg_rewards, color=colors, edgecolor='black')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Action')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('Average Reward by Action')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_pot_sizes(data: List[Dict], save_path: str = None):
    """Plot pot size progression."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return
    
    # Group by street
    street_pots = defaultdict(list)
    for d in data:
        street_pots[d['street']].append(d['pot_size'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    streets = ['preflop', 'flop', 'turn', 'river']
    positions = range(len(streets))
    
    # Box plot
    box_data = [street_pots[s] for s in streets]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticklabels(streets)
    ax.set_xlabel('Street')
    ax.set_ylabel('Pot Size')
    ax.set_title('Pot Size Distribution by Street')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main(filepath: str):
    """Run full analysis on training data."""
    print(f"\nLoading data from: {filepath}")
    data = load_data(filepath)
    print(f"Loaded {len(data)} decision examples")
    
    # Run all analyses
    print_summary_stats(data)
    analyze_actions_by_street(data)
    analyze_actions_by_position(data)
    analyze_raise_sizes(data)
    analyze_rewards(data)
    analyze_duplicates(data)
    
    # Replay some hands
    print("\n" + "=" * 60)
    print("SAMPLE HAND REPLAYS")
    print("=" * 60)
    replay_hands(data, max_hands=3)
    
    # Replay hands with duplicates
    replay_hands_with_duplicates(data, max_hands=2)
    
    # ML features
    ml_data = extract_ml_features(data)
    
    # Visualizations
    if HAS_MATPLOTLIB:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        plot_action_distribution(data, 'action_distribution.png')
        plot_reward_distribution(data, 'reward_distribution.png')
        plot_pot_sizes(data, 'pot_sizes.png')
        
        print("\nPlots saved!")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return data, ml_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_training_data.py <path_to_json>")
        print("\nExample:")
        print("  python analyze_training_data.py data/infinite_holdem_training_data.json")
        sys.exit(1)
    
    filepath = sys.argv[1]
    data, ml_data = main(filepath)
