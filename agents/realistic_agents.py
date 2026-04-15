"""
realistic_agents.py - Improved AI agents for Infinite Texas Hold'em

These agents produce more realistic poker gameplay with:
- Proper preflop hand selection (VPIP 15-40%)
- Realistic fold frequencies (40-70% overall)
- Reasonable bet sizing (0.5x - 1.5x pot)
- Position awareness
- Different playing styles (tight/loose, passive/aggressive)

Agent Styles:
- NitAgent: Very tight (VPIP ~15%), only premium hands
- TightAggressiveAgent: Tight preflop, aggressive postflop (VPIP ~25%)  
- LooseAggressiveAgent: Plays more hands, lots of aggression (VPIP ~40%)
- LoosePassiveAgent: Plays many hands but rarely raises (VPIP ~45%)
- BalancedAgent: Mix of strategies, harder to exploit
"""

import os
import sys
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from infinite_holdem import (
    InfiniteDeck, InfiniteHandEvaluator, InfiniteHoldemGame, 
    Card, HandRank
)


# =============================================================================
# Hand Strength Utilities
# =============================================================================

def get_preflop_hand_class(hole: List[Card]) -> Tuple[str, float]:
    """
    Classify preflop hand and return (class_name, strength).
    
    Returns strength from 0.0 (worst) to 1.0 (best).
    Based on standard preflop hand rankings.
    """
    r1, r2 = hole[0].rank_value, hole[1].rank_value
    high, low = max(r1, r2), min(r1, r2)
    suited = hole[0].suit == hole[1].suit
    gap = high - low
    
    # Premium pairs (AA, KK, QQ, JJ)
    if r1 == r2 and high >= 9:  # JJ+
        if high == 12:  # AA
            return "premium_pair", 0.95
        elif high == 11:  # KK
            return "premium_pair", 0.92
        elif high == 10:  # QQ
            return "premium_pair", 0.88
        else:  # JJ
            return "premium_pair", 0.84
    
    # Medium pairs (TT, 99, 88, 77)
    if r1 == r2 and high >= 5:
        return "medium_pair", 0.60 + (high - 5) * 0.04
    
    # Small pairs (66-22)
    if r1 == r2:
        return "small_pair", 0.40 + high * 0.02
    
    # Premium broadway (AK, AQ, AJ, KQ)
    if high == 12 and low >= 9:  # A with J+
        base = 0.70 + (low - 9) * 0.05
        if suited:
            base += 0.05
        return "premium_broadway", base
    
    if high == 11 and low == 10:  # KQ
        return "premium_broadway", 0.68 if suited else 0.63
    
    # Suited connectors
    if suited and gap == 1 and low >= 4:
        return "suited_connector", 0.45 + (low - 4) * 0.03
    
    # Suited aces
    if suited and high == 12:
        return "suited_ace", 0.50 + low * 0.02
    
    # Broadway cards (any two cards T+)
    if low >= 8:
        base = 0.35 + (low - 8) * 0.05
        if suited:
            base += 0.05
        return "broadway", base
    
    # Suited one-gappers
    if suited and gap == 2 and low >= 3:
        return "suited_gapper", 0.35 + (low - 3) * 0.02
    
    # Offsuit connectors
    if gap == 1 and low >= 5:
        return "offsuit_connector", 0.30 + (low - 5) * 0.02
    
    # Trash
    base = (high + low) / 24 * 0.25
    if suited:
        base += 0.05
    return "trash", base


def get_postflop_strength(evaluator: InfiniteHandEvaluator, 
                          hole: List[Card], 
                          board: List[Card]) -> float:
    """Get postflop hand strength as a percentile (0-1)."""
    if not board:
        _, strength = get_preflop_hand_class(hole)
        return strength
    
    result = evaluator.evaluate(hole + board)
    
    # Map hand ranks to approximate strength percentiles
    rank_strengths = {
        HandRank.HIGH_CARD: 0.15,
        HandRank.PAIR: 0.35,
        HandRank.TWO_PAIR: 0.55,
        HandRank.THREE_OF_A_KIND: 0.70,
        HandRank.STRAIGHT: 0.82,  # Rarer in infinite
        HandRank.FLUSH: 0.78,
        HandRank.FULL_HOUSE: 0.85,
        HandRank.FOUR_OF_A_KIND: 0.92,
        HandRank.FIVE_OF_A_KIND: 0.96,
        HandRank.FLUSH_HOUSE: 0.97,
        HandRank.STRAIGHT_FLUSH: 0.98,
        HandRank.FLUSH_FIVE: 0.99,
    }
    
    base = rank_strengths.get(result.rank, 0.15)
    
    # Adjust by kicker strength within the hand rank
    if result.kickers:
        kicker_bonus = sum(result.kickers) / (13 * len(result.kickers)) * 0.05
        base += kicker_bonus
    
    return min(0.99, base)


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """Abstract base class for poker agents."""
    
    def __init__(self, name: str = "Agent"):
        self.name = name
        self.evaluator = InfiniteHandEvaluator()
    
    @abstractmethod
    def get_action(self, game: InfiniteHoldemGame, player_idx: int) -> Tuple[str, Optional[int]]:
        """Return (action, amount_or_none)."""
        pass
    
    def _get_raise_amount(self, game: InfiniteHoldemGame, player_idx: int, 
                          pot_fraction: float) -> Optional[int]:
        """Calculate raise amount as fraction of pot, clamped to legal range."""
        min_raise = game.get_min_raise_to(player_idx)
        max_raise = game.get_max_raise_to(player_idx)
        
        if min_raise is None or max_raise is None:
            return None
        
        target = int(game.pot * pot_fraction)
        target = max(min_raise, min(target, max_raise))
        
        # Don't make tiny raises
        if target < game.big_blind * 2:
            target = min(game.big_blind * 2, max_raise)
        
        return target


# =============================================================================
# Nit Agent (Very Tight)
# =============================================================================

class NitAgent(BaseAgent):
    """
    Very tight player - only plays premium hands.
    VPIP: ~15%, PFR: ~12%
    """
    
    def __init__(self):
        super().__init__("Nit")
        self.vpip_threshold = 0.70  # Only play top 30% of hands
        self.raise_threshold = 0.80  # Raise with top 20%
    
    def get_action(self, game: InfiniteHoldemGame, player_idx: int) -> Tuple[str, Optional[int]]:
        legal = game.get_legal_actions(player_idx)
        to_call = game.get_to_call(player_idx)
        pot = game.pot
        hole = game.hole_cards[player_idx]
        board = game.board
        
        # Get hand strength
        if not board:
            _, strength = get_preflop_hand_class(hole)
        else:
            strength = get_postflop_strength(self.evaluator, hole, board)
        
        # === PREFLOP ===
        if not board:
            # Only enter pot with strong hands
            if strength < self.vpip_threshold:
                if to_call > 0 and "fold" in legal:
                    return "fold", None
                if "check" in legal:
                    return "check", None
                return "fold", None
            
            # Raise with premium hands
            if strength >= self.raise_threshold and "raise" in legal:
                amount = self._get_raise_amount(game, player_idx, 0.75)
                if amount:
                    return "raise", amount
            
            # Call with good hands
            if "call" in legal:
                return "call", None
            if "check" in legal:
                return "check", None
        
        # === POSTFLOP ===
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
        
        # Strong hand: bet/raise for value
        if strength > 0.70:
            if "raise" in legal:
                amount = self._get_raise_amount(game, player_idx, 0.6)
                if amount:
                    return "raise", amount
            if "check" in legal:
                return "check", None
            return "call", None
        
        # Medium hand: check/call if odds are good
        if strength > 0.45:
            if to_call == 0 and "check" in legal:
                return "check", None
            if strength > pot_odds * 1.2 and "call" in legal:
                return "call", None
            if "fold" in legal:
                return "fold", None
        
        # Weak hand: fold to any bet
        if to_call > 0 and "fold" in legal:
            return "fold", None
        if "check" in legal:
            return "check", None
        
        return "fold", None


# =============================================================================
# Tight Aggressive Agent (TAG)
# =============================================================================

class TightAggressiveAgent(BaseAgent):
    """
    Tight-aggressive player - selective but aggressive.
    VPIP: ~25%, PFR: ~20%
    """
    
    def __init__(self, aggression: float = 0.6):
        super().__init__("TAG")
        self.aggression = aggression
        self.vpip_threshold = 0.55  # Play top 45% of hands
        self.raise_threshold = 0.65  # Raise with top 35%
    
    def get_action(self, game: InfiniteHoldemGame, player_idx: int) -> Tuple[str, Optional[int]]:
        legal = game.get_legal_actions(player_idx)
        to_call = game.get_to_call(player_idx)
        pot = game.pot
        hole = game.hole_cards[player_idx]
        board = game.board
        
        if not board:
            hand_class, strength = get_preflop_hand_class(hole)
        else:
            hand_class = None
            strength = get_postflop_strength(self.evaluator, hole, board)
        
        # === PREFLOP ===
        if not board:
            # Fold trash
            if strength < self.vpip_threshold:
                if to_call > 0:
                    # Occasionally defend big blind with marginal hands
                    if player_idx == 1 and to_call <= game.big_blind and random.random() < 0.3:
                        return "call", None
                    if "fold" in legal:
                        return "fold", None
                if "check" in legal:
                    return "check", None
                return "fold", None
            
            # Raise with strong hands
            if strength >= self.raise_threshold and "raise" in legal:
                # Vary sizing based on hand strength
                size = 0.6 + (strength - 0.65) * 2  # 0.6x - 1.3x pot
                amount = self._get_raise_amount(game, player_idx, size)
                if amount:
                    return "raise", amount
            
            # Call with medium hands (suited connectors, small pairs)
            if "call" in legal:
                return "call", None
            if "check" in legal:
                return "check", None
        
        # === POSTFLOP ===
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
        
        # Very strong hand: bet/raise for value
        if strength > 0.75:
            if "raise" in legal:
                amount = self._get_raise_amount(game, player_idx, 0.7 + self.aggression * 0.3)
                if amount:
                    return "raise", amount
            if "call" in legal:
                return "call", None
            return "check", None
        
        # Strong hand: bet or call
        if strength > 0.55:
            if to_call == 0 and "raise" in legal and random.random() < self.aggression:
                amount = self._get_raise_amount(game, player_idx, 0.5)
                if amount:
                    return "raise", amount
            if "check" in legal:
                return "check", None
            if strength > pot_odds + 0.1 and "call" in legal:
                return "call", None
            if "fold" in legal:
                return "fold", None
        
        # Medium hand: check/fold
        if strength > 0.35:
            if to_call == 0 and "check" in legal:
                return "check", None
            # Bluff occasionally
            if to_call == 0 and "raise" in legal and random.random() < 0.15 * self.aggression:
                amount = self._get_raise_amount(game, player_idx, 0.5)
                if amount:
                    return "raise", amount
            if strength > pot_odds and "call" in legal:
                return "call", None
            if "fold" in legal:
                return "fold", None
        
        # Weak hand: mostly fold
        if to_call > 0:
            if "fold" in legal:
                return "fold", None
        if "check" in legal:
            return "check", None
        
        return "fold", None


# =============================================================================
# Loose Aggressive Agent (LAG)
# =============================================================================

class LooseAggressiveAgent(BaseAgent):
    """
    Loose-aggressive player - plays many hands aggressively.
    VPIP: ~40%, PFR: ~30%
    """
    
    def __init__(self, aggression: float = 0.7):
        super().__init__("LAG")
        self.aggression = aggression
        self.vpip_threshold = 0.35  # Play top 65% of hands
        self.bluff_frequency = 0.25
    
    def get_action(self, game: InfiniteHoldemGame, player_idx: int) -> Tuple[str, Optional[int]]:
        legal = game.get_legal_actions(player_idx)
        to_call = game.get_to_call(player_idx)
        pot = game.pot
        hole = game.hole_cards[player_idx]
        board = game.board
        
        if not board:
            hand_class, strength = get_preflop_hand_class(hole)
        else:
            hand_class = None
            strength = get_postflop_strength(self.evaluator, hole, board)
        
        # === PREFLOP ===
        if not board:
            # Still fold the worst hands
            if strength < self.vpip_threshold:
                if to_call > game.big_blind * 3:  # Fold to big raises
                    if "fold" in legal:
                        return "fold", None
                # Sometimes play speculative hands
                if to_call <= game.big_blind and random.random() < 0.4:
                    if "call" in legal:
                        return "call", None
                if "check" in legal:
                    return "check", None
                if "fold" in legal:
                    return "fold", None
            
            # Raise frequently
            if "raise" in legal and random.random() < 0.6:
                size = 0.5 + strength * 0.5
                amount = self._get_raise_amount(game, player_idx, size)
                if amount:
                    return "raise", amount
            
            if "call" in legal:
                return "call", None
            if "check" in legal:
                return "check", None
        
        # === POSTFLOP ===
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
        
        # Strong hand: bet/raise aggressively
        if strength > 0.65:
            if "raise" in legal:
                size = 0.6 + self.aggression * 0.4
                amount = self._get_raise_amount(game, player_idx, size)
                if amount:
                    return "raise", amount
            if "call" in legal:
                return "call", None
        
        # Medium hand: bet for thin value or as bluff
        if strength > 0.40:
            if to_call == 0 and "raise" in legal and random.random() < self.aggression:
                amount = self._get_raise_amount(game, player_idx, 0.5)
                if amount:
                    return "raise", amount
            if "check" in legal:
                return "check", None
            if strength > pot_odds and "call" in legal:
                return "call", None
        
        # Weak hand: bluff sometimes, fold to aggression
        if to_call == 0:
            # Bluff
            if "raise" in legal and random.random() < self.bluff_frequency:
                amount = self._get_raise_amount(game, player_idx, 0.6)
                if amount:
                    return "raise", amount
            if "check" in legal:
                return "check", None
        else:
            # Fold to bets with weak hands
            if strength < pot_odds * 0.8:
                if "fold" in legal:
                    return "fold", None
            if "call" in legal:
                return "call", None
        
        if "check" in legal:
            return "check", None
        return "fold", None


# =============================================================================
# Loose Passive Agent (Calling Station)
# =============================================================================

class LoosePassiveAgent(BaseAgent):
    """
    Loose-passive player - plays many hands but rarely raises.
    VPIP: ~45%, PFR: ~5%
    Classic "calling station"
    """
    
    def __init__(self):
        super().__init__("Fish")
        self.vpip_threshold = 0.25  # Play 75% of hands
    
    def get_action(self, game: InfiniteHoldemGame, player_idx: int) -> Tuple[str, Optional[int]]:
        legal = game.get_legal_actions(player_idx)
        to_call = game.get_to_call(player_idx)
        pot = game.pot
        hole = game.hole_cards[player_idx]
        board = game.board
        
        if not board:
            _, strength = get_preflop_hand_class(hole)
        else:
            strength = get_postflop_strength(self.evaluator, hole, board)
        
        # === PREFLOP ===
        if not board:
            # Only fold the absolute worst hands to big raises
            if strength < self.vpip_threshold and to_call > game.big_blind * 4:
                if "fold" in legal:
                    return "fold", None
            
            # Rarely raise - only with monsters
            if strength > 0.85 and "raise" in legal and random.random() < 0.5:
                amount = self._get_raise_amount(game, player_idx, 0.6)
                if amount:
                    return "raise", amount
            
            # Call almost everything
            if "call" in legal:
                return "call", None
            if "check" in legal:
                return "check", None
        
        # === POSTFLOP ===
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
        
        # Bet rarely, only with very strong hands
        if strength > 0.80 and "raise" in legal and random.random() < 0.4:
            amount = self._get_raise_amount(game, player_idx, 0.5)
            if amount:
                return "raise", amount
        
        # Call with any piece of the board
        if strength > 0.25:
            if "call" in legal:
                return "call", None
            if "check" in legal:
                return "check", None
        
        # Even with nothing, call small bets
        if to_call < pot * 0.3:
            if "call" in legal:
                return "call", None
        
        if "check" in legal:
            return "check", None
        if "fold" in legal:
            return "fold", None
        
        return "call", None


# =============================================================================
# Balanced Agent
# =============================================================================

class BalancedAgent(BaseAgent):
    """
    Attempts to play a balanced strategy.
    Mixes value bets with bluffs, varies bet sizing.
    VPIP: ~28%, PFR: ~22%
    """
    
    def __init__(self):
        super().__init__("Balanced")
        self.vpip_threshold = 0.50
        # Bluff:value ratio by street
        self.bluff_ratio = {"preflop": 0.1, "flop": 0.3, "turn": 0.2, "river": 0.15}
    
    def get_action(self, game: InfiniteHoldemGame, player_idx: int) -> Tuple[str, Optional[int]]:
        legal = game.get_legal_actions(player_idx)
        to_call = game.get_to_call(player_idx)
        pot = game.pot
        hole = game.hole_cards[player_idx]
        board = game.board
        street = game.street
        
        if not board:
            hand_class, strength = get_preflop_hand_class(hole)
        else:
            hand_class = None
            strength = get_postflop_strength(self.evaluator, hole, board)
        
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
        bluff_freq = self.bluff_ratio.get(street, 0.15)
        
        # === PREFLOP ===
        if not board:
            # Fold weak hands
            if strength < self.vpip_threshold:
                # Defend BB sometimes
                if player_idx == 1 and to_call <= game.big_blind * 2:
                    if random.random() < 0.25:
                        return "call", None
                if "fold" in legal:
                    return "fold", None
                if "check" in legal:
                    return "check", None
            
            # Open raise with good hands
            if "raise" in legal and strength > 0.60:
                amount = self._get_raise_amount(game, player_idx, 0.75)
                if amount:
                    return "raise", amount
            
            # 3-bet with premium and sometimes as bluff
            if "raise" in legal and to_call > 0:
                if strength > 0.80 or random.random() < bluff_freq:
                    amount = self._get_raise_amount(game, player_idx, 1.0)
                    if amount:
                        return "raise", amount
            
            if "call" in legal:
                return "call", None
            if "check" in legal:
                return "check", None
        
        # === POSTFLOP ===
        
        # Value betting with strong hands
        if strength > 0.65:
            if "raise" in legal:
                # Size based on strength
                size = 0.5 + (strength - 0.65) * 1.5
                amount = self._get_raise_amount(game, player_idx, size)
                if amount:
                    return "raise", amount
            if "call" in legal:
                return "call", None
            if "check" in legal:
                return "check", None
        
        # Semi-bluffing / thin value with medium hands
        if strength > 0.40:
            if to_call == 0 and "raise" in legal:
                if random.random() < 0.4:
                    amount = self._get_raise_amount(game, player_idx, 0.5)
                    if amount:
                        return "raise", amount
                return "check", None
            
            if strength > pot_odds + 0.05 and "call" in legal:
                return "call", None
            if "fold" in legal:
                return "fold", None
            if "check" in legal:
                return "check", None
        
        # Weak hands: bluff occasionally, otherwise fold
        if to_call == 0:
            if "raise" in legal and random.random() < bluff_freq:
                amount = self._get_raise_amount(game, player_idx, 0.6)
                if amount:
                    return "raise", amount
            if "check" in legal:
                return "check", None
        
        # Fold to bets with weak hands
        if "fold" in legal:
            return "fold", None
        if "check" in legal:
            return "check", None
        
        return "fold", None


# =============================================================================
# Random Agent (Baseline)
# =============================================================================

class RandomAgent(BaseAgent):
    """
    Random agent - but with somewhat realistic action distribution.
    Folds more when facing bets, checks more when possible.
    """
    
    def __init__(self):
        super().__init__("Random")
    
    def get_action(self, game: InfiniteHoldemGame, player_idx: int) -> Tuple[str, Optional[int]]:
        legal = game.get_legal_actions(player_idx)
        to_call = game.get_to_call(player_idx)
        
        if not legal:
            return "check", None
        
        # Weight actions more realistically
        if to_call > 0:
            # Facing a bet: 40% fold, 45% call, 15% raise
            r = random.random()
            if r < 0.40 and "fold" in legal:
                return "fold", None
            elif r < 0.85 and "call" in legal:
                return "call", None
            elif "raise" in legal:
                amount = self._get_raise_amount(game, player_idx, random.uniform(0.5, 1.0))
                if amount:
                    return "raise", amount
            return "call", None
        else:
            # No bet to call: 70% check, 30% raise
            r = random.random()
            if r < 0.70 and "check" in legal:
                return "check", None
            elif "raise" in legal:
                amount = self._get_raise_amount(game, player_idx, random.uniform(0.5, 0.8))
                if amount:
                    return "raise", amount
            return "check", None


# =============================================================================
# Helper Functions
# =============================================================================

def _normalize_action(game: InfiniteHoldemGame, player_idx: int, 
                      action: str, amount: Optional[int]) -> Tuple[str, Optional[int]]:
    """Ensure action is legal, fall back to safe default."""
    legal = game.get_legal_actions(player_idx)
    
    if not legal:
        raise ValueError(f"No legal actions for player {player_idx}")
    
    if action == "raise" and "raise" in legal:
        min_raise = game.get_min_raise_to(player_idx)
        max_raise = game.get_max_raise_to(player_idx)
        if min_raise is None or max_raise is None:
            action = "check" if "check" in legal else "call"
            return action, None
        if amount is None:
            amount = min_raise
        return "raise", max(min_raise, min(amount, max_raise))
    
    if action in legal:
        return action, None
    
    # Fallback
    for fallback in ("check", "call", "fold"):
        if fallback in legal:
            return fallback, None
    
    return legal[0], None


def play_hand(agent0: BaseAgent, agent1: BaseAgent, 
              verbose: bool = False, seed: Optional[int] = None) -> List[float]:
    """Play one heads-up hand."""
    from collections import Counter
    
    game = InfiniteHoldemGame(
        num_players=2,
        starting_stack=10000,
        small_blind=50,
        big_blind=100,
        seed=seed,
    )
    game.deal_hole_cards()
    agents = [agent0, agent1]
    
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"{agent0.name} vs {agent1.name}")
        print(f"{'=' * 50}")
        print(f"Player 0: {game.hole_cards[0]}")
        print(f"Player 1: {game.hole_cards[1]}")
        
        all_cards = game.hole_cards[0] + game.hole_cards[1]
        counts = Counter(str(c) for c in all_cards)
        dups = [c for c, n in counts.items() if n > 1]
        if dups:
            print(f"  -> Duplicates: {dups}")
        print("\n--- PREFLOP ---")
    
    prev_street = "preflop"
    
    while not game.is_hand_over():
        if game.actor_index is None:
            if game.is_betting_round_complete():
                game.advance_street()
                if verbose and game.street != prev_street:
                    print(f"\n--- {game.street.upper()} ---")
                    if game.board:
                        print(f"Board: {game.board}")
                    prev_street = game.street
                continue
            raise RuntimeError("No actor but hand not over")
        
        actor = game.actor_index
        action, amount = agents[actor].get_action(game, actor)
        action, amount = _normalize_action(game, actor, action, amount)
        
        if verbose:
            label = action if amount is None else f"{action} to {amount}"
            print(f"{agents[actor].name}: {label}")
        
        game.apply_action(actor, action, amount)
    
    winners, hand_results = game.determine_winner()
    winners, payoffs = game.settle_pot()
    
    if verbose:
        if game.street == "showdown":
            print("\n--- SHOWDOWN ---")
            for i, r in enumerate(hand_results):
                if r:
                    print(f"Player {i}: {r.description}")
        
        print(f"\nWinner: {agents[winners[0]].name}")
        print(f"Payoffs: {payoffs}")
        print(f"Sum: {sum(payoffs)}")
    
    return payoffs


def run_tournament(agents: List[BaseAgent], num_hands: int = 500) -> None:
    """Round-robin tournament."""
    print(f"\n{'=' * 60}")
    print("INFINITE HOLD'EM TOURNAMENT")
    print("=" * 60)
    print(f"Agents: {[a.name for a in agents]}")
    print(f"Hands per matchup: {num_hands}")
    
    results = {a.name: 0.0 for a in agents}
    
    for i, a0 in enumerate(agents):
        for j, a1 in enumerate(agents):
            if i >= j:
                continue
            
            print(f"\n{a0.name} vs {a1.name}...", end=" ")
            t0, t1 = 0.0, 0.0
            
            for h in range(num_hands):
                seed = (i + 1) * 100000 + (j + 1) * 1000 + h
                p0, p1 = play_hand(a0, a1, seed=seed)
                t0 += p0
                t1 += p1
            
            avg0, avg1 = t0 / num_hands, t1 / num_hands
            print(f"Avg: {avg0:+.1f} vs {avg1:+.1f}")
            
            results[a0.name] += avg0
            results[a1.name] += avg1
    
    print(f"\n{'=' * 60}")
    print("FINAL STANDINGS")
    print("=" * 60)
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score:+.1f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("INFINITE TEXAS HOLD'EM - REALISTIC AGENTS")
    print("=" * 50)
    
    # Create agents
    nit = NitAgent()
    tag = TightAggressiveAgent()
    lag = LooseAggressiveAgent()
    fish = LoosePassiveAgent()
    balanced = BalancedAgent()
    random_agent = RandomAgent()
    
    # Sample hands
    print("\n=== SAMPLE HANDS ===")
    for i in range(3):
        play_hand(tag, lag, verbose=True, seed=i * 100)
    
    # Tournament
    agents = [nit, tag, lag, fish, balanced, random_agent]
    run_tournament(agents, num_hands=300)
