"""
test_game_engine.py — Tests for GameEngine and StateEncoder.

Run from project root:  python -m pytest tests/test_game_engine.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))

import numpy as np
import pytest
from game_engine import GameEngine, GameState, Action, Street
from state_encoder import StateEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_engine(**kwargs) -> GameEngine:
    defaults = dict(starting_stack=1000, small_blind=5, big_blind=10,
                    raise_sizes=[0.5, 1.0, 2.0])
    defaults.update(kwargs)
    return GameEngine(**defaults)


# ---------------------------------------------------------------------------
# Setup / Blinds
# ---------------------------------------------------------------------------

class TestSetup:
    def test_reset_returns_game_state(self):
        engine = make_engine()
        state = engine.reset()
        assert isinstance(state, GameState)

    def test_blinds_posted_correctly(self):
        engine = make_engine()
        state = engine.reset()
        assert state.bets[0] == 5    # SB
        assert state.bets[1] == 10   # BB
        assert state.pot == 15

    def test_stacks_reduced_by_blinds(self):
        engine = make_engine()
        state = engine.reset()
        assert state.stacks[0] == 995
        assert state.stacks[1] == 990

    def test_hole_cards_dealt(self):
        engine = make_engine()
        state = engine.reset()
        assert len(state.hole_cards[0]) == 2
        assert len(state.hole_cards[1]) == 2

    def test_hole_cards_are_valid(self):
        engine = make_engine()
        state = engine.reset()
        for p in range(2):
            for c in state.hole_cards[p]:
                assert 0 <= c <= 51

    def test_board_empty_preflop(self):
        engine = make_engine()
        state = engine.reset()
        assert state.board == []

    def test_preflop_actor_is_sb(self):
        # SB (player 0) acts first preflop
        engine = make_engine()
        state = engine.reset()
        assert state.acting_player == 0

    def test_street_is_preflop(self):
        engine = make_engine()
        state = engine.reset()
        assert state.street == Street.PREFLOP


# ---------------------------------------------------------------------------
# Fold
# ---------------------------------------------------------------------------

class TestFold:
    def test_fold_ends_hand(self):
        engine = make_engine()
        state = engine.reset()
        state = engine.step(Action.FOLD)
        assert state.hand_over

    def test_fold_correct_winner(self):
        engine = make_engine()
        state = engine.reset()
        # Player 0 (SB) folds → player 1 wins
        state = engine.step(Action.FOLD)
        assert state.winner == 1

    def test_fold_winner_collects_pot(self):
        engine = make_engine()
        state = engine.reset()
        pot_before = state.pot
        winner_stack_before = state.stacks[1]
        state = engine.step(Action.FOLD)
        assert state.stacks[1] == winner_stack_before + pot_before

    def test_fold_reward_signs(self):
        engine = make_engine()
        state = engine.reset()
        state = engine.step(Action.FOLD)
        # Folder should have negative reward, winner positive
        assert state.rewards[1] > 0   # winner
        assert state.rewards[0] < 0   # loser

    def test_step_after_hand_over_raises(self):
        engine = make_engine()
        engine.reset()
        engine.step(Action.FOLD)
        with pytest.raises(RuntimeError):
            engine.step(Action.CALL)


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

class TestCall:
    def test_call_reduces_stack(self):
        engine = make_engine()
        state = engine.reset()
        # SB needs to call 5 more to match BB's 10
        state = engine.step(Action.CALL)
        assert state.stacks[0] == 990   # 1000 - 5(SB) - 5(call)

    def test_call_increases_pot(self):
        engine = make_engine()
        state = engine.reset()
        state = engine.step(Action.CALL)   # SB calls
        assert state.pot == 20

    def test_call_advances_street_when_bb_checks(self):
        engine = make_engine()
        state = engine.reset()
        state = engine.step(Action.CALL)   # SB calls
        assert not state.hand_over
        assert state.acting_player == 1    # BB's turn
        state = engine.step(Action.CALL)   # BB checks (call = 0)
        assert state.street == Street.FLOP

    def test_flop_has_three_board_cards(self):
        engine = make_engine()
        state = engine.reset()
        engine.step(Action.CALL)
        state = engine.step(Action.CALL)
        assert len(state.board) == 3

    def test_board_cards_are_valid(self):
        engine = make_engine()
        state = engine.reset()
        engine.step(Action.CALL)
        state = engine.step(Action.CALL)
        for c in state.board:
            assert 0 <= c <= 51


# ---------------------------------------------------------------------------
# Raise
# ---------------------------------------------------------------------------

class TestRaise:
    def test_raise_increases_pot(self):
        engine = make_engine()
        state = engine.reset()
        pot_before = state.pot
        state = engine.step(Action.RAISE)  # 0.5x pot raise
        assert state.pot > pot_before

    def test_raise_reduces_stack(self):
        engine = make_engine()
        state = engine.reset()
        stack_before = state.stacks[0]
        state = engine.step(Action.RAISE)
        assert state.stacks[0] < stack_before

    def test_raise_forces_opponent_to_act(self):
        engine = make_engine()
        state = engine.reset()
        state = engine.step(Action.RAISE)
        assert not state.hand_over
        assert state.acting_player == 1   # BB must respond

    def test_raise_call_advances_street(self):
        engine = make_engine()
        state = engine.reset()
        engine.step(Action.RAISE)         # SB raises
        state = engine.step(Action.CALL)  # BB calls
        assert state.street == Street.FLOP

    def test_raise_fold_ends_hand(self):
        engine = make_engine()
        state = engine.reset()
        engine.step(Action.RAISE)
        state = engine.step(Action.FOLD)
        assert state.hand_over
        assert state.winner == 0

    def test_reraise_is_legal(self):
        engine = make_engine()
        state = engine.reset()
        engine.step(Action.RAISE)         # SB raises
        state = engine.step(Action.RAISE) # BB re-raises
        assert not state.hand_over
        assert state.acting_player == 0

    def test_invalid_raise_index_raises(self):
        engine = make_engine()
        engine.reset()
        with pytest.raises(ValueError):
            engine.step(99)   # out of range


# ---------------------------------------------------------------------------
# Street progression
# ---------------------------------------------------------------------------

class TestStreetProgression:
    def _play_to_flop(self, engine):
        engine.reset()
        engine.step(Action.CALL)   # SB calls
        return engine.step(Action.CALL)  # BB checks

    def test_full_progression_reaches_river(self):
        engine = make_engine()
        state = self._play_to_flop(engine)
        assert state.street == Street.FLOP

        engine.step(Action.CALL)   # BB checks (acts first postflop)
        state = engine.step(Action.CALL)  # SB checks
        assert state.street == Street.TURN
        assert len(state.board) == 4

        engine.step(Action.CALL)
        state = engine.step(Action.CALL)
        assert state.street == Street.RIVER
        assert len(state.board) == 5

    def test_showdown_at_river_end(self):
        engine = make_engine()
        self._play_to_flop(engine)
        # 2 checks on flop, 2 on turn, 2 on river = 6 more calls
        for _ in range(5):
            engine.step(Action.CALL)
        state = engine.step(Action.CALL)
        assert state.hand_over

    def test_showdown_has_winner_or_chop(self):
        engine = make_engine()
        self._play_to_flop(engine)
        for _ in range(5):
            engine.step(Action.CALL)
        state = engine.step(Action.CALL)
        assert state.winner in (-1, 0, 1)

    def test_postflop_bb_acts_first(self):
        engine = make_engine()
        state = self._play_to_flop(engine)
        assert state.acting_player == 1   # BB acts first postflop

    def test_bets_reset_each_street(self):
        engine = make_engine()
        state = self._play_to_flop(engine)
        assert state.bets == [0, 0]


# ---------------------------------------------------------------------------
# Legal actions
# ---------------------------------------------------------------------------

class TestLegalActions:
    def test_fold_and_call_always_legal(self):
        engine = make_engine()
        engine.reset()
        actions = engine.legal_actions()
        assert Action.FOLD in actions
        assert Action.CALL in actions

    def test_raise_legal_with_chips(self):
        engine = make_engine()
        engine.reset()
        actions = engine.legal_actions()
        assert any(a >= Action.RAISE for a in actions)

    def test_num_raise_actions_matches_config(self):
        engine = make_engine(raise_sizes=[0.5, 1.0, 2.0])
        engine.reset()
        actions = engine.legal_actions()
        raise_actions = [a for a in actions if a >= Action.RAISE]
        assert len(raise_actions) <= 3   # may be fewer if some sizes unaffordable


# ---------------------------------------------------------------------------
# Chip conservation
# ---------------------------------------------------------------------------

class TestChipConservation:
    def _total_chips(self, state):
        return state.stacks[0] + state.stacks[1] + state.pot

    def test_chips_conserved_after_blinds(self):
        engine = make_engine()
        state = engine.reset()
        assert self._total_chips(state) == 2000

    def test_chips_conserved_after_raise(self):
        engine = make_engine()
        state = engine.reset()
        state = engine.step(Action.RAISE)
        assert self._total_chips(state) == 2000

    def test_chips_conserved_after_fold(self):
        engine = make_engine()
        state = engine.reset()
        state = engine.step(Action.FOLD)
        # After hand over, pot is distributed — total still 2000
        assert state.stacks[0] + state.stacks[1] == 2000

    def test_chips_conserved_at_showdown(self):
        engine = make_engine()
        engine.reset()
        # 2 calls preflop + 2 checks per postflop street x3 = 8 total
        for _ in range(7):
            engine.step(Action.CALL)
        state = engine.step(Action.CALL)
        assert state.stacks[0] + state.stacks[1] == 2000


# ---------------------------------------------------------------------------
# StateEncoder
# ---------------------------------------------------------------------------

class TestStateEncoder:
    def setup_method(self):
        self.encoder = StateEncoder()
        self.engine = make_engine()
        self.state = self.engine.reset()

    def test_obs_shape(self):
        obs = self.encoder.encode(self.state, player_id=0)
        assert obs.shape == (17,)

    def test_obs_dtype(self):
        obs = self.encoder.encode(self.state, player_id=0)
        assert obs.dtype == np.float32

    def test_hole_cards_normalised(self):
        obs = self.encoder.encode(self.state, player_id=0)
        assert 0.0 <= obs[0] <= 1.0
        assert 0.0 <= obs[1] <= 1.0

    def test_undealt_board_slots_are_minus_one_preflop(self):
        obs = self.encoder.encode(self.state, player_id=0)
        # All 5 board slots should be -1 preflop
        assert all(obs[2 + i] == -1.0 for i in range(5))

    def test_board_slots_filled_after_flop(self):
        self.engine.step(Action.CALL)
        state = self.engine.step(Action.CALL)  # advances to flop
        obs = self.encoder.encode(state, player_id=0)
        # First 3 slots filled
        assert all(obs[2 + i] >= 0.0 for i in range(3))
        # Last 2 still -1
        assert all(obs[2 + i] == -1.0 for i in range(3, 5))

    def test_street_one_hot_preflop(self):
        obs = self.encoder.encode(self.state, player_id=0)
        street_slice = obs[7:11]
        assert street_slice[Street.PREFLOP] == 1.0
        assert street_slice.sum() == 1.0

    def test_street_one_hot_flop(self):
        self.engine.step(Action.CALL)
        state = self.engine.step(Action.CALL)
        obs = self.encoder.encode(state, player_id=0)
        street_slice = obs[7:11]
        assert street_slice[Street.FLOP] == 1.0
        assert street_slice.sum() == 1.0

    def test_position_player0_is_ip(self):
        obs = self.encoder.encode(self.state, player_id=0)
        assert obs[11] == 1.0   # player 0 = BTN = in position

    def test_position_player1_is_oop(self):
        obs = self.encoder.encode(self.state, player_id=1)
        assert obs[11] == 0.0   # player 1 = BB = out of position

    def test_chip_features_normalised(self):
        obs = self.encoder.encode(self.state, player_id=0)
        chip_slice = obs[12:17]
        assert all(v >= 0.0 for v in chip_slice)
        assert all(v <= 1.0 for v in chip_slice)

    def test_pot_feature_correct(self):
        obs = self.encoder.encode(self.state, player_id=0)
        # Pot after blinds = 15, starting stack = 1000
        assert abs(obs[14] - 15 / 1000) < 1e-5

    def test_perspectives_differ(self):
        obs0 = self.encoder.encode(self.state, player_id=0)
        obs1 = self.encoder.encode(self.state, player_id=1)
        # Hole cards differ
        assert not np.array_equal(obs0[:2], obs1[:2])
        # Position bit differs
        assert obs0[11] != obs1[11]

    def test_encode_both_matches_individual(self):
        obs0, obs1 = self.encoder.encode_both(self.state)
        assert np.array_equal(obs0, self.encoder.encode(self.state, 0))
        assert np.array_equal(obs1, self.encoder.encode(self.state, 1))

    def test_feature_names_length(self):
        assert len(self.encoder.feature_names()) == self.encoder.obs_size

    def test_obs_size_attribute(self):
        assert self.encoder.obs_size == 17


if __name__ == "__main__":
    pytest.main([__file__, "-v"])