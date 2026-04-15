"""Unit tests for the phase-1 Infinite Hold'em runtime engine."""

from infinite_holdem import Card, HandRank, InfiniteDeck, InfiniteHandEvaluator, InfiniteHoldemGame


def test_duplicate_card_dealing_can_occur():
    deck = InfiniteDeck(seed=42)
    cards = deck.deal(200)
    assert len({str(card) for card in cards}) < len(cards)


def test_variant_rankings_cover_new_hand_types():
    evaluator = InfiniteHandEvaluator()

    flush_five = evaluator.evaluate([Card.from_string("As")] * 5)
    five_of_a_kind = evaluator.evaluate(
        [
            Card.from_string("Kh"),
            Card.from_string("Ks"),
            Card.from_string("Kd"),
            Card.from_string("Kc"),
            Card.from_string("Kh"),
        ]
    )
    flush_house = evaluator.evaluate(
        [
            Card.from_string("Qh"),
            Card.from_string("Qh"),
            Card.from_string("Qh"),
            Card.from_string("9h"),
            Card.from_string("9h"),
        ]
    )

    assert flush_five.rank == HandRank.FLUSH_FIVE
    assert five_of_a_kind.rank == HandRank.FIVE_OF_A_KIND
    assert flush_house.rank == HandRank.FLUSH_HOUSE


def test_straight_beats_flush():
    evaluator = InfiniteHandEvaluator()
    straight = evaluator.evaluate([Card.from_string(card) for card in ("9h", "8d", "7c", "6s", "5h")])
    flush = evaluator.evaluate([Card.from_string(card) for card in ("Kh", "Jh", "8h", "4h", "2h")])
    assert straight > flush


def test_preflop_actor_is_player_zero():
    game = InfiniteHoldemGame(seed=1)
    game.deal_hole_cards()
    assert game.street == "preflop"
    assert game.actor_index == 0


def test_postflop_actor_is_player_one():
    game = InfiniteHoldemGame(seed=2)
    game.deal_hole_cards()
    game.call(0)
    game.check(1)
    assert game.is_betting_round_complete()
    game.advance_street()
    assert game.street == "flop"
    assert game.actor_index == 1


def test_raises_reopen_action():
    game = InfiniteHoldemGame(seed=3)
    game.deal_hole_cards()
    game.bet_or_raise(0, 300)
    assert game.actor_index == 1
    game.bet_or_raise(1, 500)
    assert game.actor_index == 0
    assert game.get_to_call(0) == 200


def test_street_completion_waits_for_response():
    game = InfiniteHoldemGame(seed=4)
    game.deal_hole_cards()
    game.bet_or_raise(0, 300)
    assert not game.is_betting_round_complete()
    game.call(1)
    assert game.is_betting_round_complete()


def test_showdown_settlement_is_zero_sum():
    game = InfiniteHoldemGame(seed=5)
    game.deal_hole_cards()
    game.call(0)
    game.check(1)
    game.advance_street()
    game.check(1)
    game.check(0)
    game.advance_street()
    game.check(1)
    game.check(0)
    game.advance_street()
    game.check(1)
    game.check(0)
    _, payoffs = game.settle_pot()
    assert sum(payoffs) == 0


def test_fold_settlement_is_zero_sum():
    game = InfiniteHoldemGame(seed=6)
    game.deal_hole_cards()
    game.bet_or_raise(0, 200)
    game.fold(1)
    winners, payoffs = game.settle_pot()
    assert winners == [0]
    assert sum(payoffs) == 0


def test_split_pot_distributes_correctly():
    game = InfiniteHoldemGame(seed=7)
    game.deal_hole_cards()
    game.call(0)
    game.check(1)
    game.board = [Card.from_string(card) for card in ("Ah", "Kh", "Qh", "Jh", "Th")]
    game.street = "showdown"
    game.actor_index = None
    game.hand_over = True
    game._pending_actors = []
    winners, payoffs = game.settle_pot()

    assert winners == [0, 1]
    assert payoffs == [0, 0]
