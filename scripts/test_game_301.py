"""Quick terminal test of 301 game logic.

Simulates a 2-player game with known throws to verify scoring,
bust detection and win condition.

Usage:
    uv run python scripts/test_game_301.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.game.game_301 import Game301


def main() -> None:
    print("=== Test: 2-player 301 ===")
    game = Game301(player_names=["Alice", "Bob"])
    game.start()

    # Alice's first turn: T20, T20, T20 = 180
    for _ in range(3):
        r = game.register_throw(score=60, segment="T20", confidence=0.95)
        print(r.message)

    print(game.scoreboard())

    # Bob's first turn: 20, 20, 20 = 60
    for _ in range(3):
        r = game.register_throw(score=20, segment="20", confidence=0.90)
        print(r.message)

    print(game.scoreboard())

    # Alice's second turn: T20, T20, T19 = 117 -> 301-180-117 = 4 remaining
    r = game.register_throw(score=60, segment="T20", confidence=0.95)
    print(r.message)
    r = game.register_throw(score=60, segment="T20", confidence=0.95)
    print(r.message)
    r = game.register_throw(score=57, segment="T19", confidence=0.95)
    print(r.message)

    print(game.scoreboard())

    # Alice's third turn: test bust (4 remaining, throws 5)
    r = game.register_throw(score=5, segment="5", confidence=0.90)
    print(r.message)  # should be BUST, reset to 4

    print(game.scoreboard())

    # Alice wins: throw D2 = 4
    r = game.register_throw(score=4, segment="D2", confidence=0.95)
    print(r.message)
    print(f"Game over: {r.game_over}")
    print(f"Winner: {game.state.winner_id}")
    print(game.scoreboard())


if __name__ == "__main__":
    main()
