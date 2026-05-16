"""Quick terminal test of 301 game logic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.game.game_301 import Game301


def main() -> None:
    print("=== Test: 2-player 301 ===")
    game = Game301(player_names=["Alice", "Bob"])
    game.start()

    # Alice tur 1: T20 x3 = 180 -> 121 tilbage
    print("\n--- Alice tur 1 ---")
    for _ in range(3):
        print(game.register_throw(score=60, segment="T20", confidence=0.95).message)
    print(game.scoreboard())

    # Bob tur 1: 20 x3 = 60 -> 241 tilbage
    print("\n--- Bob tur 1 ---")
    for _ in range(3):
        print(game.register_throw(score=20, segment="20", confidence=0.90).message)
    print(game.scoreboard())

    # Alice tur 2: 20 x3 = 60 -> 61 tilbage
    print("\n--- Alice tur 2 ---")
    for _ in range(3):
        print(game.register_throw(score=20, segment="20", confidence=0.90).message)
    print(game.scoreboard())

    # Bob tur 2: 20 x3 = 60 -> 181 tilbage
    print("\n--- Bob tur 2 ---")
    for _ in range(3):
        print(game.register_throw(score=20, segment="20", confidence=0.90).message)
    print(game.scoreboard())

    # Alice tur 3: BUST (61-60=1, reset til 61)
    print("\n--- Alice tur 3 (BUST test) ---")
    for _ in range(3):
        print(game.register_throw(score=20, segment="20", confidence=0.90).message)
    print(game.scoreboard())

    # Bob tur 3: 20 x3 = 60 -> 121 tilbage
    print("\n--- Bob tur 3 ---")
    for _ in range(3):
        print(game.register_throw(score=20, segment="20", confidence=0.90).message)
    print(game.scoreboard())

    # Alice tur 4: 20 + 20 + 21 = 61 -> WIN
    print("\n--- Alice tur 4 (WIN) ---")
    print(game.register_throw(score=20, segment="20", confidence=0.95).message)
    print(game.register_throw(score=20, segment="20", confidence=0.95).message)
    r = game.register_throw(score=21, segment="T7", confidence=0.95)
    print(r.message)
    print(f"\nGame over: {r.game_over}")
    print(game.scoreboard())


if __name__ == "__main__":
    main()