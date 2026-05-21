"""Quick terminal test of 501 game logic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.game.game_501 import Game501


def main() -> None:
    print("=== Test: 2-player 501 ===")
    game = Game501(player_names=["Alice", "Bob"])
    game.start()

    # Alice tur 1: T20 x3 = 180 -> 321 tilbage
    print("\n--- Alice tur 1 ---")
    for _ in range(3):
        print(game.register_throw(score=60, segment="T20", confidence=0.95).message)
    print(game.scoreboard())

    # Bob tur 1: 20 x3 = 60 -> 441 tilbage
    print("\n--- Bob tur 1 ---")
    for _ in range(3):
        print(game.register_throw(score=20, segment="20", confidence=0.90).message)
    print(game.scoreboard())

    # Alice tur 2: T20 x3 = 180 -> 141 tilbage
    print("\n--- Alice tur 2 ---")
    for _ in range(3):
        print(game.register_throw(score=60, segment="T20", confidence=0.95).message)
    print(game.scoreboard())

    # Bob tur 2: 20 x3 = 60 -> 381 tilbage
    print("\n--- Bob tur 2 ---")
    for _ in range(3):
        print(game.register_throw(score=20, segment="20", confidence=0.90).message)
    print(game.scoreboard())

    # Alice tur 3: T20 x2 + T7 = 120 + 21 = 141 -> WIN
    print("\n--- Alice tur 3 (WIN) ---")
    print(game.register_throw(score=60, segment="T20", confidence=0.95).message)
    print(game.register_throw(score=60, segment="T20", confidence=0.95).message)
    r = game.register_throw(score=21, segment="T7", confidence=0.95)
    print(r.message)
    print(f"\nGame over: {r.game_over}")
    print(game.scoreboard())


if __name__ == "__main__":
    main()
