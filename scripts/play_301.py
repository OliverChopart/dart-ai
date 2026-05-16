"""Start a 301 game with live iPhone camera detection.

Usage:
    uv run python scripts/play_301.py
    uv run python scripts/play_301.py --players "Alice" "Bob"
    uv run python scripts/play_301.py --players "Alice" "Bob" "Charlie" "Dave"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.game.game_301 import ThrowResult
from backend.game.models import GameStatus
from backend.game.session import GameSession


def on_throw(result: ThrowResult) -> None:
    """Print throw result to terminal."""
    print(f"\n🎯 {result.message}")
    if result.turn_complete and not result.game_over:
        print("─" * 50)
        print(f"Næste spiller: {result.next_player_name}")
        print("Fjern pile fra skiven og tryk ENTER...")
    if result.game_over:
        print("\n🏆 SPILLET ER SLUT! 🏆")


def print_scoreboard(session: GameSession) -> None:
    print("\n" + session.game.scoreboard())


def main() -> None:
    parser = argparse.ArgumentParser(description="Spil 301 med live dart detection")
    parser.add_argument(
        "--players",
        nargs="+",
        default=["Spiller 1"],
        help="Spillernavne (1-4)",
    )
    parser.add_argument(
        "--model",
        default="runs/detect/runs/train/dart_25pct/weights/best.pt",
        help="Sti til YOLO model",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Deaktiver live video preview",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("       DART-AI — 301")
    print("=" * 50)
    print(f"Spillere: {', '.join(args.players)}")
    print(f"Model: {args.model}")
    print("=" * 50)
    print("\nKommandoer under spillet:")
    print("  ENTER     — ny tur (fjern pile fra skiven)")
    print("  u + ENTER — fortryd sidste kast")
    print("  s + ENTER — vis scoreboard")
    print("  q + ENTER — afslut")
    print("=" * 50)

    session = GameSession(
        player_names=args.players,
        model_path=args.model,
        show_preview=not args.no_preview,
        on_throw=on_throw,
    )

    print(f"\nStarter kamera og indlæser model...")
    session.start()
    print_scoreboard(session)
    print(f"\n{session.game.state.current_player.display_name} starter!")
    print("Kast din første pil...")

    try:
        while session.game.state.status == GameStatus.ACTIVE:
            cmd = input().strip().lower()

            if cmd == "q":
                break
            elif cmd == "u":
                print(session.undo())
                print_scoreboard(session)
            elif cmd == "s":
                print_scoreboard(session)
            elif cmd == "":
                # New turn — darts removed from board
                session.new_turn()
                print(f"\n{session.game.state.current_player.display_name}s tur!")
                print("Kast din første pil...")

    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        print("\nSpillet afsluttet.")
        print_scoreboard(session)


if __name__ == "__main__":
    main()