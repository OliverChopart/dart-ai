"""Start a 501 game with live iPhone camera detection.

Usage:
    uv run python scripts/play_501.py
    uv run python scripts/play_501.py --players "Alice" "Bob"
    uv run python scripts/play_501.py --players "Alice" "Bob" "Charlie" "Dave"

Keyboard (i preview-vinduet):
    K      — kalibrer (kan bruges når som helst)
    SPACE  — score pil
    ENTER  — ny tur (fjern pile fra skiven)
    Q      — afslut

Keyboard (i terminalen):
    ENTER       — ny tur (fjern pile fra skiven)
    u + ENTER   — fortryd sidste kast
    s + ENTER   — vis scoreboard
    q + ENTER   — afslut
"""

import argparse
import logging
import select
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Slå debug/info logs fra — kun warnings og fejl vises i terminalen
logging.basicConfig(level=logging.WARNING)

from backend.game.game_501 import ThrowResult
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


def read_terminal_input() -> str | None:
    """Non-blocking check for terminal input. Returns stripped line or None."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip().lower()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Spil 501 med live dart detection")
    parser.add_argument(
        "--players",
        nargs="+",
        default=["Spiller 1"],
        help="Spillernavne (1-4)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Deaktiver live video preview",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("       DART-AI — 501")
    print("=" * 50)
    print(f"Spillere: {', '.join(args.players)}")
    print("=" * 50)
    print("\nFør spillet starter:")
    print("  1. Sørg for at hele dartskiven er synlig i kamera-vinduet")
    print("  2. Tryk SPACE i kamera-vinduet for at kalibrere")
    print("  3. Vent på '✅ Kalibrering lykkedes!' i terminalen")
    print("\nKommandoer under spillet:")
    print("  ENTER     — ny tur (fjern pile fra skiven)")
    print("  u + ENTER — fortryd sidste kast")
    print("  s + ENTER — vis scoreboard")
    print("  q + ENTER — afslut")
    print("=" * 50)

    session = GameSession(
        player_names=args.players,
        show_preview=not args.no_preview,
        on_throw=on_throw,
    )

    print("\nStarter kamera og indlæser model...")
    session.start()
    print_scoreboard(session)
    print(f"\n{session.game.state.current_player.display_name} starter!")
    print("Kast din første pil...")

    try:
        while session.game.state.status == GameStatus.ACTIVE:
            # Tick preview — skal ske på main thread (macOS krav)
            if not session.tick_preview():
                break

            # Non-blocking terminal input
            cmd = read_terminal_input()
            if cmd is None:
                continue

            if cmd == "q":
                break
            elif cmd == "u":
                print(session.undo())
                print_scoreboard(session)
            elif cmd == "s":
                print_scoreboard(session)
            elif cmd == "":
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
