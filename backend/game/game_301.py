"""301 game logic.

Rules:
- Each player starts at 301 points.
- Each turn a player throws up to 3 darts.
- The score of each dart is subtracted from the remaining score.
- A player wins by reaching exactly 0.
- If a throw would take the score below 0 or to exactly 1, it is a BUST:
  the turn ends immediately and the player's score resets to what it was
  at the start of the turn.
- Scores are registered one dart at a time.

Usage::

    game = Game301(players=["Alice", "Bob"])
    game.start()

    result = game.register_throw(score=20, segment="20", confidence=0.9)
    print(result)  # ThrowResult
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from backend.game.models import (
    GameState,
    GameStatus,
    PlayerState,
    Throw,
    ThrowStatus,
)
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ThrowResult:
    """Result of registering a single throw."""
    throw: Throw
    player: PlayerState
    turn_complete: bool       # True if this was the 3rd dart or a bust/win
    game_over: bool
    next_player_name: str | None
    message: str


class Game301:
    """301 game engine supporting 1-4 players.

    Args:
        player_names: List of 1-4 display names.
        starting_score: Starting score (default 301).
    """

    MAX_PLAYERS = 4
    THROWS_PER_TURN = 3

    def __init__(
        self,
        player_names: list[str],
        starting_score: int = 301,
    ) -> None:
        if not 1 <= len(player_names) <= self.MAX_PLAYERS:
            raise ValueError(
                f"Number of players must be between 1 and {self.MAX_PLAYERS}, "
                f"got {len(player_names)}."
            )

        self._starting_score = starting_score
        self._state = GameState(
            players=[
                PlayerState(
                    player_id=uuid4(),
                    display_name=name,
                    score_remaining=starting_score,
                )
                for name in player_names
            ]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> GameState:
        """Current game state (read-only snapshot)."""
        return self._state

    def start(self) -> None:
        """Transition the game from WAITING to ACTIVE."""
        if self._state.status != GameStatus.WAITING:
            raise RuntimeError("Game is already started.")
        self._state.status = GameStatus.ACTIVE
        logger.info(
            "game started",
            players=[p.display_name for p in self._state.players],
            starting_score=self._starting_score,
        )

    def register_throw(
        self,
        score: int,
        segment: str,
        confidence: float = 1.0,
        dart_x: float | None = None,
        dart_y: float | None = None,
    ) -> ThrowResult:
        """Register a single dart throw for the current player.

        Args:
            score: Points scored (0-60).
            segment: Segment label e.g. 'T20', 'D16', 'Bull', 'Miss'.
            confidence: YOLO detection confidence.
            dart_x: Normalised board x coordinate.
            dart_y: Normalised board y coordinate.

        Returns:
            ThrowResult describing what happened.
        """
        if self._state.status != GameStatus.ACTIVE:
            raise RuntimeError("Game is not active.")

        player = self._state.current_player
        score_before_turn = self._score_at_turn_start(player)

        throw = Throw(
            score=score,
            segment=segment,
            confidence=confidence,
            dart_x=dart_x,
            dart_y=dart_y,
        )

        new_score = player.score_remaining - score

        # --- Win ---
        if new_score == 0:
            throw.status = ThrowStatus.WIN
            player.score_remaining = 0
            player.throws.append(throw)
            player.has_won = True
            self._state.status = GameStatus.FINISHED
            self._state.winner_id = player.player_id
            player.turns_played += 1
            logger.info("game won", player=player.display_name)
            return ThrowResult(
                throw=throw,
                player=player,
                turn_complete=True,
                game_over=True,
                next_player_name=None,
                message=f"{player.display_name} vinder! 🎯",
            )

        # --- Bust ---
        if new_score < 0 or new_score == 1:
            throw.status = ThrowStatus.BUST
            player.throws.append(throw)
            # Reset score to start of turn
            player.score_remaining = score_before_turn
            player.turns_played += 1
            self._advance_player()
            logger.info(
                "bust",
                player=player.display_name,
                score_reset_to=score_before_turn,
            )
            return ThrowResult(
                throw=throw,
                player=player,
                turn_complete=True,
                game_over=False,
                next_player_name=self._state.current_player.display_name,
                message=f"BUST! {player.display_name} går tilbage til {score_before_turn}.",
            )

        # --- Normal throw ---
        player.score_remaining = new_score
        player.throws.append(throw)

        throws_this_turn = len(player.throws_this_turn)
        turn_complete = throws_this_turn >= self.THROWS_PER_TURN

        if turn_complete:
            player.turns_played += 1
            self._advance_player()
            next_name = self._state.current_player.display_name
            message = (
                f"{player.display_name} scorede {score} ({segment}). "
                f"Resterende: {player.score_remaining}. "
                f"{next_name}s tur."
            )
        else:
            next_name = None
            darts_left = self.THROWS_PER_TURN - throws_this_turn
            message = (
                f"{player.display_name}: {segment} = {score} pts. "
                f"Resterende: {player.score_remaining}. "
                f"{darts_left} pil(e) tilbage."
            )

        logger.info(
            "throw registered",
            player=player.display_name,
            segment=segment,
            score=score,
            remaining=player.score_remaining,
            turn_complete=turn_complete,
        )

        return ThrowResult(
            throw=throw,
            player=player,
            turn_complete=turn_complete,
            game_over=False,
            next_player_name=next_name,
            message=message,
        )

    def undo_last_throw(self) -> str:
        """Undo the last registered throw (useful for mis-detections)."""
        player = self._state.current_player
        if not player.throws_this_turn:
            return "Ingen kast at fortryde i dette tur."

        last = player.throws.pop()
        player.score_remaining += last.score
        logger.info("throw undone", player=player.display_name, segment=last.segment)
        return f"Fortrød {last.segment} ({last.score} pts). Resterende: {player.score_remaining}."

    def scoreboard(self) -> str:
        """Return a simple text scoreboard."""
        lines = ["=== 301 Scoreboard ==="]
        for i, p in enumerate(self._state.players):
            marker = " <<" if i == self._state.current_player_index and self._state.status == GameStatus.ACTIVE else ""
            lines.append(
                f"  {p.display_name}: {p.score_remaining} tilbage"
                f" | Snit: {p.average_per_turn:.1f}/tur{marker}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _advance_player(self) -> None:
        """Move to the next player in rotation."""
        self._state.current_player_index = (
            self._state.current_player_index + 1
        ) % len(self._state.players)

    def _score_at_turn_start(self, player: PlayerState) -> int:
        """Calculate what the player's score was at the start of this turn."""
        turn_throws = player.throws_this_turn
        return player.score_remaining + sum(t.score for t in turn_throws)
