"""Game session — bridges DartPipeline events to Game301 logic."""

from __future__ import annotations

from backend.game.game_301 import Game301, ThrowResult
from backend.game.models import GameStatus
from utils.logging import get_logger
from vision.pipeline import DartPipeline, ScoreEvent

logger = get_logger(__name__)


class GameSession:
    """Manages a single 301 game session with live dart detection.

    Args:
        player_names: 1-4 player names.
        model_path: Path to trained YOLO model.
        show_preview: Show OpenCV video preview.
        on_throw: Optional callback called after each registered throw.
    """

    def __init__(
        self,
        player_names: list[str],
        model_path: str | None = None,
        show_preview: bool = True,
        on_throw: callable = None,
    ) -> None:
        self._game = Game301(player_names=player_names)
        self._on_throw = on_throw
        self._pipeline = DartPipeline(
            on_score_callback=self._on_score_event,
            model_path=model_path,
            show_preview=show_preview,
        )
        self._throws_this_turn = 0

    def start(self) -> None:
        """Start the game and pipeline."""
        self._game.start()
        self._pipeline.start()
        logger.info(
            "session started",
            players=[p.display_name for p in self._game.state.players],
        )

    def stop(self) -> None:
        """Stop the pipeline."""
        self._pipeline.stop()

    def undo(self) -> str:
        """Undo the last throw."""
        msg = self._game.undo_last_throw()
        self._throws_this_turn = max(0, self._throws_this_turn - 1)
        return msg

    def new_turn(self) -> None:
        """Call when darts are physically removed from the board."""
        self._throws_this_turn = 0
        self._pipeline.reset_dart_count()
        logger.info(
            "new turn",
            player=self._game.state.current_player.display_name,
        )

    def tick_preview(self) -> bool:
        """Display the latest camera frame and handle keyboard input.

        Must be called from the main thread on every loop iteration.
        Returns False if the user pressed Q (quit).
        """
        return self._pipeline.tick_preview()

    @property
    def game(self) -> Game301:
        return self._game

    @property
    def pipeline(self) -> DartPipeline:
        return self._pipeline

    def _on_score_event(self, event: ScoreEvent) -> None:
        """Called by pipeline when a new dart is detected."""
        if self._game.state.status != GameStatus.ACTIVE:
            return

        new_dart_index = event.dart_count - 1
        if new_dart_index < 0 or new_dart_index >= len(event.results):
            return

        result = event.results[new_dart_index]
        throw_result = self._game.register_throw(
            score=result.score,
            segment=result.segment,
            confidence=result.confidence,
            dart_x=result.board_x,
            dart_y=result.board_y,
        )

        self._throws_this_turn += 1

        if self._on_throw:
            self._on_throw(throw_result)

        logger.info(
            "throw registered via pipeline",
            segment=result.segment,
            score=result.score,
            message=throw_result.message,
        )
