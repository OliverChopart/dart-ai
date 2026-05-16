"""Core data models for game state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4


class GameStatus(str, Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    FINISHED = "finished"


class ThrowStatus(str, Enum):
    VALID = "valid"
    BUST = "bust"       # score went below 0 or to 1
    WIN = "win"


@dataclass
class Throw:
    """A single dart throw."""
    score: int
    segment: str
    confidence: float
    dart_x: float | None = None
    dart_y: float | None = None
    status: ThrowStatus = ThrowStatus.VALID


@dataclass
class PlayerState:
    """State of a single player in a 301 game."""
    player_id: UUID
    display_name: str
    score_remaining: int = 301
    throws: list[Throw] = field(default_factory=list)
    turns_played: int = 0
    has_won: bool = False

    @property
    def throws_this_turn(self) -> list[Throw]:
        """The throws in the current (incomplete) turn."""
        start = self.turns_played * 3
        return self.throws[start:]

    @property
    def total_throws(self) -> int:
        return len(self.throws)

    @property
    def average_per_turn(self) -> float:
        if self.turns_played == 0:
            return 0.0
        total_scored = 301 - self.score_remaining
        return total_scored / self.turns_played


@dataclass
class GameState:
    """Complete state of a 301 game."""
    game_id: UUID = field(default_factory=uuid4)
    players: list[PlayerState] = field(default_factory=list)
    current_player_index: int = 0
    status: GameStatus = GameStatus.WAITING
    winner_id: UUID | None = None

    @property
    def current_player(self) -> PlayerState:
        return self.players[self.current_player_index]

    @property
    def throws_this_turn(self) -> int:
        return len(self.current_player.throws_this_turn)
