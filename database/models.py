"""SQLAlchemy ORM models for the dart-ai database."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    total_matches: Mapped[int] = mapped_column(Integer, default=0)
    avg_score_per_throw: Mapped[float] = mapped_column(Float, default=0.0)

    match_players: Mapped[list["MatchPlayer"]] = relationship(back_populates="player", cascade="all, delete-orphan")


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="active")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    winner_player_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=True)

    match_players: Mapped[list["MatchPlayer"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    throws: Mapped[list["Throw"]] = relationship(back_populates="match", cascade="all, delete-orphan")


class MatchPlayer(Base):
    __tablename__ = "match_players"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False)
    player_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=False)
    turn_order: Mapped[int] = mapped_column(Integer, nullable=False)
    score_remaining: Mapped[int] = mapped_column(Integer, default=301)
    is_winner: Mapped[bool] = mapped_column(Boolean, default=False)

    match: Mapped["Match"] = relationship(back_populates="match_players")
    player: Mapped["Player"] = relationship(back_populates="match_players")
    throws: Mapped[list["Throw"]] = relationship(back_populates="match_player", cascade="all, delete-orphan")


class Throw(Base):
    __tablename__ = "throws"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False)
    match_player_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("match_players.id"), nullable=False)
    throw_number: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    segment: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    dart_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dart_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thrown_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    match: Mapped["Match"] = relationship(back_populates="throws")
    match_player: Mapped["MatchPlayer"] = relationship(back_populates="throws")
