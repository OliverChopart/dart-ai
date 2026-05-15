"""One-time setup script for the dart-ai project.

Run once after cloning:
    python scripts/setup.py
"""

import os
import subprocess
import sys


def run(cmd: str, check: bool = True) -> None:
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=check)


def main() -> None:
    print("\n=== dart-ai setup ===")

    # 1. Create .env
    if not os.path.exists(".env"):
        print("\n[1/4] Creating .env from .env.example")
        run("cp .env.example .env")
    else:
        print("\n[1/4] .env already exists - skipping")

    # 2. Install Python dependencies
    print("\n[2/4] Installing Python dependencies via uv")
    run("uv sync")

    # 3. Create PostgreSQL database
    print("\n[3/4] Creating PostgreSQL database 'dartai'")
    run("createdb dartai", check=False)

    # 4. Run migrations
    print("\n[4/4] Running database migrations")
    run("uv run alembic upgrade head")

    print("\n=== Setup complete ===")
    print("Start the server:")
    print("  uv run python -m backend")
    print("Then open: http://localhost:8000/health")


if __name__ == "__main__":
    main()
