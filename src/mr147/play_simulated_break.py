"""Moved to mr147.snooker.play (pool: mr147.pool.play). This name keeps the old command working:

    uv run python -m mr147.play_simulated_break --seed 0
"""

from mr147.snooker.play import main, robot_eyes

__all__ = ["main", "robot_eyes"]

if __name__ == "__main__":
    main()
