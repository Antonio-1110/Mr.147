"""Moved: the simulated snooker table is mr147.snooker.simulated_table, its layouts are
mr147.snooker.layouts (pool: mr147.pool.simulated_table, mr147.pool.layouts).
This name keeps old imports working."""

from mr147.snooker.layouts import after_break_off, full_rack, open_table
from mr147.snooker.simulated_table import SimulatedTable

__all__ = ["SimulatedTable", "after_break_off", "full_rack", "open_table"]
