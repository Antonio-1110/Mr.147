"""Play a shot on a TableState with pooltool and report what happened.

Works for any game: the table size, ball size and pocket openings come from a
GameTable (mr147.snooker.table.SNOOKER_TABLE, mr147.pool.table.POOL_TABLE, ...).

pooltool uses different coordinates from the rest of the code:

    ours:     (0, 0) at the table centre, x along the length, y across.
    pooltool: (0, 0) at a corner,         x across the width, y along the length.

So x and y swap and shift. All conversion happens in this file and nowhere else.
"""

import math
from dataclasses import dataclass

import attrs
import pooltool as pt
from pooltool.objects import PocketTableSpecs, SnookerTableSpecs

from mr147.game_table import GameTable, Position
from mr147.interfaces.cue_arm import Shot
from mr147.snooker.table import SNOOKER_TABLE
from mr147.table_memory.table_state import TableState

_CUSHION_EVENTS = {pt.EventType.BALL_LINEAR_CUSHION, pt.EventType.BALL_CIRCULAR_CUSHION}


@dataclass(frozen=True)
class ShotResult:
    final_table: TableState  # balls still on the table, where they stopped
    potted_ball_ids: tuple[str, ...]  # in the order they went down
    first_ball_hit: str | None  # the first ball the cue ball touched
    first_ball_hit_direction_deg: float | None  # which way that ball moved off
    cue_ball_path: tuple[Position, ...]  # cue ball position at each bounce/collision
    pockets_entered: tuple[int, ...] = ()  # pocket (index into PhysicsSimulator.pockets) for each potted ball
    cushion_after_first_hit: bool = False  # did any ball touch a cushion after the first contact?
    cue_ball_hits: tuple[str, ...] = ()  # every ball the cue ball touched, in order (first_ball_hit first)


def from_pooltool_velocity(pooltool_velocity) -> float:
    """Direction of a pooltool velocity vector, in our degrees."""
    vx, vy = pooltool_velocity[0], pooltool_velocity[1]
    return math.degrees(math.atan2(vx, vy))  # (vx, vy) in pooltool is (vy, vx) in ours


def to_pooltool_direction(direction_deg: float) -> float:
    """Swapping x and y mirrors angles about the 45 degree line: a -> 90 - a."""
    return (90.0 - direction_deg) % 360


class PhysicsSimulator:
    def __init__(self, table: GameTable = SNOOKER_TABLE):
        self.table = table
        if table.simulator_style == "snooker":
            specs_type, game_type = SnookerTableSpecs, pt.GameType.SNOOKER
        elif table.simulator_style == "pool":
            specs_type, game_type = PocketTableSpecs, pt.GameType.NINEBALL
        else:
            raise ValueError(f"unknown simulator_style {table.simulator_style!r}")

        specs = specs_type(
            l=table.length,
            w=table.width,
            corner_pocket_width=table.corner_pocket_opening,
            side_pocket_width=table.middle_pocket_opening,
        )
        self.pooltool_table = pt.Table.from_table_specs(specs)

        ball = pt.BallParams.default(game_type=game_type)
        self._ball_params = attrs.asdict(attrs.evolve(ball, R=table.ball_radius))

        pockets_by_name = sorted(self.pooltool_table.pockets.items())
        self.pocket_names: list[str] = [name for name, _ in pockets_by_name]
        self.pockets: list[Position] = [self.from_pooltool_position(p.center) for _, p in pockets_by_name]

    # ---- coordinate conversion -------------------------------------------------------

    def to_pooltool_position(self, position: Position) -> tuple[float, float]:
        x, y = position
        return (y + self.table.width / 2, x + self.table.length / 2)

    def from_pooltool_position(self, pooltool_xyz) -> Position:
        px, py = float(pooltool_xyz[0]), float(pooltool_xyz[1])
        return (py - self.table.length / 2, px - self.table.width / 2)

    # ---- simulation ------------------------------------------------------------------

    def simulate(self, table: TableState, shot: Shot) -> ShotResult:
        """Play `shot` on `table` and let every ball come to rest."""
        balls = {
            ball_id: pt.Ball.create(ball_id, xy=self.to_pooltool_position(position), **self._ball_params)
            for ball_id, position in table.positions.items()
        }
        cue = pt.Cue(cue_ball_id=self.table.cue_ball)
        cue.set_state(
            V0=shot.speed,
            phi=to_pooltool_direction(shot.direction_deg),
            theta=shot.elevation_deg,
            a=shot.side,
            b=shot.height,
        )
        system = pt.System(cue=cue, table=self.pooltool_table, balls=balls)
        pt.simulate(system, inplace=True)
        return self._read_result(system, cue_ball_start=table.positions[self.table.cue_ball])

    def _read_result(self, system: pt.System, cue_ball_start: Position) -> ShotResult:
        cue_id = self.table.cue_ball

        # Where everything ended up.
        final_positions = {}
        for ball_id, ball in system.balls.items():
            if ball.state.s != pt.constants.pocketed:
                final_positions[ball_id] = self.from_pooltool_position(ball.state.rvw[0])

        # Go through the events in time order.
        potted: list[str] = []
        pockets_entered: list[int] = []
        first_ball_hit = None
        first_ball_hit_direction = None
        cushion_after_first_hit = False
        cue_ball_hits: list[str] = []
        cue_ball_path = [cue_ball_start]
        for event in system.events:
            if event.event_type == pt.EventType.BALL_POCKET:
                ball_id = next(i for i in event.ids if i in system.balls)
                pocket_id = next(i for i in event.ids if i in self.pooltool_table.pockets)
                potted.append(ball_id)
                pockets_entered.append(self.pocket_names.index(pocket_id))

            if event.event_type in _CUSHION_EVENTS and first_ball_hit is not None:
                cushion_after_first_hit = True

            if cue_id in event.ids and event.event_type != pt.EventType.STICK_BALL:
                cue_ball_after = event.get_ball(cue_id, initial=False)
                cue_ball_path.append(self.from_pooltool_position(cue_ball_after.state.rvw[0]))

            is_cue_hit = event.event_type == pt.EventType.BALL_BALL and cue_id in event.ids
            if is_cue_hit:
                cue_ball_hits.append(next(i for i in event.ids if i != cue_id))
            if is_cue_hit and first_ball_hit is None:
                first_ball_hit = next(i for i in event.ids if i != cue_id)
                object_ball_after = event.get_ball(first_ball_hit, initial=False)
                first_ball_hit_direction = from_pooltool_velocity(object_ball_after.state.rvw[1])

        if cue_id in final_positions:
            cue_ball_path.append(final_positions[cue_id])

        return ShotResult(
            final_table=TableState(final_positions),
            potted_ball_ids=tuple(potted),
            first_ball_hit=first_ball_hit,
            first_ball_hit_direction_deg=first_ball_hit_direction,
            cue_ball_path=tuple(cue_ball_path),
            pockets_entered=tuple(pockets_entered),
            cushion_after_first_hit=cushion_after_first_hit,
            cue_ball_hits=tuple(cue_ball_hits),
        )
