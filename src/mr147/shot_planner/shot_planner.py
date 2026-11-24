"""The shot search.

Works for any game. The game's own folder says what a shot must do (a ShotGoal)
and the simulator knows the table (a GameTable), so nothing here is snooker-
or pool-specific except the snooker shortcut `plan()`.

Given the table and the ball on, find the shot that pots it and leaves the cue
ball in a good place for the next ball. Cheapest filters first:

  1. List pots:        geometry lists every sensible (ball, pocket) pair;
                       keep the easiest few.
  2. Tune the aim:     ghost-ball aiming ignores "throw" (friction between the
                       balls), so adjust the direction in the simulator until the
                       object ball really heads for the pocket.
  3. Try variations:   simulate each pot at several speeds and spins. Keep those
                       that do the job (pot the ball on, cue ball stays on).
                       Score where the cue ball stops.
     Split search:     if balls are tied up, also try a wider range of speeds and
                       spins on the best pots, looking for shots that pot the ball
                       AND send the cue ball into the cluster ("split" shots).
  4. Stress test:      replay the best few with arm and camera error, then
                       re-test the top 3 with fresh replays (picking the best of
                       many short tests over-rates the winner). For every
                       replay that goes in, look one shot ahead: is there a next
                       ball to pot from where the cue ball stopped, and how open is
                       the table? A split changes the table differently every time,
                       so it is judged on these averages, never on one simulation.
  5. Choose:           score = chance it goes in
                               x chance there is a next ball afterwards (the safety net)
                               x quality of the position (easy next ball, open table).
                       This is "how likely the break carries on, and how well".

The robot then plays only this one shot, looks at the table again and plans the
next one (plan a little ahead, play one, replan).
"""

import random
from dataclasses import dataclass, field, replace

from mr147.interfaces.cue_arm import Shot
from mr147.physics import PhysicsSimulator, ShotResult
from mr147.shot_planner import pot_geometry
from mr147.shot_planner.pot_geometry import PotOption
from mr147.shot_planner.shot_goal import ShotGoal
from mr147.shot_planner.table_openness import ball_is_tied_up, table_openness
from mr147.shot_planner.uncertainty import Uncertainty
from mr147.snooker.balls import Colour
from mr147.snooker.table import Position
from mr147.table_memory.table_state import TableState


@dataclass
class PlannerSettings:
    pot_options_to_try: int = 6  # how many (ball, pocket) pairs to simulate
    speeds: tuple[float, ...] = (0.8, 1.3, 2.0, 3.0)  # metres per second
    heights: tuple[float, ...] = (-0.5, -0.25, 0.0, 0.25, 0.5)  # screw ... follow
    sides: tuple[float, ...] = (0.0,)  # side spin (off by default)

    aim_good_enough_deg: float = 0.02  # stop tuning when the object ball is this close
    aim_tuning_attempts: int = 4

    stress_tests_per_pot_option: int = 2  # best-position shots per (ball, pocket) to stress test
    stress_test_replays: int = 24
    # Picking the best of many short stress tests flatters the winner (some of its luck
    # is counted as skill). So the top few are tested again with fresh replays, and the
    # choice is made on those fresh numbers only.
    confirm_top: int = 3
    confirm_replays: int = 30
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    # 0 = only care about potting this ball, 1 = position counts as much as the pot.
    # Kept low on purpose: in a 147, one miss ends everything.
    position_weight: float = 0.4
    # Within "position": 0 = only the next ball matters, 1 = only how open the table is.
    # This is what makes the robot split a pack instead of leaving it tied up. It is
    # scaled by how tied up the table is before the shot: on an open table it counts
    # for nothing, so it never trades an easy next ball for openness it doesn't need.
    openness_weight: float = 0.35

    # Split search: only when the table is tied up (openness below `tied_up_below`).
    tied_up_below: float = 0.7
    split_search_pots: int = 2  # the best this-many pots get the wider search
    split_speeds: tuple[float, ...] = (1.6, 2.4, 3.2, 4.0)
    split_heights: tuple[float, ...] = (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6)
    split_sides: tuple[float, ...] = (-0.35, 0.0, 0.35)
    split_finalists_per_pot_option: int = 1  # split shots stress-tested per (ball, pocket)
    random_seed: int = 0


@dataclass(frozen=True)
class PlannedShot:
    shot: Shot
    target: str  # ball id being potted, e.g. "red_07"
    pocket: int  # index into PhysicsSimulator.pockets
    chance_of_potting: float  # share of stress-test replays that went in
    position_quality: float  # 0..1, how easy the next pot is from where the cue ball stops (average over the stress test)
    score: float
    cue_ball_stops_at: Position
    cue_ball_path: tuple[Position, ...]
    predicted_table: TableState  # the table after the shot (with any respot done)
    table_openness: float = 1.0  # 0..1, how open the table is afterwards (average over the stress test)
    chance_break_continues: float = 1.0  # of the replays that went in: share that left a next ball to pot
    splits_the_pack: bool = False  # the cue ball runs into tied-up balls after the pot


@dataclass(frozen=True)
class TriedShot:
    """One shot that potted the ball in the one perfect simulation (step 3), kept for pictures."""

    target: str
    pocket: int
    shot: Shot
    cue_ball_path: tuple[Position, ...]
    cue_ball_stops_at: Position
    position_value: float  # 0..1, what step 4 used to pick finalists
    splits_the_pack: bool


@dataclass(frozen=True)
class PlanSearch:
    """Everything one plan_for() call looked at, so a person can see why it chose what it did.

    tried:      every shot that potted the ball in the perfect simulation
    finalists:  the ones stress tested, with their scores (best first)
    confirmed:  the top few re-tested with fresh replays (best first; [0] is the choice)
    """

    table: TableState
    targets: tuple[str, ...]
    tried: tuple[TriedShot, ...]
    finalists: tuple["PlannedShot", ...]
    confirmed: tuple["PlannedShot", ...]

    @property
    def chosen(self) -> "PlannedShot | None":
        if self.confirmed:
            return self.confirmed[0]
        return self.finalists[0] if self.finalists else None


@dataclass(frozen=True)
class _Candidate:
    """A shot that pots the ball on in the simulator (before stress testing)."""

    pot: PotOption
    shot: Shot
    result: ShotResult
    table_after: TableState
    position_quality: float
    openness: float
    splits: bool = False  # the cue ball touched a tied-up ball after the first contact

    def position_value(self, openness_weight: float) -> float:
        return (1 - openness_weight) * self.position_quality + openness_weight * self.openness


class ShotPlanner:
    def __init__(self, physics: PhysicsSimulator, settings: PlannerSettings | None = None):
        self.physics = physics
        self.settings = settings or PlannerSettings()
        self.last_search: PlanSearch | None = None  # what the latest plan_for() looked at

    def plan(
        self,
        table: TableState,
        ball_on: Colour,
        next_ball_on: Colour | None,
        respotted_colours: set[Colour] = frozenset(),
    ) -> PlannedShot | None:
        """Snooker: the best shot to pot `ball_on` while playing position for `next_ball_on`.

        Returns None if no pot could be found.
        """
        from mr147.snooker.shot_goal import snooker_goal  # snooker imports the planner, not the other way round

        return self.plan_for(table, snooker_goal(table, ball_on, next_ball_on, respotted_colours))

    def plan_for(self, table: TableState, goal: ShotGoal) -> PlannedShot | None:
        """Any game: the best shot that does what `goal` asks. None if no pot could be found."""
        rng = random.Random(self.settings.random_seed)
        openness_now = self._openness(table, goal)
        self._openness_weight = self.settings.openness_weight * (1.0 - openness_now)
        self._tied_up = self._tied_up_balls(table, goal)

        pots = pot_geometry.all_pot_options(table, goal.targets, self.physics.pockets, self.physics.table)
        pots = pots[: self.settings.pot_options_to_try]

        candidates = self._shots_that_pot(table, pots, goal)
        if openness_now < self.settings.tied_up_below and self._tied_up:
            candidates += self._split_shots(table, pots[: self.settings.split_search_pots], goal)
        tried = tuple(self._tried(c) for c in candidates)
        self.last_search = PlanSearch(table, tuple(sorted(goal.targets)), tried, (), ())
        if not candidates:
            return None

        tested = [(self._stress_test(table, c, goal, rng), c) for c in self._pick_finalists(candidates)]
        tested.sort(key=lambda pair: pair[0].score, reverse=True)
        finalists = tuple(p for p, _ in tested)
        confirmed: tuple[PlannedShot, ...] = ()
        if self.settings.confirm_top and self.settings.confirm_replays and self.settings.stress_test_replays:
            top = tested[: self.settings.confirm_top]
            tested = [(self._stress_test(table, c, goal, rng, self.settings.confirm_replays), c) for _, c in top]
            tested.sort(key=lambda pair: pair[0].score, reverse=True)
            confirmed = tuple(p for p, _ in tested)
        self.last_search = PlanSearch(table, tuple(sorted(goal.targets)), tried, finalists, confirmed)
        return tested[0][0] if tested else None

    def _tried(self, c: "_Candidate") -> TriedShot:
        return TriedShot(
            c.pot.target,
            c.pot.pocket,
            c.shot,
            c.result.cue_ball_path,
            c.table_after.positions.get(self.physics.table.cue_ball, c.result.cue_ball_path[-1]),
            c.position_value(self._openness_weight),
            c.splits,
        )

    # ---- step 2 and 3: tune the aim, try speeds and spins ----------------------------

    def _shots_that_pot(self, table: TableState, pots: list[PotOption], goal: ShotGoal) -> list[_Candidate]:
        s = self.settings
        candidates = []
        for pot in pots:
            for speed in s.speeds:
                direction = self._tune_aim(table, pot, Shot(pot.aim_direction_deg, speed))
                if direction is None:
                    continue
                for height in s.heights:
                    for side in s.sides:
                        shot = Shot(direction, speed, side=side, height=height)
                        candidate = self._try_shot(table, pot, shot, goal)
                        if candidate is not None:
                            candidates.append(candidate)
        return candidates

    def _split_shots(self, table: TableState, pots: list[PotOption], goal: ShotGoal) -> list[_Candidate]:
        """Wider search on the best pots, keeping only shots whose cue ball goes into the cluster."""
        s = self.settings
        splits = []
        for pot in pots:
            for speed in s.split_speeds:
                direction = self._tune_aim(table, pot, Shot(pot.aim_direction_deg, speed))
                if direction is None:
                    continue
                for height in s.split_heights:
                    for side in s.split_sides:
                        candidate = self._try_shot(table, pot, Shot(direction, speed, side=side, height=height), goal)
                        if candidate is not None and candidate.splits:
                            splits.append(candidate)
        return splits

    def _tied_up_balls(self, table: TableState, goal: ShotGoal) -> frozenset[str]:
        """Balls still to clear that are touching (or nearly touching) another ball."""
        if goal.balls_to_clear is None:
            return frozenset()
        game = self.physics.table
        return frozenset(
            b
            for b in goal.balls_to_clear(table)
            if b in table.positions
            and ball_is_tied_up(table.positions[b], table.positions_except(b, game.cue_ball), game)
        )

    def _tune_aim(self, table: TableState, pot: PotOption, shot: Shot) -> float | None:
        """Adjust the shot's direction until the object ball leaves straight at the pocket.

        Each attempt simulates the shot and measures how far off the object ball's
        direction is. After two attempts we know roughly how much the error changes
        per degree of aim, and jump to where the error should be zero (the secant
        method). Returns the best direction found, or None if the cue ball misses
        the target entirely.
        """
        object_ball = table.positions[pot.target]
        wanted = pot_geometry.direction_deg(object_ball, self.physics.pockets[pot.pocket])

        tried_directions: list[float] = []
        errors: list[float] = []
        direction = shot.direction_deg
        for _ in range(self.settings.aim_tuning_attempts):
            result = self.physics.simulate(table, replace(shot, direction_deg=direction))
            if result.first_ball_hit != pot.target or result.first_ball_hit_direction_deg is None:
                break
            error = pot_geometry.angle_between_deg(wanted, result.first_ball_hit_direction_deg)
            tried_directions.append(direction)
            errors.append(error)
            if abs(error) < self.settings.aim_good_enough_deg:
                break

            if len(errors) < 2 or errors[-1] == errors[-2]:
                direction += 0.05  # small nudge to learn how the error responds
            else:
                error_per_degree = (errors[-1] - errors[-2]) / (tried_directions[-1] - tried_directions[-2])
                direction -= errors[-1] / error_per_degree

        if not tried_directions:
            return None
        best = min(range(len(errors)), key=lambda i: abs(errors[i]))
        return tried_directions[best]

    def _try_shot(self, table: TableState, pot: PotOption, shot: Shot, goal: ShotGoal) -> _Candidate | None:
        """Simulate the shot. If it does its job, score where the cue ball ends up."""
        result = self.physics.simulate(table, shot)
        if not goal.is_success(result, pot.pocket):
            return None
        if pot.target not in result.potted_ball_ids:
            return None  # a fluke on another ball: legal, but not the shot we are planning

        table_after = goal.respot(result.final_table, result)
        next_targets = goal.next_targets(result, table_after)
        position = pot_geometry.next_pot_ease(table_after, next_targets, self.physics.pockets, self.physics.table)
        splits = any(b in self._tied_up for b in result.cue_ball_hits[1:])
        return _Candidate(pot, shot, result, table_after, position, self._openness(table_after, goal), splits)

    def _openness(self, table_after: TableState, goal: ShotGoal) -> float:
        if goal.balls_to_clear is None:
            return 1.0
        return table_openness(table_after, goal.balls_to_clear(table_after), self.physics.pockets, self.physics.table)

    # ---- step 4 and 5: stress test the best and choose ---------------------------------

    def _pick_finalists(self, candidates: list[_Candidate]) -> list[_Candidate]:
        """The best-position shots of *every* pot option.

        Picking per pot option means a safe pot isn't dropped just because another
        pocket gives slightly better position.
        """
        w = self._openness_weight
        by_position = sorted(candidates, key=lambda c: c.position_value(w), reverse=True)
        picked: dict[tuple[str, int, bool], int] = {}
        finalists = []
        for candidate in by_position:
            key = (candidate.pot.target, candidate.pot.pocket, candidate.splits)
            limit = (
                self.settings.split_finalists_per_pot_option
                if candidate.splits
                else self.settings.stress_tests_per_pot_option
            )
            if picked.get(key, 0) < limit:
                picked[key] = picked.get(key, 0) + 1
                finalists.append(candidate)
        return finalists

    def _stress_test(
        self, table: TableState, candidate: _Candidate, goal: ShotGoal, rng: random.Random, replays: int | None = None
    ) -> PlannedShot:
        s = self.settings
        replays = s.stress_test_replays if replays is None else replays

        # Re-tune the aim for this exact speed and spin (spin changes the throw a little).
        direction = self._tune_aim(table, candidate.pot, candidate.shot)
        if direction is not None:
            retuned = self._try_shot(table, candidate.pot, replace(candidate.shot, direction_deg=direction), goal)
            candidate = retuned or candidate

        # Replay with errors, and for every replay that goes in, look one shot ahead:
        # is there a next ball, how easy is it, how open is the table? Averaging over
        # the replays is what makes a pack split judgeable at all.
        went_in = 0
        next_ease_when_in: list[float] = []
        openness_when_in: list[float] = []
        continues = 0
        for _ in range(replays):
            table_as_seen = s.uncertainty.add_camera_error(table, rng)
            shot_as_played = s.uncertainty.cue_arm.add_to(candidate.shot, rng)
            result = self.physics.simulate(table_as_seen, shot_as_played)
            if not goal.is_success(result, candidate.pot.pocket):
                continue
            went_in += 1
            table_after = goal.respot(result.final_table, result)
            next_targets = goal.next_targets(result, table_after)
            ease = pot_geometry.next_pot_ease(table_after, next_targets, self.physics.pockets, self.physics.table)
            continues += ease > 0
            next_ease_when_in.append(ease)
            openness_when_in.append(self._openness(table_after, goal))

        if replays == 0:  # no stress test: trust the one perfect simulation
            chance, carries_on = 1.0, float(candidate.position_quality > 0)
            position, openness = candidate.position_quality, candidate.openness
        elif went_in == 0:
            chance, carries_on, position, openness = 0.0, 0.0, 0.0, candidate.openness
        else:
            chance = went_in / replays
            carries_on = continues / went_in
            position = sum(next_ease_when_in) / went_in
            openness = sum(openness_when_in) / went_in

        w = self._openness_weight
        position_value = (1 - w) * position + w * openness
        score = chance * carries_on * ((1 - s.position_weight) + s.position_weight * position_value)
        return PlannedShot(
            shot=candidate.shot,
            target=candidate.pot.target,
            pocket=candidate.pot.pocket,
            chance_of_potting=chance,
            position_quality=position,
            score=score,
            cue_ball_stops_at=candidate.table_after.positions[self.physics.table.cue_ball],
            cue_ball_path=candidate.result.cue_ball_path,
            predicted_table=candidate.table_after,
            table_openness=openness,
            chance_break_continues=carries_on,
            splits_the_pack=candidate.splits,
        )

