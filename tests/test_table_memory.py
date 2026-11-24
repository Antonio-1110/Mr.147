from mr147.interfaces.vision import Detection, DetectionFrame
from mr147.snooker.balls import Colour
from mr147.snooker.table import SPOTS
from mr147.table_memory import TableMemory

RED, BLACK, WHITE, PINK = Colour.RED, Colour.BLACK, Colour.WHITE, Colour.PINK

CUE = (WHITE, 0.0, 0.0)
RED_A = (RED, 0.5, 0.3)
RED_B = (RED, 0.8, -0.4)
RED_C = (RED, -0.6, 0.2)
BLACK_ON_SPOT = (BLACK, *SPOTS[BLACK])
START = [CUE, RED_A, RED_B, RED_C, BLACK_ON_SPOT]


def frame(*balls) -> DetectionFrame:
    return DetectionFrame(tuple(Detection(colour, x, y) for colour, x, y in balls))


def memory_after_first_look() -> TableMemory:
    memory = TableMemory()
    assert memory.first_look(frame(*START)).accepted
    return memory


def test_balls_missing_at_the_start_count_as_already_potted():
    memory = memory_after_first_look()
    assert memory.colours_on_table()[RED] == 3
    assert memory.number_potted(RED) == 12
    assert memory.number_potted(PINK) == 1


def test_red_potted_as_planned():
    memory = memory_after_first_look()
    check = memory.after_shot(frame(CUE, RED_A, RED_C, BLACK_ON_SPOT), planned_pots=[RED])
    assert check.accepted
    assert check.potted[RED] == 1
    assert memory.number_potted(RED) == 13


def test_a_hidden_red_is_not_counted_as_potted():
    memory = memory_after_first_look()
    two_reds_gone = frame(CUE, RED_A, BLACK_ON_SPOT)
    check = memory.after_shot(two_reds_gone, planned_pots=[RED])
    assert not check.accepted
    assert check.view_request.look_closer  # tells the camera where to look
    assert memory.number_potted(RED) == 12  # memory unchanged


def test_a_potted_red_can_never_come_back():
    memory = memory_after_first_look()
    memory.after_shot(frame(CUE, RED_A, RED_C, BLACK_ON_SPOT), planned_pots=[RED])
    check = memory.after_shot(frame(*START), planned_pots=[])
    assert not check.accepted
    assert any("more red" in reason for reason in check.view_request.reasons)


def test_black_potted_and_respotted_counts_as_potted():
    memory = memory_after_first_look()
    cue_moved = (WHITE, 0.1, 0.1)
    check = memory.after_shot(
        frame(cue_moved, RED_A, RED_B, RED_C, BLACK_ON_SPOT), planned_pots=[BLACK], respotted_colours={BLACK}
    )
    assert check.accepted
    assert check.potted[BLACK] == 1


def test_black_potted_but_not_respotted_yet_is_rejected():
    memory = memory_after_first_look()
    check = memory.after_shot(frame(CUE, RED_A, RED_B, RED_C), planned_pots=[BLACK], respotted_colours={BLACK})
    assert not check.accepted


def test_impossible_detections_ask_for_a_better_view():
    memory = TableMemory()
    seen_twice = (RED, 0.5, 0.31)
    assert not memory.first_look(frame(*START, seen_twice)).accepted
    assert not memory.first_look(frame(RED_A, RED_B, BLACK_ON_SPOT)).accepted  # no cue ball
    assert not memory.first_look(frame(*START, (PINK, 3.0, 0.0))).accepted  # off the table
    assert not memory.first_look(frame(*START, (BLACK, 0.0, 0.5))).accepted  # two blacks


def test_trusting_the_camera_records_what_really_happened():
    memory = memory_after_first_look()
    black_went_instead = frame(CUE, RED_A, RED_B, RED_C)
    check = memory.after_shot(black_went_instead, planned_pots=[RED], trust_camera=True)
    assert check.accepted
    assert check.potted[BLACK] == 1
    assert check.potted[RED] == 0
