"""Never confidently wrong: follow a ball through nearby frames, average, and leave doubtful balls unnamed."""

from mr147.vision.detector import PixelDetection
from mr147.vision.vote import UNNAMED, averaged, follow, name_balls, with_bar


def ball(x, y, *scores, w=40):
    return PixelDetection(scores[0][0], (x, y, x + w, y + w), scores[0][1], tuple(scores))


def test_follow_keeps_each_ball_with_its_nearest_match_and_drops_lost_ones():
    frames = [
        [ball(100, 100, ("8", 0.9)), ball(300, 100, ("10", 0.9))],
        [ball(110, 102, ("8", 0.8)), ball(305, 100, ("10", 0.7))],  # centre
        [ball(118, 104, ("10", 0.6)), ball(900, 500, ("2", 0.9))],  # the 10 is lost (jumped too far)
    ]
    tracks = follow(frames, 1)
    assert [len(t) for t in tracks] == [3, 2]
    assert sorted(d.box[0] for d in tracks[0]) == [100, 110, 118]


def test_averaging_over_frames_outvotes_one_bad_look():
    track = [ball(0, 0, ("8", 0.9), ("10", 0.1)), ball(0, 0, ("10", 0.85), ("8", 0.15)),
             ball(0, 0, ("8", 0.95), ("10", 0.05))]
    s = averaged(track)
    assert max(s, key=s.get) == "8" and abs(s["8"] - 2.0 / 3) < 1e-9


def test_below_the_bar_a_ball_is_left_unnamed_not_guessed():
    named = [ball(0, 0, ("8", 0.97)), ball(50, 0, ("10", 0.6)), PixelDetection("pocket", (0, 0, 9, 9), 0.3)]
    out = with_bar(named, 0.9)
    assert [d.name for d in out] == ["8", UNNAMED, "pocket"]


def test_name_balls_with_one_frame_is_the_old_one_of_each_naming():
    frames = [[ball(100, 100, ("8", 0.9), ("10", 0.1)), ball(300, 100, ("8", 0.6), ("10", 0.4))]]
    out = name_balls(frames, 0)
    assert sorted(d.name for d in out) == ["10", "8"]


def test_a_ball_read_wrongly_in_one_frame_is_named_right_after_the_vote():
    frames = [[ball(100 + 3 * k, 100, ("8", 0.9), ("10", 0.1))] for k in range(5)]
    frames[2] = [ball(106, 100, ("10", 0.9), ("8", 0.1))]  # the centre frame alone says 10
    assert name_balls(frames[2:3], 0)[0].name == "10"
    assert name_balls(frames, 2)[0].name == "8"
