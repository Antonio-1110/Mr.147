# Mr. 147

A robot that plays snooker. Simulation first, real robot later.

## Setting up on a new computer

Needs [uv](https://docs.astral.sh/uv/) (Mac: `brew install uv`) and git.

    git clone https://github.com/Antonio-1110/Mr.147.git
    cd Mr.147
    uv sync --extra train     # Python, the simulator and YOLO
    uv run pytest -q          # every test should pass

GitHub only has the code. Your own photos, clips, answer key and trained models
are too big for it: copy `data/real-pool/` and `models/` from the old computer
(the zip `mr147-files-for-new-computer.zip` holds both) and unzip it inside the
`Mr.147` folder. Training pictures (`data/synth-*`) are made again on the new
computer by `scripts/overnight_pool.sh`.

Training picks the fastest chip by itself: an NVIDIA graphics card, else the
Apple chip, else the processor. With more memory than an 8 GB Mac, add
`BATCH=16` to the overnight command so the detector trains faster.

**Windows with an NVIDIA card.** Install uv with
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`,
then `uv sync --extra train` as above. On Windows it takes PyTorch's CUDA
build, so training runs on the graphics card. Check it with
`uv run python -c "import torch; print(torch.cuda.is_available())"` (should say True).
The overnight run is `scripts\overnight_pool.ps1` (batch sizes set for a 12 GB RTX 3060):

    powershell -ExecutionPolicy Bypass -File scripts\overnight_pool.ps1
    powershell -ExecutionPolicy Bypass -File scripts\overnight_pool.ps1 -SkipGen -SkipDetector   # only the reader

## Ball vision: training pictures from the simulator

The robot sees the table like a player does: cameras at head height, walking
around the table. To teach it to find the balls without labelling photos by
hand, we render a 3D table from thousands of head-height spots and let the
program write the labels itself (it knows where every ball is).

Make pictures (on your Mac), for snooker or pool:

    uv run mr147-gen --game snooker --layouts 500 --views 4
    uv run mr147-gen --game pool --layouts 500 --views 4

Check that the boxes sit on the balls and pockets:

    uv run mr147-preview data/synth-pool --n 12

What comes out, in `data/synth-snooker/` or `data/synth-pool/`:

| Folder | What's in it |
|---|---|
| `images/train`, `images/val` | the pictures |
| `labels/train`, `labels/val` | one line per ball: colour number and box (YOLO format) |
| `meta/` | every ball's real 3D position, how much of it is visible, and the camera's position |
| `data.yaml` | the file the YOLO trainer reads |

Labels:

| Game | Label numbers |
|---|---|
| Snooker (green cloth) | 0 red, 1 yellow, 2 green, 3 brown, 4 blue, 5 pink, 6 black, 7 white, 8 pocket |
| Pool (blue cloth, 9-foot table) | 0 cue ball, 1-15 the numbered balls, 16 pocket |

Pool balls are painted as real ones: 1-8 solid, 9-15 striped, with the number in
a white circle, and each ball is turned to a random angle.

A ball or pocket gets a label when at least 30% of it is visible. Each layout gets random
lighting, cloth shade (always green for snooker, blue for pool), wood and room colours; each picture gets random blur,
noise, exposure and JPEG quality.

Code lives in `src/mr147/vision/`. Run the checks with `uv run pytest`.

**The reader sees only the ball.** Before the close-up reader looks at a ball,
everything outside the ball's circle is painted grey. A reader that also saw the
cloth learnt that a blue ring round a black ball means "the blue 10, seen from
its black cap", so on blue cloth the black 8 came out as a 10. Readers trained
before this change still get the cloth (they have no `pool-reader.json`).

## Training the detector

    uv sync --extra train                                    # once: YOLO + PyTorch
    uv run mr147-gen --game snooker --layouts 1500 --views 4  # about 6,000 pictures
    uv run mr147-train --game snooker                         # uses the Mac's graphics chip

The best model is copied to `models/snooker.pt`; scores and charts are in
`runs/snooker/`. Train one model per game (`--game pool`). The trainer starts
from a small ready-made YOLO model and only learns our labels.

Note: `ultralytics` (the YOLO package) is free for personal and open-source
use, but selling the robot or its software needs their paid licence.

## Checking the detector on a real video

    uv run mr147-check-video match.mp4 --model models/snooker.pt --frames 12

Picks 12 random frames (one from each twelfth of the video), draws the model's
boxes, and writes `video-check.jpg`. Frames where the count is impossible (16
reds, two blacks) are marked in red. Add `--save-for-fixing data/real-snooker`
to also save those frames with the model's boxes, ready to correct in a
box-editing app (X-AnyLabeling). The corrected frames are both a real score
and real training data:

    uv run mr147-train --game snooker --data data/real-snooker --start-from models/snooker.pt --epochs 20

## From boxes to table positions

`vision/camera_vision.py` is the robot's vision. It speaks the same contract as
the simulated camera (`interfaces/vision.py`), so the brain can't tell them apart.

1. The robot's body says roughly where the camera is (a few cm).
2. The pockets in the picture pin the camera down exactly (`vision/locate.py`).
3. Each ball box becomes a table position: where a ball must sit to fill that box.
4. If the table memory isn't sure, it asks for a better view. Vision picks
   where to stand (close, a clear line of sight, pockets in view), looks again,
   uses the balls it already placed to pin the new camera position, and merges
   the two looks.

Snooker and pool both work (`CameraVision(..., spec=POOL)` and
`TableMemory(balls=POOL_BALLS)`). Pool balls are named "1" to "15" and "white".

**One of each.** The detector scores every name a ball might have ("8: 0.6, 4: 0.3").
Vision then names all balls together so no name is used more often than the game
allows (one 8 ball, at most 15 reds): the surest 8 keeps its name, the others get
their next-best one (`vision/identify.py`). A ball left with a name the detector
barely considered gets a low confidence, so the table memory asks for a closer
look. `mr147-check-video` shows this too ("4 (was 8)"); add `--raw` to see the
model's own picks.

In simulation with near-perfect boxes this places balls to about 4-5 mm. Almost
all of that comes from the camera position, not the balls: a sharper, wider
camera and more known marks (cushion lines, spots) are the way to get to 1 mm.

## Seeing it work

    uv run python -m mr147.play_simulated_break --seed 0 --camera rendered --pictures outputs/break0

`--camera rendered` makes the robot look at the simulated table through a
rendered head camera and the real vision code (with a stand-in detector that
reads the simulator's boxes). `--camera models/snooker.pt` uses your trained
model instead. `--pictures` saves a picture of every planned shot: from above,
and from behind the cue ball, with the pot line, the cue ball's path and where
it should stop.

## Choosing shots: the robot's brain

Vision says where the balls are. The brain keeps its own copy of the table,
works out which ball is on, and picks the shot that pots it **and** leaves the
cue ball in a good place for the next one.

    src/mr147/
      game_table.py         GameTable: table size, ball size, pocket openings (any game)
      snooker/              everything snooker
        table.py, balls.py    table numbers, ball colours and points
        rules.py              the 147 sequence: which ball is on, what's next, respots
        shot_goal.py          what counts as a good snooker shot
        robot.py              SnookerRobot: look -> plan -> shoot -> check
        simulated_table.py    fake camera + cue arm for testing on a laptop
        layouts.py            random tables, full rack, after a break-off
        play.py               watch it play a break
      pool/                 everything pool (8-ball and 9-ball)
        table.py, balls.py    9-foot table numbers, balls "1".."15" and "white"
        eight_ball.py, nine_ball.py   the rules (replayed on every simulated shot)
        shot_goals.py         what counts as a good pool shot
        break_shot.py         picks the break: tries cue spots, aims and speeds on messy racks
        ball_in_hand.py       where to put the cue ball when the robot places it
        robot.py              PoolRobot: break, then look -> plan -> shoot -> check
        simulated_table.py    fake camera + cue arm for pool
        layouts.py            9-ball and 8-ball racks, random tables
        play.py               watch it clear a rack
      interfaces/           contracts with the other teams
        vision.py             Detection, DetectionFrame, ViewRequest, VisionSystem
        cue_arm.py            Shot, CueArm
      table_memory/         our copy of the table; checks each frame, remembers pots
      physics/              the only code that uses pooltool (coordinate conversion here)
      shot_planner/         any game: geometry -> tune aim -> try speeds/spins -> stress test
                            (plan_viewer.py: the --show-plans web page)
      rules/, robot.py, simulated_hardware/, play_simulated_break.py
                            old paths; they still work and point at snooker/

There is no referee in the simulation: the robot is meant to clear the rack in
one visit, so a foul or a miss just ends the run.

**Coordinates** (vision too): metres, (0, 0) is the table centre, x along the
length (baulk end negative), y across. **Ball names** are colours: red,
yellow, green, brown, blue, pink, black, white (the cue ball).

**What vision sends:** a `DetectionFrame` with one `Detection(colour, x, y,
confidence)` per ball, once the balls have stopped. If it doesn't add up (a red
vanished that wasn't meant to be potted, two blacks, a ball off the table, low
confidence), the robot calls `request_better_view(ViewRequest)` with the
reasons and where to look, and vision sends a new frame.

**Potted balls are remembered**, so a hidden red is never counted as potted
and a red can never come back.

Watch it play:

    uv run python -m mr147.play_simulated_break --seed 0 --perfect-hardware
    uv run python -m mr147.play_simulated_break --seed 0      # 0.08 deg aim, 3% speed, 1 mm camera error
    uv run python -m mr147.play_simulated_break --help

Pool:

    uv run python -m mr147.pool.play --game nine --seed 0 --racks 5
    uv run python -m mr147.pool.play --game eight --seed 0 --perfect-hardware

**See what it was thinking.** Add `--show-plans outputs/plans.html` to either
command and open the page in a browser. For every shot it shows the table from
above, every shot that potted the ball (faint), the finalists with their odds
(click a row), the chosen cue-ball path and where it should stop, and in red
what really happened with the arm's error:

    uv run python -m mr147.pool.play --game nine --seed 0 --show-plans outputs/pool-plans.html
    uv run python -m mr147.snooker.play --seed 3 --show-plans outputs/snooker-plans.html

**Precision is the hard part.** Over a metre, 0.1 degrees of aim error or 1-2 mm
of camera error turns a pot into a miss. Tell the planner how precise the real
hardware is (`Uncertainty`) so it picks shots that still go in.
