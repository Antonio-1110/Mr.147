#!/usr/bin/env bash
# Overnight pool run: new pictures -> check them -> train the detector and the
# close-up ball reader -> test everything on real footage.
#
#   bash scripts/overnight_pool.sh
#
# Stops at the first problem and says which step failed. Everything is logged to
# logs/overnight-pool-<time>.log. Keeps the Mac (and its display) awake while it runs.
#
# Settings (override like: EPOCHS=10 bash scripts/overnight_pool.sh):
LAYOUTS=${LAYOUTS:-1000}      # table layouts; 3 pictures each
VIEWS=${VIEWS:-3}
SEED=${SEED:-31}              # picture names start with the seed; batches use SEED, SEED+1, ...
WIDTH=${WIDTH:-1920}          # full HD pictures, so the reader's close-ups have detail
HEIGHT=${HEIGHT:-1080}
EPOCHS=${EPOCHS:-20}
IMAGE_SIZE=${IMAGE_SIZE:-960}
BATCH=${BATCH:-4}             # 8 runs out of memory on an 8 GB Mac; try 16 on a bigger computer
READER_EPOCHS=${READER_EPOCHS:-15}
DEVICE=${DEVICE:-auto}        # auto: NVIDIA graphics card, else Apple chip, else CPU
NAME=${NAME:-pool3}           # training run name (runs/$NAME)
DATA=${DATA:-data/synth-$NAME}
REAL=${REAL:-data/real-pool}
FRAMES=${FRAMES:-4}           # random frames per real video in the footage test
RUN=${RUN:-uv run --extra train}  # the extra keeps YOLO installed
CHUNK=${CHUNK:-100}           # layouts per picture-maker run (a crash only loses one batch)
SKIP_GEN=${SKIP_GEN:-}        # SKIP_GEN=1: reuse the pictures already in $DATA
SKIP_DETECTOR=${SKIP_DETECTOR:-}  # SKIP_DETECTOR=1: keep models/pool.pt, only (re)train the reader
BALL_SET=${BALL_SET:-modern}  # colours of the balls in the real footage: modern (4 pink, 5 purple) or traditional

set -euo pipefail
cd "$(dirname "$0")/.."

# Keep the Mac awake (system and display) for the whole run.
if [[ "$(uname)" == "Darwin" && -z "${OVERNIGHT_AWAKE:-}" ]]; then
    export OVERNIGHT_AWAKE=1
    exec caffeinate -dimsu bash "$0" "$@"
fi

mkdir -p logs models
LOG="logs/overnight-pool-$(date +%Y%m%d-%H%M).log"
exec > >(tee -a "$LOG") 2>&1

STEP="starting"
on_error() {
    echo; echo "FAILED during: $STEP  (see $LOG)"; echo "at $(date)"
    if [[ $STEP == 6/9* ]]; then
        echo "to continue the detector training where it stopped:"
        echo "  caffeinate -dimsu uv run --extra train yolo train resume model=runs/$NAME/weights/last.pt"
        echo "or to redo the run without remaking the pictures:  SKIP_GEN=1 bash scripts/overnight_pool.sh"
    elif [[ $STEP == 7/9* ]]; then
        echo "the new detector is saved (models/pool.pt). To redo only the reader:"
        echo "  caffeinate -dimsu uv run --extra train mr147-train-reader --data $DATA-crops --game pool --device $DEVICE"
    fi
}
trap on_error ERR
step() { STEP="$1"; echo; echo "=== $(date +%H:%M:%S)  $1"; }

step "1/9 tests (the ones that guard pool pictures, naming and the reader)"
# Only the pool vision tests: a planner or snooker test failing shouldn't cost the night.
$RUN pytest -q tests/test_data_pipeline.py tests/test_pool_ball_sets.py tests/test_pool_vision_memory.py \
    tests/test_reader.py -k "not snooker"

step "2/9 keep the current model safe"
BEFORE=models/pool-before-$NAME.pt
if [[ -f models/pool.pt && ! -f $BEFORE ]]; then
    cp models/pool.pt "$BEFORE"
    echo "saved models/pool.pt as $BEFORE"
fi
OLD_READER=models/pool-reader-before-$NAME.pt
if [[ -f models/pool-reader.pt && ! -f $OLD_READER ]]; then
    cp models/pool-reader.pt "$OLD_READER"
    [[ -f models/pool-reader.json ]] && cp models/pool-reader.json "${OLD_READER%.pt}.json"
    echo "saved models/pool-reader.pt as $OLD_READER"
fi
START_FROM=$BEFORE
[[ -f "$START_FROM" ]] || START_FROM=yolo11n.pt
echo "training will start from $START_FROM"

step "3/9 make $((LAYOUTS * VIEWS)) pictures (${WIDTH}x${HEIGHT}) into $DATA"
if [[ -n "$SKIP_GEN" ]]; then
    echo "SKIP_GEN set: using the pictures already in $DATA"
else
if [[ -d "$DATA" ]]; then  # never mix with a half-finished earlier set
    OLD="$DATA-old-$(date +%Y%m%d-%H%M)"
    mv "$DATA" "$OLD"
    echo "moved the existing $DATA to $OLD"
fi
# Made in batches of $CHUNK layouts, each in a fresh process with its own seed (the
# picture names start with the seed, so batches never overwrite each other). If the
# picture maker dies, that batch is removed and redone with the next seed.
made=0; seed=$SEED; restarts=0
while (( made < LAYOUTS )); do
    n=$(( LAYOUTS - made < CHUNK ? LAYOUTS - made : CHUNK ))
    echo "batch: layouts $((made + 1))-$((made + n)) of $LAYOUTS, seed $seed"
    if $RUN mr147-gen --game pool --out "$DATA" --layouts "$n" --views "$VIEWS" --seed "$seed" \
            --width "$WIDTH" --height "$HEIGHT"; then
        made=$((made + n))
    else
        code=$?
        restarts=$((restarts + 1))
        echo "picture maker stopped (exit code $code) in the seed $seed batch: removing it and redoing it ($restarts restarts so far)"
        find "$DATA" -name "$(printf %02d "$seed")??????.*" -delete
        if (( restarts > 8 )); then echo "too many restarts, giving up"; false; fi
    fi
    seed=$((seed + 1))
    (( seed <= 99 )) || { echo "ran out of seeds (picture names need 2-digit seeds)"; false; }
done
echo "made $(find "$DATA/images" -name '*.jpg' | wc -l | tr -d ' ') pictures"
fi

step "4/9 check the pictures and labels"
$RUN mr147-check-data "$DATA" --game pool
$RUN mr147-preview "$DATA" --out "$DATA/preview.jpg"

step "5/9 cut ball close-ups for the reader"
$RUN mr147-crops "$DATA"

step "6/9 train the detector ($EPOCHS passes, $IMAGE_SIZE px, batch $BATCH, $DEVICE)"
if [[ -n "$SKIP_DETECTOR" ]]; then
    echo "SKIP_DETECTOR set: keeping models/pool.pt"
else
$RUN mr147-train --game pool --data "$DATA" --start-from "$START_FROM" --epochs "$EPOCHS" \
    --image-size "$IMAGE_SIZE" --batch "$BATCH" --device "$DEVICE" --name "$NAME" --skip-check
# (the check already ran in step 4)
fi

step "7/9 train the close-up reader ($READER_EPOCHS passes, $DEVICE)"
$RUN mr147-train-reader --data "$DATA-crops" --game pool --epochs "$READER_EPOCHS" --device "$DEVICE"

step "8/9 test on the real footage: old detector, new detector, new detector + reader"
trap - ERR; set +e  # the new models are already saved; a problem here shouldn't hide that
SOURCES=$(ls "$REAL"/*.MOV "$REAL"/*.mov "$REAL"/*.JPG "$REAL"/*.jpg 2>/dev/null | grep -v -- '-check' || true)
check() {  # check <tag> <args...>
    local tag=$1; shift
    # shellcheck disable=SC2086
    $RUN mr147-check-video $SOURCES --game pool --frames "$FRAMES" "$@" \
        --out "$REAL/pool-check-$tag.jpg" | tee "logs/check-$tag.txt"
}
if [[ -z "$SOURCES" ]]; then
    echo "no footage in $REAL, skipping"
else
    cp models/pool.pt "models/pool-$NAME.pt"
    if [[ -f $BEFORE ]]; then
        check before-raw --model "$BEFORE" --raw
        check before --model "$BEFORE"
    fi
    check new-raw --model models/pool.pt --raw
    check new --model models/pool.pt
    check reader-raw --model models/pool.pt --reader models/pool-reader.pt --ball-set "$BALL_SET" --raw
    check reader --model models/pool.pt --reader models/pool-reader.pt --ball-set "$BALL_SET"
fi
set -e; trap on_error ERR

step "9/9 summary"
echo "frames with an impossible count (e.g. two 8 balls), out of all test frames:"
for tag in before new reader; do
    [[ -f logs/check-$tag-raw.txt ]] || continue
    total=$(grep -cE '\.(MOV|mov|JPG|jpg) ' "logs/check-$tag-raw.txt" || true)
    raw=$(grep -c "impossible" "logs/check-$tag-raw.txt" || true)
    named=$(grep -c "impossible" "logs/check-$tag.txt" || true)
    echo "  $tag: model alone $raw, after one-of-each naming $named (of $total)"
done
if [[ -f "$REAL/answer-key.json" ]]; then
    echo
    echo "answer key ($REAL/answer-key.json: real balls with their true numbers):"
    trap - ERR; set +e  # a scoring problem shouldn't hide the rest of the summary
    [[ -f $BEFORE ]] && $RUN python -m mr147.vision.score_key --model "$BEFORE" --key "$REAL/answer-key.json" --label "  old detector"
    $RUN python -m mr147.vision.score_key --model models/pool.pt --key "$REAL/answer-key.json" --label "  new detector"
    [[ -f $OLD_READER ]] && $RUN python -m mr147.vision.score_key --model models/pool.pt --reader "$OLD_READER" \
        --ball-set "$BALL_SET" --key "$REAL/answer-key.json" --label "  new detector + old reader"
    $RUN python -m mr147.vision.score_key --model models/pool.pt --reader models/pool-reader.pt --ball-set "$BALL_SET" \
        --key "$REAL/answer-key.json" --label "  new detector + reader"
    set -e; trap on_error ERR
fi
echo
echo "detector training scores: runs/$NAME/results.csv"
tail -1 "runs/$NAME/results.csv" 2>/dev/null | awk -F, '{printf "  final mAP50 %.3f  mAP50-95 %.3f\n", $8, $9}' || true
echo "reader training scores: runs/pool-reader/results.csv"
tail -1 runs/pool-reader/results.csv 2>/dev/null | awk -F, '{printf "  final top-1 accuracy %.3f\n", $4}' || true
echo "models: models/pool.pt (new detector, also models/pool-$NAME.pt), models/pool-reader.pt, old: $BEFORE, $OLD_READER"
echo "contact sheets: $REAL/pool-check-before*.jpg, -new*.jpg, -reader*.jpg"
echo
echo "DONE at $(date)"
