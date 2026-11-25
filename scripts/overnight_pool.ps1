# Overnight pool run for Windows (same steps as overnight_pool.sh on the Mac):
# new pictures -> check them -> train the detector and the close-up ball reader
# -> test everything on real footage.
#
#   powershell -ExecutionPolicy Bypass -File scripts\overnight_pool.ps1
#
# Settings go after the file name, for example:
#   ... -File scripts\overnight_pool.ps1 -SkipGen -SkipDetector     (only retrain the reader)
#   ... -File scripts\overnight_pool.ps1 -Epochs 10 -Batch 8
#
# Stops at the first problem and says which step failed. Everything is logged to
# logs\overnight-pool-<time>.log. Keeps the PC awake (and the screen on) while it runs.
param(
    [int]$Layouts = 1000,        # table layouts; 3 pictures each
    [int]$Views = 3,
    [int]$Seed = 31,             # picture names start with the seed; batches use Seed, Seed+1, ...
    [int]$Width = 1920,          # full HD pictures, so the reader's close-ups have detail
    [int]$Height = 1080,
    [int]$Epochs = 20,
    [int]$ImageSize = 960,
    [int]$Batch = 16,            # fits the 12 GB RTX 3060 at 960 px; use 8 if it runs out of memory
    [int]$ReaderEpochs = 15,
    [int]$ReaderBatch = 128,     # the reader's close-ups are small, 128 fits easily in 12 GB
    [string]$Device = "auto",    # auto: NVIDIA graphics card, else CPU
    [string]$Name = "pool3",     # training run name (runs\<Name>)
    [string]$Data = "",          # default: data\synth-<Name>
    [string]$Real = "data\real-pool",
    [int]$Frames = 4,            # random frames per real video in the footage test
    [int]$Chunk = 100,           # layouts per picture-maker run (a crash only loses one batch)
    [switch]$SkipGen,            # reuse the pictures already in <Data>
    [switch]$SkipDetector,       # keep models\pool.pt, only (re)train the reader
    [string]$BallSet = "modern"  # colours of the balls in the real footage: modern or traditional
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not $Data) { $Data = "data\synth-$Name" }
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"   # where the uv installer puts uv
$env:PYTHONIOENCODING = "utf-8"

# Keep the PC awake (system and display) until this script ends.
Add-Type -Namespace Mr147 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
[void][Mr147.Power]::SetThreadExecutionState([uint32]2147483651)  # CONTINUOUS | SYSTEM | DISPLAY

New-Item -ItemType Directory -Force logs, models | Out-Null
$Log = "logs\overnight-pool-$(Get-Date -Format yyyyMMdd-HHmm).log"
Start-Transcript -Path $Log -Append | Out-Null

$script:Step = "starting"
function Step($text) {
    $script:Step = $text
    Write-Host ""
    Write-Host "=== $(Get-Date -Format HH:mm:ss)  $text"
}
# Runs a project command with uv (the train extra keeps YOLO installed).
# Stops the night if it fails, unless -NoStop.
# "| Out-Host" so the log (Start-Transcript) keeps the output; Windows PowerShell 5.1 drops it otherwise.
function Run {
    & uv run --extra train @args | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "'$($args -join ' ')' failed (exit code $LASTEXITCODE)" }
}
function RunNoStop { & uv run --extra train @args | Out-Host }

$Before = "models\pool-before-$Name.pt"
$OldReader = "models\pool-reader-before-$Name.pt"
try {
    Step "1/9 tests (the ones that guard pool pictures, naming and the reader)"
    # Only the pool vision tests: a planner or snooker test failing shouldn't cost the night.
    Run pytest -q tests/test_data_pipeline.py tests/test_pool_ball_sets.py tests/test_pool_vision_memory.py `
        tests/test_reader.py -k "not snooker"

    Step "2/9 keep the current model safe"
    if ((Test-Path models\pool.pt) -and -not (Test-Path $Before)) {
        Copy-Item models\pool.pt $Before
        Write-Host "saved models\pool.pt as $Before"
    }
    if ((Test-Path models\pool-reader.pt) -and -not (Test-Path $OldReader)) {
        Copy-Item models\pool-reader.pt $OldReader
        if (Test-Path models\pool-reader.json) { Copy-Item models\pool-reader.json ($OldReader -replace '\.pt$', '.json') }
        Write-Host "saved models\pool-reader.pt as $OldReader"
    }
    $StartFrom = if (Test-Path $Before) { $Before } else { "yolo11n.pt" }
    Write-Host "training will start from $StartFrom"

    Step "3/9 make $($Layouts * $Views) pictures (${Width}x${Height}) into $Data"
    if ($SkipGen) {
        Write-Host "-SkipGen: using the pictures already in $Data"
    } else {
        if (Test-Path $Data) {  # never mix with a half-finished earlier set
            $Old = "$Data-old-$(Get-Date -Format yyyyMMdd-HHmm)"
            Move-Item $Data $Old
            Write-Host "moved the existing $Data to $Old"
        }
        # Made in batches of $Chunk layouts, each in a fresh process with its own seed (the
        # picture names start with the seed, so batches never overwrite each other). If the
        # picture maker dies, that batch is removed and redone with the next seed.
        $made = 0; $s = $Seed; $restarts = 0
        while ($made -lt $Layouts) {
            $n = [Math]::Min($Chunk, $Layouts - $made)
            Write-Host "batch: layouts $($made + 1)-$($made + $n) of $Layouts, seed $s"
            RunNoStop mr147-gen --game pool --out $Data --layouts $n --views $Views --seed $s `
                --width $Width --height $Height
            if ($LASTEXITCODE -eq 0) {
                $made += $n
            } else {
                $restarts++
                Write-Host "picture maker stopped (exit code $LASTEXITCODE) in the seed $s batch: removing it and redoing it ($restarts restarts so far)"
                Get-ChildItem $Data -Recurse -File -Filter ("{0:D2}??????.*" -f $s) -ErrorAction SilentlyContinue | Remove-Item -Force
                if ($restarts -gt 8) { throw "too many restarts, giving up" }
            }
            $s++
            if ($s -gt 99) { throw "ran out of seeds (picture names need 2-digit seeds)" }
        }
        Write-Host "made $((Get-ChildItem "$Data\images" -Recurse -Filter *.jpg).Count) pictures"
    }

    Step "4/9 check the pictures and labels"
    Run mr147-check-data $Data --game pool
    Run mr147-preview $Data --out "$Data\preview.jpg"

    Step "5/9 cut ball close-ups for the reader"
    Run mr147-crops $Data

    Step "6/9 train the detector ($Epochs passes, $ImageSize px, batch $Batch, $Device)"
    if ($SkipDetector) {
        Write-Host "-SkipDetector: keeping models\pool.pt"
    } else {
        Run mr147-train --game pool --data $Data --start-from $StartFrom --epochs $Epochs `
            --image-size $ImageSize --batch $Batch --device $Device --name $Name --skip-check
        # (the check already ran in step 4)
    }

    Step "7/9 train the close-up reader ($ReaderEpochs passes, batch $ReaderBatch, $Device)"
    Run mr147-train-reader --data "$Data-crops" --game pool --epochs $ReaderEpochs --batch $ReaderBatch --device $Device
} catch {
    Write-Host ""
    Write-Host "FAILED during: $script:Step  (see $Log)"
    Write-Host "  $($_.Exception.Message)"
    Write-Host "at $(Get-Date)"
    if ($script:Step -like "6/9*") {
        Write-Host "to continue the detector training where it stopped:"
        Write-Host "  uv run --extra train yolo train resume model=runs/$Name/weights/last.pt"
        Write-Host "or to redo the run without remaking the pictures:  add -SkipGen"
    } elseif ($script:Step -like "7/9*") {
        Write-Host "the new detector is saved (models\pool.pt). To redo only the reader: add -SkipGen -SkipDetector"
    }
    Stop-Transcript | Out-Null
    [void][Mr147.Power]::SetThreadExecutionState([uint32]2147483648)
    exit 1
}

# From here on the new models are saved; a problem shouldn't hide that.
$ErrorActionPreference = "Continue"

Step "8/9 test on the real footage: old detector, new detector, new detector + reader"
$Sources = @(Get-ChildItem $Real -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -match '^\.(mov|jpg)$' -and $_.Name -notmatch '-check' } |
    ForEach-Object { $_.FullName })
function Check($tag) {
    $extra = $args
    & uv run --extra train mr147-check-video @Sources --game pool --frames $Frames @extra --out "$Real\pool-check-$tag.jpg" |
        Tee-Object -FilePath "logs\check-$tag.txt"
}
if ($Sources.Count -eq 0) {
    Write-Host "no footage in $Real, skipping"
} else {
    Copy-Item models\pool.pt "models\pool-$Name.pt"
    if (Test-Path $Before) {
        Check before-raw --model $Before --raw
        Check before --model $Before
    }
    Check new-raw --model models\pool.pt --raw
    Check new --model models\pool.pt
    Check reader-raw --model models\pool.pt --reader models\pool-reader.pt --ball-set $BallSet --raw
    Check reader --model models\pool.pt --reader models\pool-reader.pt --ball-set $BallSet
}

Step "9/9 summary"
Write-Host "frames with an impossible count (e.g. two 8 balls), out of all test frames:"
foreach ($tag in "before", "new", "reader") {
    if (-not (Test-Path "logs\check-$tag-raw.txt")) { continue }
    $total = @(Select-String -Path "logs\check-$tag-raw.txt" -Pattern '\.(MOV|mov|JPG|jpg) ' -CaseSensitive).Count
    $raw = @(Select-String -Path "logs\check-$tag-raw.txt" -Pattern 'impossible').Count
    $named = @(Select-String -Path "logs\check-$tag.txt" -Pattern 'impossible').Count
    Write-Host "  ${tag}: model alone $raw, after one-of-each naming $named (of $total)"
}
$Key = "$Real\answer-key.json"
if (Test-Path $Key) {
    Write-Host ""
    Write-Host "answer key (${Key}: real balls with their true numbers):"
    if (Test-Path $Before) { RunNoStop python -m mr147.vision.score_key --model $Before --key $Key --label "  old detector" }
    RunNoStop python -m mr147.vision.score_key --model models\pool.pt --key $Key --label "  new detector"
    if (Test-Path $OldReader) {
        RunNoStop python -m mr147.vision.score_key --model models\pool.pt --reader $OldReader `
            --ball-set $BallSet --key $Key --label "  new detector + old reader"
    }
    RunNoStop python -m mr147.vision.score_key --model models\pool.pt --reader models\pool-reader.pt --ball-set $BallSet `
        --key $Key --label "  new detector + reader"
}
Write-Host ""
Write-Host "detector training scores: runs\$Name\results.csv"
if (Test-Path "runs\$Name\results.csv") {
    $c = (Get-Content "runs\$Name\results.csv" -Tail 1).Split(",")
    Write-Host ("  final mAP50 {0:N3}  mAP50-95 {1:N3}" -f [double]$c[7], [double]$c[8])
}
Write-Host "reader training scores: runs\pool-reader\results.csv"
if (Test-Path "runs\pool-reader\results.csv") {
    $c = (Get-Content "runs\pool-reader\results.csv" -Tail 1).Split(",")
    Write-Host ("  final top-1 accuracy {0:N3}" -f [double]$c[3])
}
Write-Host "models: models\pool.pt (new detector, also models\pool-$Name.pt), models\pool-reader.pt, old: $Before, $OldReader"
Write-Host "contact sheets: $Real\pool-check-before*.jpg, -new*.jpg, -reader*.jpg"
Write-Host ""
Write-Host "DONE at $(Get-Date)"
Stop-Transcript | Out-Null
[void][Mr147.Power]::SetThreadExecutionState([uint32]2147483648)
