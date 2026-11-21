# Filter Portal

[![tests](https://github.com/aarseniii/filter-portal/actions/workflows/tests.yml/badge.svg)](https://github.com/aarseniii/filter-portal/actions/workflows/tests.yml)

Hold up both hands and the shape between your thumbs and index fingers
becomes a "portal" showing your webcam through a filter. Everything outside
the portal stays normal.

The index fingertips make the top edge and the thumbs the bottom edge. Twist
one hand until its thumb is above its index finger and the lines cross into
an X, with the filter in both halves.

## Setup

Needs Python 3.10 or newer and a webcam. Works on macOS and Windows (and
Linux); the automatic tests run on all three.

Get the code: `git clone https://github.com/aarseniii/filter-portal.git`, or
on GitHub click **Code > Download ZIP** and unzip it.

### Windows

1. Install Python from [python.org](https://www.python.org/downloads/). In
   the installer, tick **"Add python.exe to PATH"**.
2. In the `filter-portal` folder, double-click **`Filter Portal.bat`**.

The first time, it sets up everything it needs (takes a few minutes, needs
internet). After that, double-clicking it starts Filter Portal straight
away. Tip: right-click it > **Show more options > Send to > Desktop (create
shortcut)** to start it from your desktop.

If the camera won't open, check **Settings > Privacy & security > Camera**:
"Camera access" and "Let desktop apps access your camera" must be on.

### Mac / Linux

```bash
cd filter-portal
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On a Mac you can also turn it into a normal app with a Dock icon (needs
Xcode or its command line tools):

```bash
.venv/bin/python build_app.py
```

## Run

**Windows:** double-click **`Filter Portal.bat`**.

**Mac (after `build_app.py`):** click **Filter Portal** in the Dock (or
press Cmd+Space, type "Filter Portal", press Enter).

- The very first time, macOS asks to let Filter Portal use the camera:
  click **Allow**. If you clicked "Don't Allow" by mistake, the app shows a
  button that opens the right Settings page; turn Filter Portal on there.
- While it's open, its window shows up in the Dock with a Python icon.
- If it's already open, clicking it again just tells you so.
- Problems show in a message box; details go to
  `~/Library/Logs/Filter Portal.log`.

**From a terminal** (in this folder):

```bash
.venv/bin/python filter_portal.py        # Mac / Linux
.venv\Scripts\python filter_portal.py    # Windows
```

The first run downloads MediaPipe's hand model (`hand_landmarker.task`,
~8 MB). On a Mac, when run from a terminal, it's the terminal app that needs
camera access (System Settings > Privacy & Security > Camera).

### Rebuilding the Mac app (Mac only)

```bash
.venv/bin/python build_app.py
```

This (re)makes `~/Applications/Filter Portal.app` and adds it to the Dock
if it isn't there yet (`--no-dock` to skip). The app runs Filter Portal from
this folder, so if you move or rename the folder, run this again. It needs
Xcode or its command line tools, because the app's small launcher is
compiled Swift: it asks macOS for camera access the proper way, then sets
up `.venv` if it's missing and starts `filter_portal.py`.

## Controls

| Input | Action |
|---|---|
| Both hands up, thumbs and index fingers spread | Open the portal |
| `1` to `6` | Purple tint, negative, grayscale, blur, edges, pixelate |
| Thumb tip on pinky tip (either hand, any pose) | Next filter |
| `D` | Show how sure the model is that each hand is touching |
| `Q` or `Esc` | Quit |

Settings such as smoothing, filter strength and gesture sensitivity are at
the top of `filter_portal.py`.

## Pinky-thumb gesture model

`pinky_thumb_model.npz` is a small neural network trained on hand landmarks
from [HaGRIDv2](https://github.com/hukenovs/hagrid), about a million hands
from thousands of people. HaGRID only shows the touch with the other three
fingers up, so training adds made-up touches built from real hands in every
other pose, including the L-shaped portal hand.

Results on HaGRID's held-out test set (people not seen in training):

- Real touch photos: 99.7% recognised.
- Made-up touches from the portal L-shape: 99.8% recognised.
- Other, real hand poses: 0.06% wrongly seen as a touch.

To retrain:

```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python train_gesture_model.py
```

This downloads HaGRIDv2's annotations (~720 MB) into `data/` the first time.

### Make it work better for you

If the switch still misses your touches, or fires when it shouldn't, record
about a minute of your own examples. Only hand landmark positions are saved,
never images:

```bash
.venv/bin/python record_my_gesture.py
.venv/bin/python train_gesture_model.py
```

In the recorder, press `T` to record touches, `N` to record anything else
(the portal pose especially), and `S` to save. Training then reports how
well the model does on the newest part of your recordings, which it didn't
train on.

On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python` in the
commands in this README.

## Tests

The tests don't use the webcam or open any windows:

```bash
.venv/bin/python -m unittest discover tests -v
bash tests/test_app_launcher.sh
```

The second one (Mac only) checks the Mac app's launcher with fake projects,
using a throwaway copy of the app (your real app and Dock aren't touched).

GitHub runs these automatically on Windows, macOS and Linux after every
push (see the badge at the top and the **Actions** tab).

## Files

| File | What it is |
|---|---|
| `filter_portal.py` | The app |
| `train_gesture_model.py` | Trains the pinky-thumb model |
| `record_my_gesture.py` | Records your own gesture examples (landmarks only) |
| `Filter Portal.bat` | Windows: double-click to set up (first time) and start |
| `build_app.py` | Makes the Mac app in `~/Applications` |
| `pinky_thumb_model.npz` | The trained model (plain numpy arrays) |
| `tests/` | Headless tests |
| `.github/workflows/tests.yml` | Runs the tests on Windows, macOS and Linux |

## License and credits

- The code is under the [MIT License](LICENSE).
- `pinky_thumb_model.npz` and the sample hands in `tests/` are derived from
  [HaGRIDv2](https://github.com/hukenovs/hagrid) by the HaGRID authors, and
  stay under HaGRID's license (a variant of CC BY-SA 4.0).
- Hand tracking: [Google MediaPipe](https://ai.google.dev/edge/mediapipe).
  Its hand model is downloaded on first run and isn't included here.
