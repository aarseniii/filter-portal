r"""
Filter Portal
=============

A webcam toy: hold up both hands and the area between your thumbs and index
fingers becomes a "portal" that shows the video through a filter.

    1 = purple tint     2 = negative     3 = grayscale
    4 = blur            5 = edges        6 = pixelate
    Touch your thumb to your pinky (either hand) = next filter
    Q (or Esc) = quit

The pinky-thumb gesture uses a small model trained by train_gesture_model.py
(saved as pinky_thumb_model.npz). Without that file, keys still work.

How it works, per frame:
    1. Grab a webcam frame and mirror it.
    2. Ask MediaPipe for up to 2 hands.
    3. Take 4 points: thumb tip + index tip of each hand.
    4. Smooth those points over time so the shape doesn't jitter.
    5. Connect them in a fixed order: index fingertips = top edge, thumbs =
       bottom edge, each hand's index-to-thumb = a side. Twist one hand until
       its thumb is above its index finger and the lines cross over.
    6. Filter only the pixels inside that 4-sided shape and draw a white outline.
       If a hand drops out for a split second, the last shape is kept briefly
       so the portal doesn't flicker.

Run it (Mac / Linux):
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python filter_portal.py
Run it (Windows): double-click "Filter Portal.bat", or
    py -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
    .venv\Scripts\python filter_portal.py

Note: recent MediaPipe releases dropped the old `mp.solutions.hands` API, so
this uses the newer "Tasks" HandLandmarker. It needs a small model file
(hand_landmarker.task, ~8 MB), which is downloaded next to this script the
first time you run it.
"""

import os
import shutil
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# --------------------------------------------------------------------------
# Settings: tweak these freely
# --------------------------------------------------------------------------

WINDOW_NAME = "Filter Portal"
CAMERA_INDEX = 0             # 0 = default webcam; try 1, 2... for others
CAMERA_SIZE = (1280, 720)    # requested resolution (camera may pick another)
CAMERA_WAIT_SECONDS = 30     # how long to wait for you to allow camera access

# How much of the previous position to keep each frame (0.0 to 0.95).
# 0 = raw MediaPipe output (jittery); higher = smoother but laggier.
SMOOTHING = 0.5

# How sure MediaPipe must be that it sees a hand (0.0 to 1.0).
# Lower = hands are found and kept more easily, but more false detections.
HAND_CONFIDENCE = 0.3

# Keep the portal open this many seconds after a hand briefly drops out,
# so it doesn't flicker off every time tracking loses a hand for a moment.
PORTAL_HOLD_SECONDS = 0.5

OUTLINE_COLOR = (255, 255, 255)   # BGR (OpenCV uses Blue-Green-Red order)
OUTLINE_THICKNESS = 2

# Filter strengths
PURPLE_BGR = (200, 40, 150)   # tint colour
PURPLE_OPACITY = 0.45         # 0 = invisible, 1 = solid purple
BLUR_KERNEL = (35, 35)        # must be odd numbers; bigger = blurrier
EDGE_THRESHOLDS = (60, 140)   # Canny low/high; lower = more edges
PIXEL_SIZE = 16               # size of each "pixel" block, in real pixels

# Pinky-thumb gesture
GESTURE_THRESHOLD = None   # None = use the value picked during training;
                           # or set 0..1 (higher = needs a clearer touch)
GESTURE_FRAMES = 3         # touch must last this many frames to count

# MediaPipe landmark numbers (see the MediaPipe hand landmark diagram)
WRIST = 0
THUMB_TIP = 4
INDEX_BASE = 5
INDEX_TIP = 8
MIDDLE_BASE = 9
PINKY_BASE = 17
PINKY_TIP = 20

# Model files, stored next to this script
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
GESTURE_MODEL_PATH = os.path.join(HERE, "pinky_thumb_model.npz")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/latest/"
             "hand_landmarker.task")


# --------------------------------------------------------------------------
# Filters
# Each takes a BGR image and returns a NEW BGR image of the same size.
# To add one: write a function here and add it to FILTERS below.
# --------------------------------------------------------------------------

def purple_tint(img):
    """Blend the image with a solid purple layer."""
    purple = np.full_like(img, PURPLE_BGR)
    return cv2.addWeighted(img, 1 - PURPLE_OPACITY, purple, PURPLE_OPACITY, 0)


def negative(img):
    """Invert every colour (255 - value)."""
    return cv2.bitwise_not(img)


def grayscale(img):
    """Drop the colour. Converted back to 3 channels so it can be pasted in."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def blur(img):
    """Heavy Gaussian blur."""
    return cv2.GaussianBlur(img, BLUR_KERNEL, 0)


def edges(img):
    """Canny edge detection: white outlines on black."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)   # removes noise -> cleaner edges
    lines = cv2.Canny(gray, *EDGE_THRESHOLDS)
    return cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)


def pixelate(img):
    """Shrink the image, then blow it back up without smoothing."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // PIXEL_SIZE), max(1, h // PIXEL_SIZE)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# Number key -> (name shown on screen, filter function)
FILTERS = {
    "1": ("Purple Tint", purple_tint),
    "2": ("Negative", negative),
    "3": ("Grayscale", grayscale),
    "4": ("Blur", blur),
    "5": ("Edges", edges),
    "6": ("Pixelate", pixelate),
}


# --------------------------------------------------------------------------
# Hand tracking
# --------------------------------------------------------------------------

def ensure_model():
    """Download the MediaPipe hand model once, if it isn't there yet."""
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand-tracking model (one time only)...")
    # Use certifi's certificates if available: python.org installs on macOS
    # often can't verify HTTPS sites out of the box.
    try:
        import ssl
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None
    partial = MODEL_PATH + ".part"
    try:
        with urllib.request.urlopen(MODEL_URL, context=context) as response, \
                open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(partial, MODEL_PATH)   # only "finish" once fully downloaded
    except Exception as err:
        if os.path.exists(partial):
            os.remove(partial)
        sys.exit(f"Could not download the hand model: {err}\n"
                 f"Download it manually from:\n  {MODEL_URL}\n"
                 f"and save it as:\n  {MODEL_PATH}")
    print("Model saved to", MODEL_PATH)


def open_camera(wait_seconds=CAMERA_WAIT_SECONDS):
    """
    Open the webcam, or return None if it can't be opened.

    The first time an app uses the camera, macOS asks "allow camera
    access?" and the camera stays blocked until you answer. So instead of
    giving up straight away, keep trying for a while. (Quitting at once
    would also make the question disappear before you could answer it.)
    """
    deadline = time.monotonic() + wait_seconds
    waiting_message_shown = False
    while True:
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
            return camera
        camera.release()
        if time.monotonic() >= deadline:
            return None
        if not waiting_message_shown:
            print("Waiting for camera access. If macOS asks, click Allow.")
            waiting_message_shown = True
        time.sleep(1)


if sys.platform == "darwin":
    CAMERA_HELP = ("Could not open the webcam. Allow camera access for Filter "
                   "Portal (or for the terminal you started it from) in System "
                   "Settings > Privacy & Security > Camera, then open it "
                   "again.")
elif sys.platform == "win32":
    CAMERA_HELP = ("Could not open the webcam. Close other apps that might be "
                   "using it, and check Settings > Privacy & security > "
                   "Camera: 'Camera access' and 'Let desktop apps access your "
                   "camera' must be on. Then start Filter Portal again.")
else:
    CAMERA_HELP = ("Could not open the webcam. Check it's plugged in, not used "
                   "by another app, and that you're allowed to use it (on "
                   "Linux, your user may need to be in the 'video' group).")


def create_hand_tracker():
    """Create a MediaPipe HandLandmarker set up for video and 2 hands."""
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,   # reuses tracking between frames
        num_hands=2,
        min_hand_detection_confidence=HAND_CONFIDENCE,
        min_hand_presence_confidence=HAND_CONFIDENCE,
        min_tracking_confidence=HAND_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(tracker, frame_bgr, timestamp_ms):
    """
    Find up to 2 hands. Returns a list with one (21, 2) array per hand: the
    pixel positions of MediaPipe's 21 hand landmarks (wrist, knuckles, tips).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # MediaPipe wants RGB
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = tracker.detect_for_video(image, timestamp_ms)

    # Landmarks come back as 0..1 fractions of the image; convert to pixels.
    h, w = frame_bgr.shape[:2]
    return [np.array([(lm.x * w, lm.y * h) for lm in hand], dtype=np.float32)
            for hand in result.hand_landmarks[:2]]


# --------------------------------------------------------------------------
# Pinky-thumb gesture
# --------------------------------------------------------------------------

def hand_features(hand_xy):
    """
    Turn a hand's 21 landmark positions into 42 numbers that describe only
    its *shape*, so the gesture model doesn't care where the hand is, how big
    it is, how it's tilted, or whether it's a left or right hand:
      - move the wrist to (0, 0)
      - rotate so the base of the middle finger points straight up
      - scale so the wrist-to-middle-finger-base distance is 1
      - mirror so the index finger side is always on the left
    train_gesture_model.py uses this same function, so the model sees
    exactly the same kind of input during training and in the app.
    Returns None for a degenerate hand (all points on top of each other).
    """
    pts = np.asarray(hand_xy, dtype=np.float32)[:, :2]
    pts = pts - pts[WRIST]
    size = float(np.hypot(*pts[MIDDLE_BASE]))
    if size < 1e-6:
        return None
    ux, uy = pts[MIDDLE_BASE] / size
    rotate_upright = np.array([[-uy, ux], [-ux, -uy]], dtype=np.float32)
    pts = pts @ rotate_upright.T / size
    if pts[INDEX_BASE, 0] > pts[PINKY_BASE, 0]:
        pts[:, 0] = -pts[:, 0]
    return pts.ravel()


class PinkyThumbDetector:
    """
    The small neural network trained by train_gesture_model.py. Given one
    hand's landmarks, it gives the probability (0..1) that the thumb tip is
    touching the pinky tip. Runs with plain numpy (no scikit-learn needed).
    """

    def __init__(self, path, threshold=None):
        with np.load(path) as model:   # "with" closes the file afterwards
            count = int(model["num_layers"])
            self.layers = [(model[f"weight{i}"], model[f"bias{i}"])
                           for i in range(count)]
            saved_threshold = float(model["threshold"])
        self.threshold = saved_threshold if threshold is None else threshold

    def probability(self, hand_xy):
        x = hand_features(hand_xy)
        if x is None:
            return 0.0
        for i, (weight, bias) in enumerate(self.layers):
            x = x @ weight + bias
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0)        # ReLU between layers
        logit = float(np.clip(x[0], -30, 30))   # clip avoids exp overflow
        return 1 / (1 + np.exp(-logit))         # sigmoid -> probability

    def is_touching(self, hand_xy):
        return self.probability(hand_xy) >= self.threshold


class TouchTrigger:
    """
    Turns a per-frame "touching / not touching" signal into one event per
    touch: it fires after the touch has lasted `frames` frames, then waits
    until you've let go (for `frames` frames) before it can fire again.
    This stops a held touch, or a flickery detection, from switching
    filters over and over.
    """

    def __init__(self, frames):
        self.frames = frames
        self.on_count = 0
        self.off_count = 0
        self.armed = True

    def update(self, touching):
        if touching:
            self.on_count += 1
            self.off_count = 0
        else:
            self.off_count += 1
            self.on_count = 0
        if self.armed and self.on_count >= self.frames:
            self.armed = False
            return True
        if not self.armed and self.off_count >= self.frames:
            self.armed = True
        return False


def load_gesture_detector():
    """Load the pinky-thumb model if it has been trained, else None."""
    if not os.path.exists(GESTURE_MODEL_PATH):
        print("No pinky_thumb_model.npz found: pinky-thumb switching is off. "
              "Run train_gesture_model.py to create it.")
        return None
    return PinkyThumbDetector(GESTURE_MODEL_PATH, GESTURE_THRESHOLD)


def next_filter(key):
    """The filter key after `key`, wrapping from the last back to the first."""
    keys = list(FILTERS)
    return keys[(keys.index(key) + 1) % len(keys)]


# --------------------------------------------------------------------------
# Shape helpers
# --------------------------------------------------------------------------

def portal_corners(hands):
    """
    The portal's 4 corners from two hands, in drawing order:
        left index tip -> right index tip -> right thumb tip -> left thumb tip
    so the index fingers make the top edge and the thumbs the bottom edge.
    "Left" and "right" are the hands' places on screen (by wrist), which
    stays the same even when MediaPipe lists the hands in a different order.

    The corners are deliberately NOT re-sorted: twist one hand until its
    thumb is above its index finger and the top and bottom edges cross,
    making an X-shaped "bow-tie" portal.
    """
    left, right = sorted(hands[:2], key=lambda hand: hand[WRIST, 0])
    return np.array([left[INDEX_TIP], right[INDEX_TIP],
                     right[THUMB_TIP], left[THUMB_TIP]], dtype=np.float32)


class PointSmoother:
    """
    Smooths the 4 corners with an exponential moving average:
        smoothed = SMOOTHING * previous + (1 - SMOOTHING) * new
    Each corner always belongs to the same fingertip (see portal_corners),
    so old and new corners can simply be averaged one-to-one.
    """

    def __init__(self, amount):
        self.amount = amount
        self.previous = None

    def reset(self):
        """Forget the old position (call when the hands disappear)."""
        self.previous = None

    def update(self, points):
        if self.previous is None:
            self.previous = points
        else:
            self.previous = (self.amount * self.previous
                             + (1 - self.amount) * points)
        return self.previous


def apply_filter_inside(frame, shape, filter_fn):
    """
    Apply filter_fn to the pixels inside `shape` (an int32 array of corner
    points), changing `frame` in place. Everything outside stays untouched.
    """
    frame_h, frame_w = frame.shape[:2]

    # Only filter the rectangle around the shape (clipped to the frame).
    # This is much faster than filtering the whole frame.
    x, y, w, h = cv2.boundingRect(shape)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, frame_w), min(y + h, frame_h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return   # shape is off-screen or too thin to bother

    region = frame[y0:y1, x0:x1]   # a view: writing to it edits `frame`
    filtered = filter_fn(region)

    # Mask: 255 inside the shape, 0 outside (in the region's coordinates).
    mask = np.zeros(region.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [shape - (x0, y0)], 255)
    inside = mask == 255

    region[inside] = filtered[inside]


# --------------------------------------------------------------------------
# On-screen text
# --------------------------------------------------------------------------

def put_text(frame, text, position, scale=0.7):
    """White text with a black outline, readable on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, position, font, scale, (255, 255, 255), 2,
                cv2.LINE_AA)


def draw_hud(frame, filter_name, hands_seen, portal_open):
    put_text(frame, f"Filter: {filter_name}", (20, 40), scale=0.9)
    put_text(frame, "1-6 or thumb on pinky: switch filter   "
                    "D: touch confidence   Q: quit", (20, 75), scale=0.55)
    if not portal_open:
        put_text(frame, f"Hands seen: {hands_seen}/2 - show both hands, "
                        "thumbs and index fingers spread", (20, 105),
                 scale=0.55)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    ensure_model()

    camera = open_camera()
    if camera is None:
        sys.exit(CAMERA_HELP)

    tracker = create_hand_tracker()
    gesture = load_gesture_detector()
    touch_trigger = TouchTrigger(GESTURE_FRAMES)
    smoother = PointSmoother(SMOOTHING)
    current_filter = "1"
    last_timestamp = -1
    portal_shape = None    # corners of the current portal, or None if closed
    last_seen_both = 0.0   # when both hands were last visible
    show_confidence = False   # D key toggles the touch-confidence readout

    cv2.namedWindow(WINDOW_NAME)
    print("Filter Portal running. Keys 1-6 switch filters"
          + (" (or touch thumb to pinky)" if gesture else "") + ", Q quits.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Lost the webcam feed.")
                break

            frame = cv2.flip(frame, 1)   # mirror, like looking in a mirror

            # VIDEO mode needs timestamps that always go up.
            timestamp = max(int(time.monotonic() * 1000), last_timestamp + 1)
            last_timestamp = timestamp

            hands = detect_hands(tracker, frame, timestamp)
            hands_seen = len(hands)
            now = time.monotonic()

            # Thumb touching pinky on either hand -> next filter.
            touching = False
            if gesture is not None:
                touch_probs = [gesture.probability(hand) for hand in hands]
                touching = any(p >= gesture.threshold for p in touch_probs)
                if touch_trigger.update(touching):
                    current_filter = next_filter(current_filter)

            if hands_seen >= 2:
                # Both hands visible: update the portal shape. While a hand
                # does the pinky-thumb touch, its thumb leaves the corner, so
                # the shape is frozen until it lets go.
                if not touching:
                    smoothed = smoother.update(portal_corners(hands))
                    portal_shape = smoothed.round().astype(np.int32)
                last_seen_both = now
            elif now - last_seen_both > PORTAL_HOLD_SECONDS:
                # A hand has been gone too long: close the portal.
                # (If it only just dropped out, keep the last shape.)
                smoother.reset()
                portal_shape = None

            if portal_shape is not None:
                filter_fn = FILTERS[current_filter][1]
                apply_filter_inside(frame, portal_shape, filter_fn)
                cv2.polylines(frame, [portal_shape], isClosed=True,
                              color=OUTLINE_COLOR,
                              thickness=OUTLINE_THICKNESS,
                              lineType=cv2.LINE_AA)
            else:
                # No portal: show the normal video, plus a dot on each
                # fingertip we can see so you know tracking is working.
                for hand in hands:
                    for tip in (THUMB_TIP, INDEX_TIP):
                        x, y = hand[tip]
                        cv2.circle(frame, (int(x), int(y)), 8, OUTLINE_COLOR,
                                   OUTLINE_THICKNESS, cv2.LINE_AA)

            # D key: show how sure the model is that each hand is touching.
            if show_confidence and gesture is not None:
                for hand, p in zip(hands, touch_probs):
                    x, y = hand[WRIST]
                    label = "TOUCH" if p >= gesture.threshold else "touch"
                    put_text(frame, f"{label} {p:.0%}",
                             (int(x) - 50, int(y) + 35), scale=0.7)

            draw_hud(frame, FILTERS[current_filter][0], hands_seen,
                     portal_open=portal_shape is not None)
            cv2.imshow(WINDOW_NAME, frame)

            # Keyboard: waitKey also gives the window time to redraw.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # 27 = Esc
                break
            if chr(key) in FILTERS:
                current_filter = chr(key)
            if key in (ord("d"), ord("D")):
                show_confidence = not show_confidence

            # Quit if the window's close button was clicked.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
