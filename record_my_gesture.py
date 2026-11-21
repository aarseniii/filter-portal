"""
Record your own pinky-thumb examples
====================================

The gesture model learned from other people's photos (HaGRID). Your hand
pose, camera and lighting can be different, so about a minute of YOUR own
examples makes the switch much more reliable for you.

Only hand landmark positions are saved (21 x/y points per hand) into
data/my_gesture_samples.npz. No images or video are kept.

Keys (press once to start, again to stop):
    T = record TOUCH: thumb tip on pinky tip, the way you'll do it in the
        app. Move and tilt your hand, use each hand, and try it while
        holding the portal pose with the other hand.
    N = record NOT a touch: the portal pose, open hand, fist, pointing,
        peace sign... anything you do in the app that should NOT switch.
    S = save and quit        Q / Esc = quit without saving

Aim for about 30 seconds of each. Running it again adds to what you
recorded before. Then retrain:
    .venv/bin/python train_gesture_model.py
"""

import os
import sys
import time

import cv2
import numpy as np

import filter_portal as fp

SAMPLES_PATH = os.path.join(fp.HERE, "data", "my_gesture_samples.npz")
WINDOW_NAME = "Record your gesture"
MODE_KEYS = {ord("t"): 1, ord("T"): 1, ord("n"): 0, ord("N"): 0}
MODE_NAMES = {1: "TOUCH", 0: "NOT a touch"}


def save_samples(path, hands, labels):
    """
    Add recorded hands (each a (21, 2) array) and labels (1 = touch,
    0 = not) to the samples file, keeping anything recorded earlier.
    Returns the total number of samples in the file.
    """
    hands = np.asarray(hands, dtype=np.float32).reshape(-1, 21, 2)
    labels = np.asarray(labels, dtype=np.int8)
    if os.path.exists(path):
        old = np.load(path)
        hands = np.concatenate([old["hands"], hands])
        labels = np.concatenate([old["labels"], labels])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, hands=hands, labels=labels)
    return len(labels)


def main():
    fp.ensure_model()
    camera = fp.open_camera()
    if camera is None:
        sys.exit(fp.CAMERA_HELP)

    tracker = fp.create_hand_tracker()
    gesture = fp.load_gesture_detector()   # shows the current model's guess
    recorded_hands, recorded_labels = [], []
    mode = None            # None = paused, 1 = touch, 0 = not a touch
    last_timestamp = -1

    cv2.namedWindow(WINDOW_NAME)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Lost the webcam feed.")
                break
            frame = cv2.flip(frame, 1)
            timestamp = max(int(time.monotonic() * 1000), last_timestamp + 1)
            last_timestamp = timestamp

            hands = fp.detect_hands(tracker, frame, timestamp)
            if mode is not None:
                for hand in hands:
                    recorded_hands.append(hand)
                    recorded_labels.append(mode)

            # Show the tracked points and the current model's opinion.
            for hand in hands:
                for x, y in hand:
                    cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)
                if gesture is not None:
                    x, y = hand[fp.WRIST]
                    fp.put_text(frame,
                                f"model: touch {gesture.probability(hand):.0%}",
                                (int(x) - 80, int(y) + 35), scale=0.6)

            touches = sum(recorded_labels)
            status = ("Paused" if mode is None
                      else f"RECORDING: {MODE_NAMES[mode]}")
            fp.put_text(frame, status, (20, 40), scale=0.9)
            fp.put_text(frame, "T: touch   N: not a touch   S: save & quit   "
                               "Q: quit without saving", (20, 75), scale=0.55)
            fp.put_text(frame, f"This session: {touches} touch, "
                               f"{len(recorded_labels) - touches} not-touch "
                               "hands", (20, 105), scale=0.55)
            if mode is not None:   # red "recording" dot
                cv2.circle(frame, (frame.shape[1] - 40, 40), 14,
                           (0, 0, 255), -1)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in MODE_KEYS:
                new_mode = MODE_KEYS[key]
                mode = None if mode == new_mode else new_mode
            elif key in (ord("s"), ord("S")):
                if recorded_labels:
                    total = save_samples(SAMPLES_PATH, recorded_hands,
                                         recorded_labels)
                    print(f"Saved {len(recorded_labels)} hands "
                          f"({touches} touch). {total} in total in "
                          f"{SAMPLES_PATH}")
                    print("Now retrain: .venv/bin/python "
                          "train_gesture_model.py")
                else:
                    print("Nothing recorded, nothing saved.")
                break
            elif key in (ord("q"), ord("Q"), 27):
                print("Quit without saving.")
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed without saving.")
                break
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
