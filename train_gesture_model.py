r"""
Train the pinky-thumb gesture model used by filter_portal.py
=============================================================

Data: HaGRIDv2 (HAnd Gesture Recognition Image Dataset v2)
    https://github.com/hukenovs/hagrid
    About a million labelled hands from thousands of people. We only need its
    annotation file, which includes MediaPipe hand landmarks for every hand,
    so no images are downloaded. License: a variant of CC BY-SA 4.0
    (credit the HaGRID authors if you share the trained model).

The gesture: HaGRID's "three" class is the hand with index, middle and ring
fingers up and the thumb pressed onto the pinky. That's our "touch". Every
other class (fist, peace, ok, palm, ...) plus "no_gesture" is "not a touch".

HaGRID has no photos of the touch made from other poses, like the L-shaped
portal hand, so a model trained on it alone missed those. To fix that, extra
"made-up" touches are built from real hands of every class by moving the
thumb onto the pinky tip (see synthetic_touch). False triggers are still
measured on real, unedited hands.

Your own examples: if you recorded some with record_my_gesture.py
(data/my_gesture_samples.npz), they're added to training too, so the model
also learns the way YOU do the touch, with your camera and lighting.

Steps:
    1. Download annotations.zip (~720 MB) into data/ (only the first time).
    2. Turn each hand's 21 landmarks into 42 shape numbers with the SAME
       hand_features() function the app uses.
    3. Train a small neural network (scikit-learn MLP) on HaGRID plus your
       own examples, if any.
    4. Pick a confidence threshold on the validation set that keeps false
       triggers rare, then report results on the held-out test set
       (different people from training), next to a simple distance rule.
    5. Save the weights as pinky_thumb_model.npz: plain numpy arrays, so
       the app runs it without scikit-learn.

Run (Mac / Linux):
    .venv/bin/pip install -r requirements-train.txt
    .venv/bin/python train_gesture_model.py
Run (Windows):
    .venv\Scripts\pip install -r requirements-train.txt
    .venv\Scripts\python train_gesture_model.py
"""

import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from filter_portal import (HERE, GESTURE_MODEL_PATH, PINKY_TIP, THUMB_TIP,
                           PinkyThumbDetector, hand_features)

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

ANNOTATIONS_URL = ("https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/"
                   "datasets/hagrid_v2/annotations_with_landmarks/"
                   "annotations.zip")
ANNOTATIONS_PATH = os.path.join(HERE, "data", "hagrid_v2_annotations.zip")

POSITIVE_CLASS = "three"   # thumb on pinky, other three fingers up

# HaGRID landmarks are fractions of the image width and height, and the
# dataset doesn't say each photo's shape. Hand proportions suggest a mix of
# portrait and landscape photos. So every hand is stretched by a random
# width/height ratio in this range: whatever the true shape was, the model
# learns to recognise the gesture at the webcam's real proportions too.
ASPECT_RANGE = (0.5, 1.8)

# HaGRID only shows the touch with the other three fingers up. To recognise
# it from any pose (the L-shaped portal hand, a fist, pointing...), extra
# touch examples are made from real hands of every other gesture by moving
# the thumb onto the pinky tip. Each original hand stays in as a "not a
# touch" example, so the model learns that the thumb is what matters.
SYNTHETIC_PER_CLASS = {"train": 2000, "val": 300, "test": 500}

# Your own recordings (record_my_gesture.py). They're few compared with
# HaGRID, so each one counts many times. The newest part is held back to
# test on.
MY_SAMPLES_PATH = os.path.join(HERE, "data", "my_gesture_samples.npz")
MY_SAMPLE_COPIES = 20
MY_TEST_SHARE = 0.25

# "three" hands whose thumb isn't actually near the pinky (a loose pose)
# would teach the wrong thing, so they're left out of training.
MAX_TOUCH_DISTANCE = 0.35  # in wrist-to-middle-knuckle lengths, at ratio 1

MAX_PER_CLASS = {"train": 15000, "val": 3000, "test": 5000}
POSITIVE_COPIES = 4        # augmented copies of each training "touch" hand
FEATURE_NOISE = 0.02       # simulated tracking jitter added in training
MAX_FALSE_TRIGGER_RATE = 0.001   # threshold target on validation (0.1%)
HIDDEN_LAYERS = (64, 32)
SEED = 0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def download_annotations():
    if os.path.exists(ANNOTATIONS_PATH):
        return
    os.makedirs(os.path.dirname(ANNOTATIONS_PATH), exist_ok=True)
    print("Downloading HaGRIDv2 annotations (~720 MB, one time only)...")
    try:
        import ssl
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None
    partial = ANNOTATIONS_PATH + ".part"
    with urllib.request.urlopen(ANNOTATIONS_URL, context=context) as response, \
            open(partial, "wb") as out:
        shutil.copyfileobj(response, out)
    os.replace(partial, ANNOTATIONS_PATH)


def thumb_pinky_distance(features):
    pts = features.reshape(21, 2)
    return float(np.hypot(*(pts[THUMB_TIP] - pts[PINKY_TIP])))


def synthetic_touch(xy, rng):
    """
    A copy of hand `xy` (21 landmarks) with the thumb moved onto the pinky
    tip, the way a real hand does it: the thumb's base swings in towards
    the palm, the tip lands near (not exactly on) the pinky tip, and the
    middle thumb joint sits roughly between them with a slight bend.
    """
    xy = xy.copy()
    palm_size = np.linalg.norm(xy[9] - xy[0])
    palm_centre = xy[[0, 5, 17]].mean(axis=0)
    xy[2] += (palm_centre - xy[2]) * rng.uniform(0.0, 0.35)   # thumb base
    angle = rng.uniform(0, 2 * np.pi)
    xy[4] = xy[20] + (palm_size * rng.uniform(0.03, 0.25)    # thumb tip
                      * np.array([np.cos(angle), np.sin(angle)]))
    along = xy[4] - xy[2]
    sideways = np.array([-along[1], along[0]])
    xy[3] = (xy[2] + along * rng.uniform(0.45, 0.6)          # thumb joint
             + sideways * rng.uniform(-0.15, 0.15))
    return xy


def load_split(archive, split, rng):
    """
    Read one split (train / val / test). Returns features X, labels y
    (1 = touch, 0 = not) and the class name of every row ("fist", or
    "fist+touch" for a made-up touch built from a fist hand).
    """
    augment = split == "train"
    features, labels, classes = [], [], []
    skipped_loose = 0

    def add(xy, label, class_name, copies=1):
        for _ in range(copies):
            aspect = np.exp(rng.uniform(*np.log(ASPECT_RANGE)))
            f = hand_features(xy * (aspect, 1))
            if f is None:
                continue
            if augment:
                f = f + rng.normal(0, FEATURE_NOISE, f.shape)
            features.append(f)
            labels.append(label)
            classes.append(class_name)

    files = sorted(n for n in archive.namelist()
                   if n.startswith(f"annotations/{split}/")
                   and n.endswith(".json") and "checkpoint" not in n)
    for name in files:
        images = json.loads(archive.read(name))
        hands = [(label, landmarks)
                 for image in images.values()
                 for label, landmarks in zip(image["labels"],
                                             image["hand_landmarks"])
                 if len(landmarks) == 21]
        rng.shuffle(hands)

        taken = {}
        for label, landmarks in hands:
            if taken.get(label, 0) >= MAX_PER_CLASS[split]:
                continue
            is_touch = label == POSITIVE_CLASS
            xy = np.array(landmarks, dtype=np.float32)

            square = hand_features(xy)
            if square is None:
                continue
            if is_touch and thumb_pinky_distance(square) > MAX_TOUCH_DISTANCE:
                skipped_loose += 1
                continue
            taken[label] = taken.get(label, 0) + 1

            if is_touch:
                add(xy, 1, label, POSITIVE_COPIES if augment else 1)
            else:
                add(xy, 0, label)
                if taken[label] <= SYNTHETIC_PER_CLASS[split]:
                    add(synthetic_touch(xy, rng), 1, label + "+touch")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int8)
    made_up = sum(c.endswith("+touch") for c in classes)
    print(f"  {split:5s}: {len(y):7d} hands  ({y.sum()} touch, of which "
          f"{made_up} made-up; {len(y) - y.sum()} not)  [skipped "
          f"{skipped_loose} loose '{POSITIVE_CLASS}' hands]")
    return X, y, np.array(classes)


def load_my_samples(rng, path=None):
    """
    Your own recordings, split into training and test parts, or None if you
    haven't recorded any. The newest MY_TEST_SHARE of each kind (touch / not)
    is the test part: neighbouring video frames are near-copies of each
    other, so a random split would test on almost the same frames it
    trained on. Training examples are copied with a little jitter.
    """
    path = path or MY_SAMPLES_PATH
    if not os.path.exists(path):
        return None
    with np.load(path) as data:   # "with" closes the file afterwards
        all_hands, all_labels = data["hands"], data["labels"]
    train_X, train_y, test_X, test_y = [], [], [], []
    for label in (0, 1):
        hands = all_hands[all_labels == label]   # oldest first
        cut = int(round(len(hands) * (1 - MY_TEST_SHARE)))
        for i, hand in enumerate(hands):
            f = hand_features(hand)
            if f is None:
                continue
            if i < cut:
                for _ in range(MY_SAMPLE_COPIES):
                    train_X.append(f + rng.normal(0, FEATURE_NOISE, f.shape))
                    train_y.append(label)
            else:
                test_X.append(f)
                test_y.append(label)
    as_arrays = (lambda X, y: (np.array(X, np.float32).reshape(-1, 42),
                               np.array(y, np.int8)))
    X_train, y_train = as_arrays(train_X, train_y)
    X_test, y_test = as_arrays(test_X, test_y)
    print(f"  yours: {int(all_labels.sum())} touch and "
          f"{int((all_labels == 0).sum())} not-touch hands recorded "
          f"({len(y_test)} kept for testing)")
    return X_train, y_train, X_test, y_test


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------

def threshold_for_false_rate(scores, y, max_rate):
    """Lowest threshold whose false-trigger rate on negatives is <= max_rate."""
    negatives = np.sort(scores[y == 0])
    index = int(np.ceil(len(negatives) * (1 - max_rate)))
    return float(negatives[min(index, len(negatives) - 1)])


def report(name, triggered, y, classes):
    recall = triggered[y == 1].mean()
    false_rate = triggered[y == 0].mean()
    print(f"  {name:28s} catches {recall:6.1%} of touches, "
          f"false triggers on {false_rate:.2%} of other hands")
    return recall, false_rate


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)
    download_annotations()

    print("Loading landmarks...")
    started = time.time()
    with zipfile.ZipFile(ANNOTATIONS_PATH) as archive:
        X_train, y_train, _ = load_split(archive, "train", rng)
        X_val, y_val, _ = load_split(archive, "val", rng)
        X_test, y_test, c_test = load_split(archive, "test", rng)
    mine = load_my_samples(rng)
    if mine is not None:
        X_train = np.concatenate([X_train, mine[0]])
        y_train = np.concatenate([y_train, mine[1]])
    print(f"  done in {time.time() - started:.0f}s")

    print("Training...")
    started = time.time()
    model = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, batch_size=512,
                          max_iter=60, early_stopping=True,
                          n_iter_no_change=5, random_state=SEED)
    model.fit(X_train, y_train)
    print(f"  {model.n_iter_} epochs in {time.time() - started:.0f}s")

    # Pick the threshold on validation data, then judge on test data.
    val_scores = model.predict_proba(X_val)[:, 1]
    print("\nValidation trade-off (choose with MAX_FALSE_TRIGGER_RATE):")
    for rate in (0.005, 0.002, 0.001, 0.0005):
        t = threshold_for_false_rate(val_scores, y_val, rate)
        caught = (val_scores[y_val == 1] >= t).mean()
        print(f"  false triggers <= {rate:.2%}: threshold {t:.3f}, "
              f"catches {caught:.1%} of touches")
    threshold = threshold_for_false_rate(val_scores, y_val,
                                         MAX_FALSE_TRIGGER_RATE)
    test_scores = model.predict_proba(X_test)[:, 1]

    # Baseline for comparison: "thumb tip close to pinky tip", with its
    # distance cut-off picked the same way.
    val_dist = np.array([thumb_pinky_distance(f) for f in X_val])
    test_dist = np.array([thumb_pinky_distance(f) for f in X_test])
    dist_cut = -threshold_for_false_rate(-val_dist, y_val,
                                         MAX_FALSE_TRIGGER_RATE)

    print(f"\nTest set (people not seen in training), "
          f"threshold {threshold:.3f}:")
    print(f"  model ROC AUC: {roc_auc_score(y_test, test_scores):.4f}")
    report("trained model", test_scores >= threshold, y_test, c_test)
    report(f"distance rule (< {dist_cut:.2f})", test_dist <= dist_cut,
           y_test, c_test)

    triggered = test_scores >= threshold
    real = c_test == POSITIVE_CLASS
    print(f"\n  Real touches ('{POSITIVE_CLASS}' photos): model catches "
          f"{triggered[real].mean():.1%}")
    print("  Made-up touches, by the pose the hand started from:")
    for c in sorted(np.unique(c_test[y_test == 1])):
        if c.endswith("+touch"):
            print(f"    {c[:-6]:16s} {triggered[c_test == c].mean():6.1%}")

    print("\n  Gestures the model most often mistakes for a touch:")
    rates = sorted(((triggered[c_test == c].mean(), c)
                    for c in np.unique(c_test[y_test == 0])), reverse=True)
    for rate, name in rates[:5]:
        print(f"    {name:16s} {rate:.2%}")

    # How it does on the held-back part of YOUR recordings.
    if mine is not None and len(mine[3]):
        X_mine, y_mine = mine[2], mine[3]
        hits = model.predict_proba(X_mine)[:, 1] >= threshold
        print("\nYour recordings (newest part, not used in training):")
        if (y_mine == 1).any():
            print(f"  catches {hits[y_mine == 1].mean():.1%} of your touches")
        if (y_mine == 0).any():
            print(f"  false triggers on {hits[y_mine == 0].mean():.1%} of "
                  "your other poses")

    # Save as plain numpy arrays.
    arrays = {"num_layers": np.array(len(model.coefs_)),
              "threshold": np.array(threshold),
              "positive_class": np.array(POSITIVE_CLASS)}
    for i, (weight, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
        arrays[f"weight{i}"] = weight.astype(np.float32)
        arrays[f"bias{i}"] = bias.astype(np.float32)
    np.savez(GESTURE_MODEL_PATH, **arrays)

    # Check the app's numpy version gives the same answers as scikit-learn.
    detector = PinkyThumbDetector(GESTURE_MODEL_PATH)
    sample = X_test[:: max(1, len(X_test) // 500)]
    numpy_probs = np.array([detector.probability(_unfeature(f))
                            for f in sample])
    gap = np.abs(numpy_probs - model.predict_proba(sample)[:, 1]).max()
    print(f"\nSaved {GESTURE_MODEL_PATH}")
    print(f"  app/numpy vs scikit-learn max difference: {gap:.1e}")
    if gap > 1e-3:
        sys.exit("Mismatch between saved model and trained model!")


def _unfeature(features):
    """Features are already landmark positions (normalised), so they can be
    fed back in as a 'hand': hand_features() of them returns them unchanged."""
    return features.reshape(21, 2)


if __name__ == "__main__":
    main()
