"""
Tests for filter_portal.py. They never use the webcam or open a window:
the main-loop test swaps the camera, hand tracker and window for fakes.

Run from the project folder:
    .venv/bin/python -m unittest discover tests -v
"""

import os
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import filter_portal as fp  # noqa: E402

# Real hands from the HaGRIDv2 test set (CC BY-SA 4.0 variant), as pixel
# landmarks in a 1080x1920 frame. "three" is the pinky-thumb touch.
SAMPLE_HANDS = {
    "three": [[789, 1015], [760, 1004], [750, 968], [768, 933], [787, 907], [734, 876], [717, 816], [709, 776], [702, 740], [761, 865], [753, 793], [751, 746], [747, 706], [785, 874], [791, 812], [790, 769], [788, 729], [808, 900], [812, 872], [805, 893], [798, 918]],
    "peace": [[862, 1368], [825, 1329], [813, 1277], [821, 1232], [838, 1200], [822, 1210], [811, 1146], [807, 1111], [804, 1079], [857, 1213], [881, 1143], [895, 1103], [907, 1072], [884, 1233], [890, 1182], [867, 1202], [851, 1223], [904, 1262], [896, 1237], [875, 1228], [857, 1222]],
    "fist": [[656, 1084], [645, 1079], [638, 1069], [638, 1059], [642, 1052], [641, 1058], [639, 1051], [640, 1059], [642, 1063], [647, 1057], [646, 1051], [646, 1060], [647, 1062], [653, 1057], [652, 1051], [652, 1060], [652, 1063], [659, 1058], [658, 1052], [657, 1059], [657, 1063]],
    "four": [[413, 1051], [408, 1045], [406, 1035], [410, 1025], [413, 1019], [403, 1022], [400, 1009], [398, 999], [398, 992], [408, 1021], [406, 1006], [405, 996], [405, 989], [413, 1022], [414, 1009], [414, 1000], [416, 993], [418, 1027], [421, 1020], [424, 1015], [427, 1010]],
}


class FilterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        self.shape = np.array([[300, 200], [900, 200], [900, 600], [300, 600]],
                              np.int32)
        self.mask = np.zeros((720, 1280), np.uint8)
        cv2.fillPoly(self.mask, [self.shape], 255)

    def test_each_filter_changes_only_the_inside(self):
        for key, (name, fn) in fp.FILTERS.items():
            with self.subTest(filter=name):
                frame = self.frame.copy()
                fp.apply_filter_inside(frame, self.shape, fn)
                outside, inside = self.mask == 0, self.mask == 255
                self.assertTrue(np.array_equal(frame[outside],
                                               self.frame[outside]))
                self.assertFalse(np.array_equal(frame[inside],
                                                self.frame[inside]))

    def test_tiny_and_offscreen_shapes_dont_crash(self):
        shapes = [[[5, 5], [7, 5], [7, 8], [5, 8]],
                  [[-50, -50], [30, -40], [40, 20], [-60, 30]],
                  [[2000, 2000], [2100, 2000], [2100, 2100], [2000, 2100]]]
        for fn in (f for _, f in fp.FILTERS.values()):
            for shape in shapes:
                fp.apply_filter_inside(self.frame.copy(),
                                       np.array(shape, np.int32), fn)


def make_hand(wrist, index_tip, thumb_tip):
    hand = np.zeros((21, 2), np.float32)
    hand[fp.WRIST], hand[fp.INDEX_TIP], hand[fp.THUMB_TIP] = (
        wrist, index_tip, thumb_tip)
    return hand


def segments_cross(a, b):
    """True if line segment a (2 points) crosses line segment b."""
    def side(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1])
                       - (q[1] - p[1]) * (r[0] - p[0]))
    return (side(a[0], a[1], b[0]) != side(a[0], a[1], b[1])
            and side(b[0], b[1], a[0]) != side(b[0], b[1], a[1]))


class ShapeTests(unittest.TestCase):
    LEFT = make_hand(wrist=(300, 600), index_tip=(400, 200),
                     thumb_tip=(400, 500))
    RIGHT = make_hand(wrist=(1000, 600), index_tip=(900, 200),
                      thumb_tip=(900, 500))

    def test_index_tips_top_thumbs_bottom_in_any_hand_order(self):
        for hands in ([self.LEFT, self.RIGHT], [self.RIGHT, self.LEFT]):
            corners = fp.portal_corners(hands)
            np.testing.assert_array_equal(
                corners, [[400, 200], [900, 200], [900, 500], [400, 500]])
            self.assertFalse(segments_cross(corners[[0, 1]], corners[[2, 3]]))

    def test_twisted_hand_makes_lines_cross(self):
        # Left hand twisted: its thumb is now above its index finger.
        twisted = make_hand(wrist=(300, 600), index_tip=(400, 500),
                            thumb_tip=(400, 200))
        corners = fp.portal_corners([twisted, self.RIGHT])
        top_edge, bottom_edge = corners[[0, 1]], corners[[2, 3]]
        self.assertTrue(segments_cross(top_edge, bottom_edge))

        # The filter fills both triangles of the X, not the gap between.
        frame = np.full((720, 1280, 3), 100, np.uint8)
        fp.apply_filter_inside(frame, corners.astype(np.int32), fp.negative)
        self.assertTrue((frame[350, 450] != 100).all())    # left triangle
        self.assertTrue((frame[350, 850] != 100).all())    # right triangle
        self.assertTrue((frame[250, 650] == 100).all())    # above the X

    def test_smoother_averages_each_corner(self):
        smoother = fp.PointSmoother(0.5)
        corners = fp.portal_corners([self.LEFT, self.RIGHT])
        smoother.update(corners)
        np.testing.assert_allclose(smoother.update(corners + 2), corners + 1)


class GestureTests(unittest.TestCase):
    def test_features_ignore_position_size_rotation_and_mirroring(self):
        hand = np.array(SAMPLE_HANDS["three"], np.float32)
        base = fp.hand_features(hand)
        angle = np.deg2rad(37)
        rotate = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
        moved = (hand - 500) @ rotate.T * 2.3 + 900
        np.testing.assert_allclose(fp.hand_features(moved), base, atol=1e-4)
        np.testing.assert_allclose(fp.hand_features(hand * (-1, 1)), base,
                                   atol=1e-4)

    def test_degenerate_hand_gives_none(self):
        self.assertIsNone(fp.hand_features(np.zeros((21, 2))))

    def test_trigger_fires_once_per_touch(self):
        trigger = fp.TouchTrigger(3)
        sequence = [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                    0, 0, 0, 1, 0, 1, 0]
        fired = [i for i, on in enumerate(sequence) if trigger.update(bool(on))]
        # 2: first touch. A 1-frame let-go (5) doesn't re-arm. 14: second
        # touch after a real let-go. 1-frame blips at the end are ignored.
        self.assertEqual(fired, [2, 14])

    def test_next_filter_wraps_around(self):
        self.assertEqual([fp.next_filter(k) for k in "123456"], list("234561"))

    @unittest.skipUnless(os.path.exists(fp.GESTURE_MODEL_PATH),
                         "pinky_thumb_model.npz not trained yet")
    def test_model_on_real_hands(self):
        detector = fp.PinkyThumbDetector(fp.GESTURE_MODEL_PATH)
        for name, hand in SAMPLE_HANDS.items():
            with self.subTest(hand=name):
                self.assertEqual(detector.is_touching(np.array(hand)),
                                 name == "three")


class MainLoopTest(unittest.TestCase):
    """Runs main() for scripted frames with fake camera, hands and window."""

    FPS = 30

    def run_main(self, script, touching_frames=(), keys=None,
                 snapshot_folder=None, displayed=None):
        """
        keys: {frame number: key pressed after that frame, e.g. "s"}.
        displayed: if a list, gets a copy of every frame shown on screen.
        Returns one (filter shown or None if the text was hidden, portal
        shape or None) per frame.
        """
        keys = keys or {}
        clock = {"t": 1000.0}
        frame_no = {"i": -1}
        log = []

        class FakeCamera:
            def __init__(self, *args): pass
            def set(self, *args): pass
            def isOpened(self): return True
            def release(self): pass
            def read(self):
                if frame_no["i"] + 1 >= len(script):
                    return False, None
                frame_no["i"] += 1
                clock["t"] += 1 / MainLoopTest.FPS
                return True, np.full((720, 1280, 3), 100, np.uint8)

        def fake_detect(tracker, frame, timestamp):
            points = script[frame_no["i"]]
            hands = []
            for k in range(len(points) // 2):
                hand = np.zeros((21, 2), np.float32)
                hand[fp.THUMB_TIP] = points[2 * k]
                hand[fp.INDEX_TIP] = points[2 * k + 1]
                hand[fp.WRIST] = (points[2 * k] + points[2 * k + 1]) / 2
                hands.append(hand)
            return hands

        class FakeGesture:
            threshold = 0.5
            def probability(self, hand):
                return 1.0 if frame_no["i"] in touching_frames else 0.0

        shown = {}
        def fake_hud(frame, filter_name, hands_seen, portal_open):
            shown["filter"] = filter_name
            frame[0, 0] = 255   # stands in for the on-screen text
        def fake_apply(frame, shape, fn):
            shown["shape"] = shape.copy()
        def fake_imshow(name, frame):
            log.append((shown.pop("filter", None), shown.pop("shape", None)))
            if displayed is not None:
                displayed.append(frame.copy())
        def fake_wait_key(*args):
            key = keys.get(frame_no["i"])
            return ord(key) if key else 255

        patches = [
            mock.patch.object(fp, "ensure_model", lambda: None),
            mock.patch.object(fp, "create_hand_tracker",
                              lambda: mock.Mock()),
            mock.patch.object(fp, "load_gesture_detector", FakeGesture),
            mock.patch.object(fp, "detect_hands", fake_detect),
            mock.patch.object(fp, "draw_hud", fake_hud),
            mock.patch.object(fp, "apply_filter_inside", fake_apply),
            mock.patch.object(fp.time, "monotonic", lambda: clock["t"]),
            mock.patch.object(fp.cv2, "VideoCapture", FakeCamera),
            mock.patch.object(fp.cv2, "namedWindow", lambda *a: None),
            mock.patch.object(fp.cv2, "imshow", fake_imshow),
            mock.patch.object(fp.cv2, "waitKey", fake_wait_key),
            mock.patch.object(fp.cv2, "getWindowProperty", lambda *a: 1),
            mock.patch.object(fp.cv2, "destroyAllWindows", lambda: None),
            mock.patch("builtins.print", lambda *a, **k: None),
            mock.patch.object(fp, "SNAPSHOT_FOLDER",
                              snapshot_folder or "/nonexistent-test-folder"),
        ]
        for p in patches:
            p.start()
        try:
            fp.main()
        finally:
            for p in reversed(patches):
                p.stop()
        return log

    TWO = np.array([[400, 200], [420, 500], [900, 200], [880, 500]],
                   np.float32)

    def test_portal_holds_through_short_dropouts_only(self):
        one = self.TWO[:2]
        script = [self.TWO] * 30 + [one] * 6 + [self.TWO] * 15 + [one] * 45
        log = self.run_main(script)
        is_open = [shape is not None for _, shape in log]
        self.assertTrue(all(is_open[:51]))      # incl. a 0.2 s dropout
        self.assertTrue(all(is_open[51:63]))    # first 0.4 s of a long one
        self.assertFalse(any(is_open[78:]))     # closed after 0.5 s

    def test_touch_switches_filter_and_freezes_portal(self):
        moving = [self.TWO + (k * 5, 0) for k in range(40)]
        touch = set(range(10, 20))
        log = self.run_main(moving, touching_frames=touch)
        filters = [name for name, _ in log]
        self.assertEqual(set(filters[:12]), {"Purple Tint"})
        self.assertEqual(set(filters[12:]), {"Negative"})   # switched once
        frozen = [shape for _, shape in log[10:20]]
        for shape in frozen:
            np.testing.assert_array_equal(shape, frozen[0])
        self.assertFalse(np.array_equal(log[25][1], frozen[0]))

    def test_s_key_saves_picture_without_the_text(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            displayed = []
            self.run_main([self.TWO] * 10, keys={3: "s"},
                          snapshot_folder=folder, displayed=displayed)
            files = os.listdir(folder)
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].startswith("Filter Portal ")
                            and files[0].endswith(".png"))
            picture = cv2.imread(os.path.join(folder, files[0]))
        self.assertEqual(picture.shape, (720, 1280, 3))
        self.assertTrue((picture[0, 0] == 100).all())        # no text in it
        self.assertTrue((displayed[4][0, 0] == 255).all())   # but on screen

    def test_h_key_hides_and_shows_the_text(self):
        log = self.run_main([self.TWO] * 10, keys={2: "h", 6: "H"})
        text_drawn = [name is not None for name, _ in log]
        self.assertEqual(text_drawn, [True] * 3 + [False] * 4 + [True] * 3)

    def test_failed_save_shows_message_and_keeps_running(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            blocker = os.path.join(folder, "not-a-folder")
            open(blocker, "w").close()   # a file where the folder should be
            log = self.run_main([self.TWO] * 10, keys={3: "s"},
                                snapshot_folder=os.path.join(blocker, "x"))
        self.assertEqual(len(log), 10)   # every frame was still shown


class SnapshotTests(unittest.TestCase):
    def test_pictures_never_overwrite_and_handle_non_english_names(self):
        import tempfile
        frame = np.full((48, 64, 3), 77, np.uint8)
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, "Фото Filter Portal")   # Cyrillic
            first = fp.save_snapshot(frame, folder)
            second = fp.save_snapshot(frame, folder)
            self.assertNotEqual(first, second)
            self.assertEqual(sorted(os.listdir(folder)),
                             sorted([os.path.basename(first),
                                     os.path.basename(second)]))
            with open(second, "rb") as f:
                data = np.frombuffer(f.read(), np.uint8)
            self.assertTrue((cv2.imdecode(data, cv2.IMREAD_COLOR) == 77).all())


class CameraTests(unittest.TestCase):
    """open_camera() waits for the macOS permission answer (fake camera)."""

    def try_open(self, fails_before_open, wait_seconds):
        clock = {"t": 0.0}
        attempts = {"n": 0}

        class FakeCamera:
            def __init__(self, index):
                attempts["n"] += 1
                self.ok = attempts["n"] > fails_before_open
            def isOpened(self): return self.ok
            def set(self, *args): pass
            def release(self): pass

        def fake_sleep(seconds):
            clock["t"] += seconds

        with mock.patch.object(fp.cv2, "VideoCapture", FakeCamera), \
                mock.patch.object(fp.time, "monotonic", lambda: clock["t"]), \
                mock.patch.object(fp.time, "sleep", fake_sleep), \
                mock.patch("builtins.print", lambda *a, **k: None):
            return fp.open_camera(wait_seconds), attempts["n"]

    def test_waits_while_you_answer_the_permission_question(self):
        camera, attempts = self.try_open(fails_before_open=5, wait_seconds=30)
        self.assertIsNotNone(camera)
        self.assertEqual(attempts, 6)

    def test_gives_up_after_the_wait(self):
        camera, attempts = self.try_open(fails_before_open=999, wait_seconds=30)
        self.assertIsNone(camera)
        self.assertEqual(attempts, 31)   # one try per second, plus the first


class TrainingDataTests(unittest.TestCase):
    """Your-own-recordings file and the made-up touches (no camera)."""

    def setUp(self):
        import tempfile
        self.folder = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.folder.name, "data", "samples.npz")

    def tearDown(self):
        self.folder.cleanup()

    def test_recordings_are_appended_not_overwritten(self):
        from record_my_gesture import save_samples
        touch = np.array(SAMPLE_HANDS["three"], np.float32)
        other = np.array(SAMPLE_HANDS["fist"], np.float32)
        self.assertEqual(save_samples(self.path, [touch] * 3, [1] * 3), 3)
        self.assertEqual(save_samples(self.path, [other] * 2, [0] * 2), 5)
        with np.load(self.path) as saved:
            self.assertEqual(saved["hands"].shape, (5, 21, 2))
            self.assertEqual(saved["labels"].tolist(), [1, 1, 1, 0, 0])

    def test_newest_recordings_are_kept_for_testing(self):
        import train_gesture_model as tg
        from record_my_gesture import save_samples
        # 8 touch hands, each one slightly different so we can tell them apart
        hands = [np.array(SAMPLE_HANDS["three"], np.float32) * (1 + k / 100)
                 for k in range(8)]
        save_samples(self.path, hands, [1] * 8)
        save_samples(self.path, [np.array(SAMPLE_HANDS["fist"])] * 4, [0] * 4)
        X_train, y_train, X_test, y_test = tg.load_my_samples(
            np.random.default_rng(0), self.path)
        self.assertEqual(len(y_train), (6 + 3) * tg.MY_SAMPLE_COPIES)
        self.assertEqual(sorted(y_test.tolist()), [0, 1, 1])
        np.testing.assert_allclose(X_test[y_test == 1],
                                   [fp.hand_features(h) for h in hands[6:]],
                                   atol=1e-5)

    def test_made_up_touch_puts_thumb_on_pinky_and_keeps_fingers(self):
        import train_gesture_model as tg
        rng = np.random.default_rng(0)
        hand = np.array(SAMPLE_HANDS["four"], np.float32)
        touch = tg.synthetic_touch(hand, rng)
        palm = np.linalg.norm(hand[fp.MIDDLE_BASE] - hand[fp.WRIST])
        gap = np.linalg.norm(touch[fp.THUMB_TIP] - touch[fp.PINKY_TIP])
        self.assertLessEqual(gap, 0.25 * palm + 1e-3)
        np.testing.assert_array_equal(touch[5:], hand[5:])   # fingers kept
        np.testing.assert_array_equal(touch[0], hand[0])     # wrist kept


if __name__ == "__main__":
    unittest.main()
