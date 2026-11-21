"""
Draws the pictures used in README.md, using the app's own filters.

    .venv/bin/python docs/make_pictures.py

No camera needed: the scene and the hands are drawn, not photographed.
Writes docs/how-it-works.png, docs/gesture.png and docs/filters.png.
"""

import os
import sys

import cv2
import numpy as np

DOCS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(DOCS))
import filter_portal as fp  # noqa: E402

SS = 3   # draw 3x bigger, then shrink: smooth edges

# Colours (OpenCV uses Blue, Green, Red)
GLOVE = (250, 246, 252)
GLOVE_EDGE = (110, 45, 85)
CARD = (48, 24, 36)
TEXT = (255, 255, 255)
GLOW = (60, 210, 255)


def bgr(rgb):
    return rgb[2], rgb[1], rgb[0]


def scene(w, h):
    """A simple, colourful landscape for the filters to work on."""
    W, H = w * SS, h * SS
    t = np.linspace(0, 1, H)[:, None, None]
    top, bottom = np.array(bgr((96, 165, 250))), np.array(bgr((255, 196, 150)))
    img = (top * (1 - t) + bottom * t).astype(np.uint8).repeat(W, axis=1)
    sun = (int(W * 0.76), int(H * 0.27))
    cv2.circle(img, sun, int(H * 0.12), bgr((255, 222, 89)), -1, cv2.LINE_AA)
    for cx, cy, r in [(0.18, 0.2, 0.06), (0.25, 0.18, 0.08), (0.32, 0.21, 0.05),
                      (0.55, 0.12, 0.045), (0.6, 0.1, 0.06)]:
        cv2.circle(img, (int(W * cx), int(H * cy)), int(H * r),
                   (255, 255, 255), -1, cv2.LINE_AA)
    far = np.array([[0, 0.62], [0.2, 0.5], [0.42, 0.6], [0.65, 0.45],
                    [0.85, 0.55], [1, 0.5], [1, 1], [0, 1]]) * (W, H)
    near = np.array([[0, 0.8], [0.3, 0.66], [0.55, 0.78], [0.8, 0.64],
                     [1, 0.72], [1, 1], [0, 1]]) * (W, H)
    cv2.fillPoly(img, [far.astype(np.int32)], bgr((84, 160, 120)), cv2.LINE_AA)
    cv2.fillPoly(img, [near.astype(np.int32)], bgr((46, 125, 80)), cv2.LINE_AA)
    # a little house, so edges / pixelate have something to show
    hx, hy, hs = int(W * 0.5), int(H * 0.74), int(H * 0.11)
    cv2.rectangle(img, (hx - hs, hy - hs), (hx + hs, hy + hs // 2),
                  bgr((244, 114, 82)), -1, cv2.LINE_AA)
    roof = np.array([[hx - hs - hs // 4, hy - hs], [hx, hy - 2 * hs],
                     [hx + hs + hs // 4, hy - hs]], np.int32)
    cv2.fillPoly(img, [roof], bgr((120, 60, 40)), cv2.LINE_AA)
    cv2.rectangle(img, (hx - hs // 4, hy - hs // 6), (hx + hs // 4, hy + hs // 2),
                  bgr((90, 50, 30)), -1, cv2.LINE_AA)
    return img


def part(img, points, width, edge=6):
    """One finger / arm: a thick rounded line with an outline."""
    pts = (np.array(points) * SS).astype(np.int32)
    for color, extra in ((GLOVE_EDGE, 2 * edge), (GLOVE, 0)):
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, tuple(a), tuple(b), color, int((width + extra) * SS / 2),
                     cv2.LINE_AA)


def palm(img, points, edge=6):
    pts = (np.array(points) * SS).astype(np.int32)
    cv2.polylines(img, [pts], True, GLOVE_EDGE, int(2 * edge * SS / 2) * 2,
                  cv2.LINE_AA)
    cv2.fillPoly(img, [pts], GLOVE, cv2.LINE_AA)


def frame_hand(img, mirror_x=None):
    """A hand making half of the frame: pointer finger up, thumb sideways.
    Returns (pointer tip, thumb tip). mirror_x flips it for the right hand."""
    def m(p):
        return (mirror_x - p[0], p[1]) if mirror_x else p
    def ms(points):
        return [m(p) for p in points]
    part(img, ms([(70, 600), (170, 455)]), 86)                       # arm
    palm(img, ms([(138, 470), (208, 474), (246, 418), (262, 338),
                  (222, 314), (180, 318), (146, 342)]))
    for chain in ([(232, 330), (246, 304), (240, 352)],            # curled
                  [(206, 330), (218, 306), (212, 356)],
                  [(180, 340), (190, 320), (186, 362)]):
        part(img, ms(chain), 30)
    pointer = [(256, 335), (272, 272), (287, 216), (298, 168)]
    thumb = [(228, 430), (266, 420), (302, 412), (338, 406)]
    part(img, ms(pointer), 32)
    part(img, ms(thumb), 34)
    return m(pointer[-1]), m(thumb[-1])


def portal_overlay(img, corners, filter_fn):
    """Filter inside the corners, white outline and fingertip dots, the way
    the app draws it."""
    shape = (np.array(corners) * SS).astype(np.int32)
    fp.apply_filter_inside(img, shape, filter_fn)
    cv2.polylines(img, [shape], True, (255, 255, 255), 3 * SS, cv2.LINE_AA)
    for x, y in shape:
        cv2.circle(img, (int(x), int(y)), 9 * SS, (255, 255, 255), -1,
                   cv2.LINE_AA)


def shrink(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def how_it_works():
    w, h = 960, 540
    img = scene(w, h)
    # Draw the portal first using the corners the hands will have, then the
    # hands on top (so the filter shows where the hands frame the scene).
    corners = [(298, 168), (w - 298, 168), (w - 338, 406), (338, 406)]
    portal_overlay(img, corners, fp.negative)
    frame_hand(img)
    frame_hand(img, mirror_x=w)
    dots = (np.array(corners) * SS).astype(np.int32)   # dots above the hands
    for x, y in dots:
        cv2.circle(img, (int(x), int(y)), 9 * SS, (255, 255, 255), -1,
                   cv2.LINE_AA)
    return rounded(shrink(img, w, h), 18)


def gesture():
    """Hand with the thumb tip on the pinky tip, the switch gesture."""
    w, h = 480, 420
    img = np.zeros((h * SS, w * SS, 3), np.uint8)
    img[:] = CARD
    cx, cy = 240, 360
    def at(points):
        return [(cx + x, cy + y) for x, y in points]
    part(img, at([(0, 90), (0, 0)]), 88)                              # arm
    palm(img, at([(-46, 10), (46, 10), (58, -60), (44, -110), (0, -118),
                  (-40, -112), (-60, -62)]))
    part(img, at([(42, -104), (60, -150), (54, -120), (34, -96)]), 30)  # pinky
    part(img, at([(22, -112), (30, -175), (34, -215), (36, -250)]), 32)  # ring
    part(img, at([(0, -116), (0, -185), (0, -228), (0, -262)]), 33)    # middle
    part(img, at([(-22, -112), (-30, -178), (-34, -218), (-36, -250)]), 32)
    part(img, at([(-50, -40), (-62, -80), (-20, -110), (26, -100)]), 34)  # thumb
    touch = (np.array([(cx + 32, cy - 100)]) * SS).astype(np.int32)[0]
    cv2.circle(img, tuple(touch), 30 * SS, GLOW, 4 * SS, cv2.LINE_AA)
    for angle in range(-150, 31, 45):      # little "click" lines
        a = np.deg2rad(angle)
        p = touch + (np.array([np.cos(a), np.sin(a)]) * 42 * SS).astype(int)
        q = touch + (np.array([np.cos(a), np.sin(a)]) * 60 * SS).astype(int)
        cv2.line(img, tuple(p), tuple(q), GLOW, 4 * SS, cv2.LINE_AA)
    return rounded(shrink(img, w, h), 18)


def filters_gallery():
    tile_w, tile_h, label_h, gap, pad = 300, 180, 46, 18, 22
    cols, rows = 3, 2
    W = pad * 2 + cols * tile_w + (cols - 1) * gap
    H = pad * 2 + rows * (tile_h + label_h) + (rows - 1) * gap
    card = np.zeros((H, W, 3), np.uint8)
    card[:] = CARD
    corners = [(70, 34), (232, 26), (244, 152), (58, 160)]
    for i, (key, (name, fn)) in enumerate(fp.FILTERS.items()):
        tile = scene(tile_w, tile_h)
        portal_overlay(tile, corners, fn)
        tile = rounded(shrink(tile, tile_w, tile_h), 12, CARD)
        x = pad + (i % cols) * (tile_w + gap)
        y = pad + (i // cols) * (tile_h + label_h + gap)
        card[y:y + tile_h, x:x + tile_w] = tile
        label = f"{key}  {name}"
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(card, label, (x + (tile_w - size[0]) // 2,
                                  y + tile_h + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT, 2, cv2.LINE_AA)
    return rounded(card, 18)


def rounded(img, radius, background=None):
    """Round the corners: transparent PNG, or a solid background colour."""
    h, w = img.shape[:2]
    mask = np.zeros((h * SS, w * SS), np.uint8)
    r = radius * SS
    cv2.rectangle(mask, (r, 0), (w * SS - r, h * SS), 255, -1)
    cv2.rectangle(mask, (0, r), (w * SS, h * SS - r), 255, -1)
    for cx in (r, w * SS - r - 1):
        for cy in (r, h * SS - r - 1):
            cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    alpha = shrink(mask, w, h)
    if background is not None:
        a = alpha[..., None] / 255.0
        return (img * a + np.array(background) * (1 - a)).astype(np.uint8)
    return np.dstack([img, alpha])


def save(name, img):
    path = os.path.join(DOCS, name)
    cv2.imwrite(path, img)
    print("wrote", path)


if __name__ == "__main__":
    save("how-it-works.png", how_it_works())
    save("gesture.png", gesture())
    save("filters.png", filters_gallery())
