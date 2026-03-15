"""
gui.py — Webcam overlay GUI with frosted-glass info card.

Renders the live webcam feed full-frame, then blends a semi-transparent
rounded card on top showing:
  • Person name  (large, white)
  • Relationship tag  (teal pill badge)
  • Interaction summary  (smaller, light-grey, word-wrapped)

The card tracks the detected face in real time using OpenCV's Haar cascade.
A simple exponential moving average smooths the card position to avoid jitter.

Press  Q  or close the window to exit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class OverlayInfo:
    name: str
    relationship: str       # e.g. "Son", "Daughter", "Friend"
    summary: str


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

CARD_ALPHA   = 0.55          # 0 = fully transparent, 1 = fully opaque
CARD_COLOR   = (60, 60, 60)  # dark-grey (BGR)
TEAL         = (180, 160, 50) # teal badge (BGR)
WHITE        = (255, 255, 255)
LIGHT_GREY   = (210, 210, 210)
FONT         = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL   = cv2.FONT_HERSHEY_SIMPLEX


def _blend_rect(frame: np.ndarray, x: int, y: int, w: int, h: int,
                color: tuple[int, int, int], alpha: float,
                radius: int = 18) -> None:
    """Blend a rounded rectangle onto *frame* in-place."""
    overlay = frame.copy()

    # filled rounded rect via two rects + four circles
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    for cx, cy in [
        (x + radius,     y + radius),
        (x + w - radius, y + radius),
        (x + radius,     y + h - radius),
        (x + w - radius, y + h - radius),
    ]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _pill_badge(frame: np.ndarray, text: str, x: int, y: int,
                bg: tuple[int, int, int] = TEAL) -> int:
    """Draw a pill-shaped badge. Returns the right edge x coordinate."""
    (tw, th), _ = cv2.getTextSize(text, FONT_SMALL, 0.55, 1)
    pad_x, pad_y, r = 12, 6, 10
    bw = tw + pad_x * 2
    bh = th + pad_y * 2
    _blend_rect(frame, x, y - th - pad_y, bw, bh, bg, 1.0, radius=r)
    cv2.putText(frame, text, (x + pad_x, y), FONT_SMALL, 0.55, WHITE, 1, cv2.LINE_AA)
    return x + bw


def _draw_wrapped_text(frame: np.ndarray, text: str, x: int, y: int,
                       max_width_px: int, font_scale: float = 0.52,
                       color: tuple = LIGHT_GREY, thickness: int = 1,
                       line_gap: int = 22) -> None:
    """Word-wrap *text* to fit within *max_width_px* and draw each line."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(test, FONT_SMALL, font_scale, thickness)
        if tw <= max_width_px:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * line_gap),
                    FONT_SMALL, font_scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main card renderer
# ---------------------------------------------------------------------------

def draw_info_card(frame: np.ndarray, info: OverlayInfo,
                   card_x: int, card_y: int) -> None:
    """
    Overlay a semi-transparent info card on *frame* (in-place) at (card_x, card_y).
    """
    fh, fw = frame.shape[:2]
    card_w = min(310, fw // 2)
    card_h = 90

    # clamp so the card never goes off-screen
    card_x = max(0, min(card_x, fw - card_w))
    card_y = max(0, min(card_y, fh - card_h))

    # --- frosted card background ---
    _blend_rect(frame, card_x, card_y, card_w, card_h,
                CARD_COLOR, CARD_ALPHA, radius=12)

    pad = 10

    # --- name ---
    (nw, nh), _ = cv2.getTextSize(info.name, FONT, 0.7, 2)
    name_x = card_x + pad
    name_y = card_y + pad + nh
    cv2.putText(frame, info.name, (name_x, name_y),
                FONT, 0.7, WHITE, 2, cv2.LINE_AA)

    # --- relationship badge (right of name) ---
    badge_x = name_x + nw + 8
    _pill_badge(frame, info.relationship, badge_x, name_y, bg=TEAL)

    # --- summary (single compact line) ---
    text_y = name_y + 20
    _draw_wrapped_text(frame, info.summary,
                       card_x + pad, text_y,
                       max_width_px=card_w - pad * 2,
                       font_scale=0.42, line_gap=16)


# ---------------------------------------------------------------------------
# Face detector + background subtractor
# ---------------------------------------------------------------------------

def _load_face_detector() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise SystemExit("Could not load Haar cascade for face detection.")
    return detector


def _motion_roi(fg_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Find the bounding box of the largest motion blob in the foreground mask.
    Returns (x, y, w, h) or None if nothing significant is moving.
    """
    # clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 3000:   # ignore tiny blobs
        return None

    return cv2.boundingRect(largest)


def _detect_best_face(gray: np.ndarray,
                      detector: cv2.CascadeClassifier,
                      roi: tuple[int, int, int, int] | None = None
                      ) -> tuple[int, int, int, int] | None:
    """
    Detect the largest face in *gray*, optionally restricted to *roi*.
    Returns absolute (x, y, w, h) or None.
    """
    if roi is not None:
        rx, ry, rw, rh = roi
        # add padding so the face isn't clipped at the ROI edge
        pad = 40
        rx = max(0, rx - pad);  ry = max(0, ry - pad)
        rw = min(gray.shape[1] - rx, rw + pad * 2)
        rh = min(gray.shape[0] - ry, rh + pad * 2)
        search = gray[ry:ry + rh, rx:rx + rw]
    else:
        search = gray
        rx, ry = 0, 0

    faces = detector.detectMultiScale(search, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return (fx + rx, fy + ry, fw, fh)   # back to absolute coords


# ---------------------------------------------------------------------------
# Live webcam loop
# ---------------------------------------------------------------------------

SMOOTH = 0.15   # EMA factor — lower = smoother but slower, higher = snappier
GREEN  = (0, 255, 0)


def run(info: OverlayInfo | None = None, camera_index: int = 0,
        get_transcript: "Callable[[], str] | None" = None,
        get_info: "Callable[[], OverlayInfo] | None" = None) -> None:
    """
    Open the webcam and render the overlay in a loop.

    Args:
        info:           Static OverlayInfo. Ignored if get_info is provided.
        camera_index:   OpenCV camera index (default 0).
        get_transcript: Callable returning the latest transcript string.
        get_info:       Callable returning the current OverlayInfo (live-updating).
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    detector  = _load_face_detector()
    bg_sub    = cv2.createBackgroundSubtractorMOG2(
                    history=300, varThreshold=50, detectShadows=False)

    cv2.namedWindow("Remember Me", cv2.WINDOW_NORMAL)

    smooth_x: float | None = None
    smooth_y: float | None = None
    card_w = 310
    card_h = 90
    gap    = 8

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]

        # --- background subtraction ---
        fg_mask = bg_sub.apply(frame)
        roi     = _motion_roi(fg_mask)

        # --- face detection (scoped to motion region) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = _detect_best_face(gray, detector, roi)

        if face is not None:
            fx, fy, fw_face, fh_face = face

            # green bounding box
            cv2.rectangle(frame, (fx, fy), (fx + fw_face, fy + fh_face),
                          GREEN, 2, cv2.LINE_AA)

            # resolve current overlay info
            current_info = get_info() if get_info is not None else info

            if current_info is not None:
                target_x = fx + (fw_face - card_w) // 2
                target_y = fy - card_h - gap

                if smooth_x is None:
                    smooth_x, smooth_y = float(target_x), float(target_y)
                else:
                    smooth_x += SMOOTH * (target_x - smooth_x)
                    smooth_y += SMOOTH * (target_y - smooth_y)

        current_info = get_info() if get_info is not None else info
        if current_info is not None and smooth_x is not None:
            draw_info_card(frame, current_info, int(smooth_x), int(smooth_y))

        # --- recording indicator (top-left) ---
        dot_x, dot_y = 14, 14
        cv2.circle(frame, (dot_x, dot_y), 6, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.putText(frame, "REC", (dot_x + 12, dot_y + 5),
                    FONT_SMALL, 0.5, (0, 0, 220), 1, cv2.LINE_AA)

        # --- live transcript bar (bottom of frame) ---
        if get_transcript is not None:
            transcript = get_transcript()
            if transcript:
                bar_h  = 32
                bar_y  = fh - bar_h
                _blend_rect(frame, 0, bar_y, fw, bar_h, (30, 30, 30), 0.65, radius=0)
                # truncate to fit frame width
                max_chars = fw // 9
                display   = transcript if len(transcript) <= max_chars else "…" + transcript[-(max_chars - 1):]
                cv2.putText(frame, display, (10, bar_y + 21),
                            FONT_SMALL, 0.52, WHITE, 1, cv2.LINE_AA)

        cv2.imshow("Remember Me", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = OverlayInfo(
        name="Jake",
        relationship="Son",
        summary="Visited last week and talked about the hackathon he went to where he got 3rd place.",
    )
    run(demo)
