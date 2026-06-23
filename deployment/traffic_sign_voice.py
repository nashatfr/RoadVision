"""
Traffic Sign Detection with Priority Voice Announcements
=========================================================
Uses a YOLO ONNX model to detect traffic signs from a video capture,
then announces them via MP3 voice messages with a priority queue and
a 5-second staleness timeout.


"""

import argparse
import heapq
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ── optional pygame for audio (preferred); falls back to playsound ─────────────
try:
    import pygame
    pygame.mixer.init()
    AUDIO_BACKEND = "pygame"
except ImportError:
    try:
        from playsound import playsound
        AUDIO_BACKEND = "playsound"
    except ImportError:
        AUDIO_BACKEND = None
        print("[WARN] No audio backend found. Install pygame: pip install pygame")

# ── ONNX Runtime ───────────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
except ImportError:
    raise SystemExit("onnxruntime not installed. Run: pip install onnxruntime")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = {
    0:  "Circle",
    1:  "Curve Ahead",
    2:  "Exit Not Allowed",
    3:  "Give Way",
    4:  "Green Light",
    5:  "Hump",
    6:  "No Entry",
    7:  "No Stop",
    8:  "Pass Either",
    9:  "Pass to the right",
    10: "Pedestrian Cross",
    11: "Red Light",
    12: "Speed Limit 100",
    13: "Speed Limit 120",
    14: "Speed Limit 20",
    15: "Speed Limit 30",
    16: "Speed Limit 40",
    17: "Speed Limit 50",
    18: "Speed Limit 60",
    19: "Speed Limit 70",
    20: "Speed Limit 80",
    21: "Speed Limit 90",
    22: "Stop",
    23: "U-Turn",
    24: "U-turn not allowed"
}

# Lower number = higher priority (0 is most urgent)
# Grouped by driver urgency:
#   0  → immediate stop / danger
#   1  → yield / caution
#   2  → restrictions (no entry, no stop, u-turn, exit)
#   3  → speed limits & informational
PRIORITY = {
    11: 0,   # Red Light        — STOP NOW
    22: 0,   # Stop             — STOP NOW
    6:  0,   # No Entry         — STOP NOW

    3:  1,   # Give Way         — yield
    10: 1,   # Pedestrian Cross — yield
    5:  1,   # Hump             — slow down

    2:  2,   # Exit Not Allowed
    7:  2,   # No Stop
    23: 2,   # U-Turn
    24: 2,
    1:  2,   # Curve Ahead

    4:  3,   # Green Light
    8:  3,   # Pass Either
    9:  3,   # Pass to the right
    0:  3,   # Circle

    12: 3,   # Speed Limit 100
    13: 3,   # Speed Limit 120
    14: 3,   # Speed Limit 20
    15: 3,   # Speed Limit 30
    16: 3,   # Speed Limit 40
    17: 3,   # Speed Limit 50
    18: 3,   # Speed Limit 60
    19: 3,   # Speed Limit 70
    20: 3,   # Speed Limit 80
    21: 3,   # Speed Limit 90
}

CONFIDENCE_THRESHOLD = 0.60   # minimum detection confidence to consider
QUEUE_TIMEOUT_SEC    = 5.0    # discard queued items older than this
COOLDOWN_SEC         = 8.0    # seconds before the same sign can re-queue

# All speed-limit class IDs — share a single cooldown group so that
# detecting '50' right after '60' doesn't trigger a second announcement.
SPEED_LIMIT_IDS: frozenset[int] = frozenset({12, 13, 14, 15, 16, 17, 18, 19, 20, 21})
MODEL_INPUT_SIZE     = 640    # YOLO input resolution (width = height)

# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _play_audio_file(path: str) -> None:
    """Blocking call to play one MP3 file."""
    if AUDIO_BACKEND == "pygame":
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        except Exception as e:
            print(f"[AUDIO ERROR] pygame: {e}")
    elif AUDIO_BACKEND == "playsound":
        try:
            playsound(path, block=True)
        except Exception as e:
            print(f"[AUDIO ERROR] playsound: {e}")
    else:
        print(f"[AUDIO] (no backend) Would play: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  VOICE MESSAGE MAP  (scans the "voice messages" folder)
# ═══════════════════════════════════════════════════════════════════════════════

def build_voice_map(voice_dir: str) -> dict[int, str]:
    """
    Scan the voice messages folder and build a {class_id: filepath} dict.
    Files are expected to start with their class number, e.g. "11 Red Light.mp3".
    """
    voice_map: dict[int, str] = {}
    voice_path = Path(voice_dir)
    if not voice_path.exists():
        print(f"[WARN] Voice directory not found: {voice_dir}")
        return voice_map

    for f in voice_path.iterdir():
        if f.suffix.lower() not in (".mp3", ".wav", ".ogg"):
            continue
        # Extract leading integer from filename
        stem = f.stem.strip()
        parts = stem.split(maxsplit=1)
        if parts and parts[0].isdigit():
            class_id = int(parts[0])
            voice_map[class_id] = str(f)

    found = sorted(voice_map.keys())
    missing = [i for i in CLASS_NAMES if i not in voice_map]
    print(f"[VOICE] Loaded {len(voice_map)} files. Missing IDs: {missing if missing else 'none'}")
    return voice_map


# ═══════════════════════════════════════════════════════════════════════════════
#  PRIORITY QUEUE ITEM
# ═══════════════════════════════════════════════════════════════════════════════

class QueueItem:
    """
    Wraps a detection for the priority queue.
    Heap ordering: (priority_level, enqueue_time) — lower is better.
    """
    __slots__ = ("class_id", "priority", "enqueue_time", "confidence")

    def __init__(self, class_id: int, confidence: float):
        self.class_id    = class_id
        self.priority    = PRIORITY.get(class_id, 99)
        self.enqueue_time = time.monotonic()
        self.confidence  = confidence

    # heapq is a min-heap; lower tuple = higher urgency
    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.enqueue_time < other.enqueue_time   # older first on tie

    def is_stale(self) -> bool:
        return (time.monotonic() - self.enqueue_time) > QUEUE_TIMEOUT_SEC


# ═══════════════════════════════════════════════════════════════════════════════
#  VOICE ANNOUNCER  (runs in its own thread)
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceAnnouncer(threading.Thread):
    """
    Background thread that pops items from the priority heap and plays them.
    Skips stale items and enforces per-class cooldowns.
    """

    def __init__(self, voice_map: dict[int, str]):
        super().__init__(daemon=True, name="VoiceAnnouncer")
        self._heap: list[QueueItem] = []
        self._heap_lock  = threading.Lock()
        self._stop_event = threading.Event()
        self.voice_map   = voice_map
        self._last_played: dict[int, float] = defaultdict(float)  # class_id → monotonic time
        self._queued_ids: set[int] = set()   # IDs currently sitting in the heap

    # ── public API ──────────────────────────────────────────────────────────────

    def _speed_limit_cooldown_active(self) -> bool:
        """Return True if any speed-limit sign was played/queued recently."""
        now = time.monotonic()
        return any(
            (now - self._last_played[sid]) < COOLDOWN_SEC
            for sid in SPEED_LIMIT_IDS
        )

    def enqueue(self, class_id: int, confidence: float) -> bool:
        """
        Thread-safe enqueue. Returns True if the item was accepted.
        Rejects if:
          - No voice file available
          - Already in queue
          - Played too recently (per-class cooldown)
          - Is a speed-limit sign and any other speed-limit was played recently
        """
        if class_id not in self.voice_map:
            return False

        now = time.monotonic()
        since_last = now - self._last_played[class_id]

        with self._heap_lock:
            if class_id in self._queued_ids:
                return False   # already waiting
            if since_last < COOLDOWN_SEC:
                return False   # individual cooldown
            # Group cooldown: suppress any speed-limit if one already queued/played
            if class_id in SPEED_LIMIT_IDS:
                if any(sid in self._queued_ids for sid in SPEED_LIMIT_IDS):
                    return False   # another speed-limit already in queue
                if self._speed_limit_cooldown_active():
                    return False   # a speed-limit was announced recently

            item = QueueItem(class_id, confidence)
            heapq.heappush(self._heap, item)
            self._queued_ids.add(class_id)

        label = CLASS_NAMES.get(class_id, str(class_id))
        print(f"[QUEUE +] {label:25s}  conf={confidence:.2f}  "
              f"priority={PRIORITY.get(class_id, 99)}")
        return True

    def stop(self) -> None:
        self._stop_event.set()

    # ── internal loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        while not self._stop_event.is_set():
            item = self._pop_next()
            if item is None:
                time.sleep(0.05)
                continue

            label = CLASS_NAMES.get(item.class_id, str(item.class_id))

            if item.is_stale():
                print(f"[QUEUE ✗] {label:25s}  DISCARDED (stale after "
                      f"{time.monotonic() - item.enqueue_time:.1f}s)")
                continue

            age = time.monotonic() - item.enqueue_time
            print(f"[PLAY  ►] {label:25s}  age={age:.2f}s")
            _play_audio_file(self.voice_map[item.class_id])

    def _pop_next(self) -> QueueItem | None:
        with self._heap_lock:
            if not self._heap:
                return None
            item = heapq.heappop(self._heap)
            self._queued_ids.discard(item.class_id)
            # Mark as "playing" immediately so the detection loop cannot
            # re-enqueue this sign while audio is still playing.
            now = time.monotonic()
            self._last_played[item.class_id] = now
            # If it's a speed-limit, stamp the whole group so no other
            # speed-limit can sneak in while this one is still playing.
            if item.class_id in SPEED_LIMIT_IDS:
                for sid in SPEED_LIMIT_IDS:
                    self._last_played[sid] = now
            return item


# ═══════════════════════════════════════════════════════════════════════════════
#  YOLO ONNX DETECTOR  (YOLO v10 / v11 / v12 / v26 — NMS-free)
# ═══════════════════════════════════════════════════════════════════════════════

class YOLODetector:
    """
    YOLO v10+ uses a one-to-one head assignment so the model emits at most
    one box per object — NMS is NOT needed and must NOT be applied.

    ONNX output shape from ultralytics export:
        [1, num_predictions, 6]   →  each row: [x1, y1, x2, y2, confidence, class_id]

    Boxes are in *letterboxed* coordinate space and must be mapped back to the
    original frame before use.
    """

    def __init__(self, model_path: str):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session    = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        input_shape     = self.session.get_inputs()[0].shape   # [1, 3, H, W]
        self.input_h    = input_shape[2] if isinstance(input_shape[2], int) else MODEL_INPUT_SIZE
        self.input_w    = input_shape[3] if isinstance(input_shape[3], int) else MODEL_INPUT_SIZE
        print(f"[MODEL] Loaded  : {model_path}")
        print(f"[MODEL] Input   : {self.input_w}×{self.input_h}")
        print(f"[MODEL] Provider: {self.session.get_providers()[0]}")
        # Log raw output shape on first init so the user can verify
        dummy = np.zeros((1, 3, self.input_h, self.input_w), dtype=np.float32)
        out   = self.session.run(None, {self.input_name: dummy})
        print(f"[MODEL] Output shape: {out[0].shape}  (expected [1, N, 6])")

    def preprocess(self, frame: np.ndarray):
        """Letterbox → RGB → CHW → float32 [0,1] → batch dim."""
        img, ratio, (dw, dh) = self._letterbox(frame, (self.input_w, self.input_h))
        img = img[:, :, ::-1].transpose(2, 0, 1)          # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img[np.newaxis], ratio, dw, dh

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Returns list of dicts: {class_id, confidence, box: [x1, y1, x2, y2]}
        Coordinates are in original (un-letterboxed) frame space.
        No NMS applied — the model's one-to-one head guarantees unique detections.
        """
        orig_h, orig_w = frame.shape[:2]
        inp, ratio, dw, dh = self.preprocess(frame)

        raw = self.session.run(None, {self.input_name: inp})[0]
        # raw: [1, N, 6] → drop batch dim → [N, 6]
        pred = raw[0]

        if pred.ndim == 1:
            # Degenerate single-detection edge case
            pred = pred[np.newaxis]

        # Each row: [x1, y1, x2, y2, confidence, class_id]
        confidences = pred[:, 4]
        class_ids   = pred[:, 5].astype(int)
        boxes_lb    = pred[:, :4]            # still in letterbox space

        # Filter by confidence threshold
        mask        = confidences >= CONFIDENCE_THRESHOLD
        confidences = confidences[mask]
        class_ids   = class_ids[mask]
        boxes_lb    = boxes_lb[mask]

        if len(boxes_lb) == 0:
            return []

        # Map letterbox coords → original frame coords
        x1 = np.clip((boxes_lb[:, 0] - dw) / ratio, 0, orig_w)
        y1 = np.clip((boxes_lb[:, 1] - dh) / ratio, 0, orig_h)
        x2 = np.clip((boxes_lb[:, 2] - dw) / ratio, 0, orig_w)
        y2 = np.clip((boxes_lb[:, 3] - dh) / ratio, 0, orig_h)

        results = []
        for i in range(len(class_ids)):
            results.append({
                "class_id":   int(class_ids[i]),
                "confidence": float(confidences[i]),
                "box":        [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
            })
        return results

    # ── internals ───────────────────────────────────────────────────────────────

    @staticmethod
    def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
        """Resize with preserved aspect ratio and pad to square."""
        h, w    = img.shape[:2]
        r       = min(new_shape[1] / h, new_shape[0] / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw      = (new_shape[0] - new_unpad[0]) / 2
        dh      = (new_shape[1] - new_unpad[1]) / 2
        img     = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top,    bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left,   right  = int(round(dw - 0.1)), int(round(dw + 0.1))
        img     = cv2.copyMakeBorder(img, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)


# ═══════════════════════════════════════════════════════════════════════════════
#  OVERLAY DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

# Colour palette per priority level
PRIORITY_COLORS = {
    0: (0,   0,   255),   # red   — danger
    1: (0,   165, 255),   # orange — caution
    2: (0,   255, 255),   # yellow — restriction
    3: (0,   200, 0  ),   # green  — informational
}

def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    for det in detections:
        cls   = det["class_id"]
        conf  = det["confidence"]
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label  = CLASS_NAMES.get(cls, str(cls))
        prio   = PRIORITY.get(cls, 99)
        color  = PRIORITY_COLORS.get(prio, (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, queue_size: int, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Queue: {queue_size}", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),  2, cv2.LINE_AA)
    # Priority legend (bottom-left)
    legend = [("DANGER",      PRIORITY_COLORS[0]),
              ("CAUTION",     PRIORITY_COLORS[1]),
              ("RESTRICTION", PRIORITY_COLORS[2]),
              ("INFO",        PRIORITY_COLORS[3])]
    for i, (lbl, col) in enumerate(legend):
        y = h - 15 - i * 22
        cv2.rectangle(frame, (10, y - 14), (24, y), col, -1)
        cv2.putText(frame, lbl, (30, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

# iVCam advertises itself under these substrings (case-insensitive).
# EpocCam and DroidCam are included as common alternatives.
_IVCAM_KEYWORDS = ("ivcam", "e2esoft", "epoccam", "droidcam", "phone camera")


def _find_ivcam_or_default(max_cams: int = 6) -> int:
    """
    Scan camera indices 0..max_cams-1 and return the index whose backend name
    matches a known iVCam keyword.  Falls back to index 0 (laptop webcam) if
    none is found.

    NOTE: cv2.VideoCapture.getBackendName() returns the *API* name (DSHOW,
    MSMF, V4L2 …), not the device name.  To get the actual device name on
    Windows we open the capture with the DirectShow backend and read the
    CAP_PROP_BACKEND property, then cross-check against a friendly-name list
    built with the 'pygrabber' library when available.  When pygrabber is not
    installed we fall back to index-based heuristics (iVCam is almost always
    index 1 when a laptop webcam is already index 0).
    """
    print("[CAM] Scanning for iVCam / phone camera ...")

    # ── Try pygrabber for reliable device names (Windows) ───────────────────
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph   = FilterGraph()
        devices = graph.get_input_devices()   # list of friendly name strings
        for idx, name in enumerate(devices):
            print(f"[CAM]   [{idx}] {name}")
            if any(kw in name.lower() for kw in _IVCAM_KEYWORDS):
                print(f"[CAM] iVCam found at index {idx} → '{name}'")
                return idx
        print("[CAM] iVCam not found via device names — using laptop webcam (0)")
        return 0
    except Exception:
        pass   # pygrabber not installed or failed — fall through

    # ── Fallback: probe each index, prefer the one that isn't index 0 ────────
    # iVCam on Windows typically registers as index 1 when the built-in
    # webcam occupies index 0.  We open each cap briefly and pick the first
    # non-zero index that works; if none, return 0.
    working = []
    for idx in range(max_cams):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            working.append(idx)
            cap.release()

    print(f"[CAM] Found camera indices: {working}")

    if len(working) >= 2:
        # Assume the laptop webcam is the lowest index; iVCam is the next one
        chosen = working[1]
        print(f"[CAM] Multiple cameras found — assuming iVCam at index {chosen}")
        print(f"[CAM] Override with --source <index> if this is wrong.")
        return chosen

    print("[CAM] Only one camera found — using index 0 (laptop webcam)")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run(source, model_path: str, voice_dir: str, display: bool = True) -> None:
    # ── load voice files ────────────────────────────────────────────────────────
    voice_map = build_voice_map(voice_dir)

    # ── start announcer thread ──────────────────────────────────────────────────
    announcer = VoiceAnnouncer(voice_map)
    announcer.start()

    # ── load YOLO model ─────────────────────────────────────────────────────────
    detector = YOLODetector(model_path)

    # ── open video source ───────────────────────────────────────────────────────
    try:
        src = int(source)          # webcam index
    except (ValueError, TypeError):
        src = source               # file path / RTSP URL

    # Auto-detect iVCam only when no explicit source was given
    if src == 0:
        src = _find_ivcam_or_default()

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {src}")

    print(f"\n[INFO] Running on source: {src}")
    print(f"[INFO] Press 'q' to quit\n")

    fps_counter = 0
    fps_time    = time.monotonic()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # End of file → loop or exit
                if isinstance(src, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # ── detect ──────────────────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── enqueue voices ──────────────────────────────────────────────────
            for det in detections:
                announcer.enqueue(det["class_id"], det["confidence"])

            # ── draw ────────────────────────────────────────────────────────────
            if display:
                frame = draw_detections(frame, detections)

                # FPS calculation
                fps_counter += 1
                elapsed = time.monotonic() - fps_time
                if elapsed >= 1.0:
                    fps_display  = fps_counter / elapsed
                    fps_counter  = 0
                    fps_time     = time.monotonic()

                with announcer._heap_lock:
                    q_size = len(announcer._heap)

                frame = draw_hud(frame, q_size, fps_display)
                cv2.imshow("Traffic Sign Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        announcer.stop()
        print("[INFO] Stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Sign Voice Announcer")
    parser.add_argument("--source",  default="0",
                        help="Video source: 0=webcam, or path to video file")
    parser.add_argument("--model",   default="best.onnx",
                        help="Path to YOLO ONNX model")
    parser.add_argument("--voices",  default="voice messages",
                        help="Path to voice messages folder")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable OpenCV window (headless mode)")
    args = parser.parse_args()

    run(
        source     = args.source,
        model_path = args.model,
        voice_dir  = args.voices,
        display    = not args.no_display,
    )
