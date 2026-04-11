"""
Video Pipeline — OpenCV Frame Extraction + MediaPipe Face Mesh

Handles all the preprocessing before analysis modules receive data:
  1. Frame extraction at configurable FPS
  2. Face detection and cropping
  3. MediaPipe face mesh landmark extraction (468 points)
  4. Audio separation from video
"""

import cv2
import numpy as np
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Generator

_MP_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
# Face detector (Haar + DNN fallback)
# ─────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    OpenCV-based face detector. Uses DNN face detector if weights are
    available, falls back to Haar cascade.
    """

    def __init__(self, min_confidence: float = 0.7):
        self.min_conf = min_confidence
        self._init_dnn()

    def _init_dnn(self):
        """Try to load OpenCV DNN face detector (ResNet-SSD)."""
        prototxt = Path(__file__).parent.parent / 'models/deploy.prototxt'
        caffemodel = Path(__file__).parent.parent / 'models/res10_300x300_ssd_iter_140000.caffemodel'
        if prototxt.exists() and caffemodel.exists():
            self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
            self.mode = 'dnn'
        else:
            # Haar cascade — fast but less accurate
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.cascade = cv2.CascadeClassifier(cascade_path)
            self.mode = 'haar'

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Returns list of (x, y, w, h) face bounding boxes.
        """
        h, w = frame.shape[:2]
        if self.mode == 'dnn':
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0)
            )
            self.net.setInput(blob)
            detections = self.net.forward()
            boxes = []
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > self.min_conf:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
            return boxes
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            return [tuple(f) for f in faces] if len(faces) else []

    def crop_face(self, frame: np.ndarray,
                  box: Tuple[int, int, int, int],
                  scale: float = 1.3) -> np.ndarray:
        """Crop face region with padding scale. Returns BGR crop."""
        x, y, w, h = box
        cx, cy = x + w // 2, y + h // 2
        half = int(max(w, h) * scale / 2)
        fh, fw = frame.shape[:2]
        x1 = max(0,  cx - half)
        y1 = max(0,  cy - half)
        x2 = min(fw, cx + half)
        y2 = min(fh, cy + half)
        return frame[y1:y2, x1:x2]


# ─────────────────────────────────────────────────────────────────
# MediaPipe Face Mesh
# ─────────────────────────────────────────────────────────────────

class LandmarkExtractor:
    """
    Extracts 468 3D face mesh landmarks from a frame using MediaPipe.
    Returns normalized (x, y, z) coordinates for each landmark.
    """

    def __init__(self, max_faces: int = 1, min_detection_conf: float = 0.5):
        if not _MP_AVAILABLE:
            self.mesh = None
            return
        mp_face_mesh = mp.solutions.face_mesh
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,         # Enables iris landmarks
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract landmarks from a single frame.

        Returns:
            np.ndarray of shape (478, 3) with (x, y, z) normalized coords,
            or None if no face detected. Note: 478 when iris landmarks enabled.
        """
        if self.mesh is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    def close(self):
        if self.mesh is not None:
            self.mesh.close()


# ─────────────────────────────────────────────────────────────────
# Video reader
# ─────────────────────────────────────────────────────────────────

class VideoReader:
    """
    Reads a video file and yields frames with associated metadata.
    """

    def __init__(self, video_path: str, target_fps: Optional[float] = None):
        self.path = video_path
        self.cap  = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.target_fps = target_fps or self.source_fps
        self.frame_skip = max(1, round(self.source_fps / self.target_fps))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def duration_sec(self) -> float:
        return self.total_frames / (self.source_fps + 1e-8)

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame_bgr) at target_fps rate."""
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % self.frame_skip == 0:
                yield frame_idx, frame
            frame_idx += 1

    def read_sequence(self, max_frames: int = 300) -> List[np.ndarray]:
        """Read up to max_frames at target_fps into a list."""
        frames = []
        for _, frame in self.frames():
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        return frames

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ─────────────────────────────────────────────────────────────────
# Audio extractor
# ─────────────────────────────────────────────────────────────────

def extract_audio(video_path: str,
                  output_path: Optional[str] = None,
                  sr: int = 16000) -> Optional[str]:
    """
    Extract audio track from a video file using ffmpeg.
    Returns path to the extracted WAV, or None if ffmpeg unavailable.

    Requires: ffmpeg on PATH (install with: apt install ffmpeg / brew install ffmpeg).
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        output_path = tmp.name
        tmp.close()

    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(sr), '-ac', '1',
        output_path, '-loglevel', 'error',
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[WARNING] ffmpeg not found. Audio analysis will be skipped.")
        return None


# ─────────────────────────────────────────────────────────────────
# Full video preprocessing pipeline
# ─────────────────────────────────────────────────────────────────

class VideoPipeline:
    """
    Orchestrates frame extraction + landmark detection for a video file.

    Returns everything the analysis modules need:
      - frames:          list of BGR frames
      - landmarks_seq:   MediaPipe landmarks per frame (or None)
      - audio_path:      path to extracted WAV (or None)
      - metadata:        basic video info dict
    """

    def __init__(self,
                 target_fps: float = 15.0,
                 max_frames: int = 300,
                 min_face_confidence: float = 0.6):
        self.target_fps   = target_fps
        self.max_frames   = max_frames
        self.face_detector = FaceDetector(min_confidence=min_face_confidence)
        self.lm_extractor  = LandmarkExtractor()

    def process(self, video_path: str) -> dict:
        """
        Full pipeline: extract frames → detect faces → extract landmarks.

        Returns dict with:
            frames, landmarks_seq, audio_path, face_boxes, metadata
        """
        frames: List[np.ndarray]              = []
        landmarks_seq: List[Optional[np.ndarray]] = []
        face_boxes: List[Optional[Tuple]]     = []

        with VideoReader(video_path, target_fps=self.target_fps) as reader:
            meta = {
                'source_fps':    reader.source_fps,
                'duration_sec':  reader.duration_sec,
                'resolution':    (reader.width, reader.height),
                'total_frames':  reader.total_frames,
                'filename':      os.path.basename(video_path),
            }
            for _, frame in reader.frames():
                boxes = self.face_detector.detect(frame)
                lm    = self.lm_extractor.extract(frame)

                frames.append(frame)
                landmarks_seq.append(lm)
                face_boxes.append(boxes[0] if boxes else None)

                if len(frames) >= self.max_frames:
                    break

        # Extract audio
        audio_path = extract_audio(video_path)

        # Face coverage stats
        faces_detected = sum(1 for b in face_boxes if b is not None)
        meta['face_coverage'] = faces_detected / max(1, len(frames))

        return {
            'frames':        frames,
            'landmarks_seq': landmarks_seq,
            'face_boxes':    face_boxes,
            'audio_path':    audio_path,
            'metadata':      meta,
        }

    def cleanup(self):
        self.lm_extractor.close()
