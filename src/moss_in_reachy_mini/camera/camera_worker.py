"""Camera worker thread with frame buffering and face tracking.

Ported from main_works.py camera_worker() function to provide:
- 30Hz+ camera polling with thread-safe frame buffering
- Face tracking integration with smooth interpolation
- Latest frame always available for tools
"""

import time
import logging
import threading

import cv2
import numpy as np
from ghoshell_common.contracts import Workspace
from ghoshell_container import Provider, IoCContainer, INSTANCE
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils.interpolation import linear_pose_interpolation

from moss_in_reachy_mini.camera.face_recognizer import FaceRecognizer
from moss_in_reachy_mini.camera.drawer import draw_detections
from moss_in_reachy_mini.camera.model import get_position_by_track_name, CameraFrame
from moss_in_reachy_mini.camera.frame_hub import FrameHub

logger = logging.getLogger(__name__)


class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking."""

    def __init__(self, reachy_mini: ReachyMini, frame_hub: FrameHub, face_recognizer: FaceRecognizer) -> None:
        """Initialize."""
        self.reachy_mini = reachy_mini
        self.frame_hub = frame_hub
        self.face_recognizer = face_recognizer

        # Thread-safe frame storage
        self.latest_frame = CameraFrame.new()

        self.frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.face_tracking_lock = threading.Lock()
        self.target_track_name = ""

        # Face tracking timing variables (same as main_works.py)
        self.last_face_detected_time: float | None = None
        self.interpolation_start_time: float | None = None
        self.interpolation_start_pose: NDArray[np.float32] | None = None
        self.face_lost_delay = 2.0  # seconds to wait before starting interpolation
        self.interpolation_duration = 1.0  # seconds to interpolate back to neutral

    def get_latest_frame(self) -> CameraFrame:
        """Get the latest frame (thread-safe)."""
        with self.frame_lock:
            # Return a copy in original BGR format (OpenCV native)
            return self.latest_frame.copy()

    def set_target_track_name(self, track_name: str) -> None:
        with self.face_tracking_lock:
            self.target_track_name = track_name

    def start(self) -> None:
        """Start the camera worker loop in a thread."""
        # FrameHub is the single camera reader.
        self.frame_hub.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        logger.debug("Camera worker stopped")

    def working_loop(self) -> None:
        """Enable the camera worker loop.

        Ported from main_works.py camera_worker() with same logic.
        """
        logger.debug("Starting camera working loop")

        # Initialize head tracker if available
        neutral_pose = np.eye(4)  # Neutral pose (identity matrix)

        while not self._stop_event.is_set():
            # Small sleep to prevent excessive CPU usage (same as main_works.py)
            time.sleep(0.2)  # 最大一秒5帧
            try:
                current_time = time.time()

                # self.reachy_mini.media.get_frame()
                # Get frame from FrameHub (single capture loop)
                frame = self.frame_hub.get_latest_frame()
                if frame is None:
                    continue

                # Handle face tracking if enabled and head tracker available
                face_positons = self.face_recognizer.get_face_positions(frame)
                track_lost = True
                face_tracking_offsets = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                if len(face_positons) > 0:
                    position = get_position_by_track_name(positions=face_positons, track_name=self.target_track_name)
                    if position:
                        track_lost = False
                        # Face detected - immediately switch to tracking
                        self.last_face_detected_time = current_time
                        self.interpolation_start_time = None  # Stop any interpolation

                        # Convert normalized coordinates to pixel coordinates
                        h, w, _ = frame.shape
                        eye_center_norm = (position.center + 1) / 2
                        eye_center_pixels = [
                            eye_center_norm[0] * w,
                            eye_center_norm[1] * h,
                        ]

                        # Get the head pose needed to look at the target, but don't perform movement
                        target_pose = self.reachy_mini.look_at_image(
                            eye_center_pixels[0],
                            eye_center_pixels[1],
                            duration=0.0,
                            perform_movement=False,
                        )

                        # Extract translation and rotation from the target pose directly
                        translation = target_pose[:3, 3]
                        rotation = R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False)

                        # Scale down translation and rotation because smaller FOV
                        translation *= 0.6
                        rotation *= 0.6

                        face_tracking_offsets = [
                            translation[0],
                            translation[1],
                            translation[2],  # x, y, z
                            rotation[0],
                            rotation[1],
                            rotation[2],  # roll, pitch, yaw
                        ]

                # No face detected while tracking enabled - set face lost timestamp
                elif self.last_face_detected_time is None or self.last_face_detected_time == current_time:
                    # Only update if we haven't already set a face lost time
                    # (current_time check prevents overriding the disable-triggered timestamp)
                    pass

                # Handle smooth interpolation (works for both face-lost and tracking-disabled cases)
                if self.last_face_detected_time is not None:
                    time_since_face_lost = current_time - self.last_face_detected_time

                    if time_since_face_lost >= self.face_lost_delay:
                        # Start interpolation if not already started
                        if self.interpolation_start_time is None:
                            self.interpolation_start_time = current_time
                            # Capture current pose as start of interpolation
                            with self.face_tracking_lock:
                                current_translation = face_tracking_offsets[:3]
                                current_rotation_euler = face_tracking_offsets[3:]
                                # Convert to 4x4 pose matrix
                                pose_matrix = np.eye(4, dtype=np.float32)
                                pose_matrix[:3, 3] = current_translation
                                pose_matrix[:3, :3] = R.from_euler(
                                    "xyz",
                                    current_rotation_euler,
                                ).as_matrix()
                                self.interpolation_start_pose = pose_matrix

                        # Calculate interpolation progress (t from 0 to 1)
                        elapsed_interpolation = current_time - self.interpolation_start_time
                        t = min(1.0, elapsed_interpolation / self.interpolation_duration)

                        # Interpolate between current pose and neutral pose
                        interpolated_pose = linear_pose_interpolation(
                            self.interpolation_start_pose,
                            neutral_pose,
                            t,
                        )

                        # Extract translation and rotation from interpolated pose
                        translation = interpolated_pose[:3, 3]
                        rotation = R.from_matrix(interpolated_pose[:3, :3]).as_euler("xyz", degrees=False)

                        # Thread-safe update of face tracking offsets
                        face_tracking_offsets = [
                            translation[0],
                            translation[1],
                            translation[2],  # x, y, z
                            rotation[0],
                            rotation[1],
                            rotation[2],  # roll, pitch, yaw
                        ]

                        # If interpolation is complete, reset timing
                        if t >= 1.0:
                            self.last_face_detected_time = None
                            self.interpolation_start_time = None
                            self.interpolation_start_pose = None
                    # else: Keep current offsets (within 2s delay period)

                # Thread-safe frame storage
                with self.frame_lock:
                    raw = frame.copy()  # BGR — 保持原始格式供人脸识别使用
                    # 颜色校正（显示/标注用 RGB）
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    annotated = draw_detections(frame_rgb, positions=face_positons)
                    self.latest_frame = CameraFrame(
                        face_tracking_offsets=face_tracking_offsets,
                        face_positons=face_positons,
                        track_name=self.target_track_name,
                        track_lost=track_lost,
                        image=annotated,
                        raw_image=raw,
                    )  # .copy()

            except Exception as e:
                logger.error(f"Camera worker error: {e}")
                time.sleep(0.1)  # Longer sleep on error

        logger.debug("Camera worker thread exited")


class CameraWorkerProvider(Provider[CameraWorker]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        mini = con.force_fetch(ReachyMini)
        frame_hub = con.force_fetch(FrameHub)

        ws = con.force_fetch(Workspace)
        face_recognizer_storage = ws.configs().sub_storage("face_recognizer")

        face_recognizer = FaceRecognizer(
            known_faces_storage=face_recognizer_storage,
        )
        return CameraWorker(reachy_mini=mini, frame_hub=frame_hub, face_recognizer=face_recognizer)
