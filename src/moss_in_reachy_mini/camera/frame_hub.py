import logging
import threading
import time
from typing import Optional

import numpy as np
from ghoshell_container import IoCContainer, Provider
from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)


class FrameHub:
    """Single camera capture loop shared by multiple consumers.

    Why:
    - Avoid concurrent calls to `mini.media.get_frame()` from multiple threads.
    - Keep CameraWorker/HeadTracker and VideoRecorder independent but stable.
    """

    def __init__(self, mini: ReachyMini, *, fps: float = 25.0):
        self._mini = mini
        self._interval = 1.0 / max(fps, 1.0)

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.debug("FrameHub started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.debug("FrameHub stopped")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self._interval)
            try:
                frame = self._mini.media.get_frame()
                if frame is None:
                    continue
                with self._lock:
                    self._latest_frame = frame
            except RuntimeError as e:
                # Common when camera is not opened yet; keep looping quietly.
                logger.debug("FrameHub camera not ready: %s", e)
            except Exception:
                logger.exception("FrameHub error")
                time.sleep(0.1)


class FrameHubProvider(Provider[FrameHub]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> FrameHub:
        mini = con.force_fetch(ReachyMini)
        return FrameHub(mini)
