import asyncio
import logging
from typing import List

import numpy as np
from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, Container, IoCContainer, INSTANCE
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from moss_in_reachy_mini.vision.camera_worker import CameraWorker
from moss_in_reachy_mini.vision.yolo.head_detector import HeadDetector
from moss_in_reachy_mini.vision.yolo.model import Position


class HeadTracker:

    def __init__(self, mini: ReachyMini, camera_worker: CameraWorker, container: IoCContainer=None):
        self._mini = mini
        self._container = Container(parent=container)
        self.logger = self._container.get(LoggerItf) or logging.getLogger("HeadTracker")
        self._camera_worker = camera_worker

        self.face_tracking_offsets: List[float] = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self.face_tracking_positions: List[Position] = []
        self.current_tracking_id = -1
        self._run_task = None

        self.enabled = asyncio.Event()
        self._quit = asyncio.Event()
        
        # Smoothing parameters
        self.min_movement_threshold = 0.05  # Minimum movement to trigger head move

    def set_tracking_id(self, tracking_id: int):
        self._camera_worker.set_tracking_id(tracking_id)

    async def run(self):
        while not self._quit.is_set():
            # Adjust loop interval to balance responsiveness and task stability
            loop_interval = 0.08  # 80ms interval - reduces task cancellation frequency
            await asyncio.sleep(loop_interval)
            if not self.enabled.is_set():
                continue

            self.face_tracking_offsets, self.face_tracking_positions, self.current_tracking_id = self._camera_worker.get_face_tracking_data()

            # Create target pose from tracking data
            target_pose = create_head_pose(
                x=self.face_tracking_offsets[0],
                y=self.face_tracking_offsets[1],
                z=self.face_tracking_offsets[2],
                roll=self.face_tracking_offsets[3],
                pitch=self.face_tracking_offsets[4],
                yaw=self.face_tracking_offsets[5],
                degrees=False,
                mm=False,
            )

            current_head_pose = self._mini.get_current_head_pose()
            
            # Check if movement is needed
            movement_magnitude = np.linalg.norm(current_head_pose - target_pose)
            if movement_magnitude < self.min_movement_threshold:
                continue

            self._mini.set_target(head=target_pose)

    async def start(self):
        self._camera_worker.start()
        self._run_task = asyncio.create_task(self.run())

    async def stop(self):
        self.enabled.clear()
        self._quit.set()

        if self._run_task:
            await self._run_task


class HeadTrackerProvider(Provider[HeadTracker]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        mini = con.force_fetch(ReachyMini)
        camera_worker = con.force_fetch(CameraWorker)
        return HeadTracker(mini, camera_worker, con)
