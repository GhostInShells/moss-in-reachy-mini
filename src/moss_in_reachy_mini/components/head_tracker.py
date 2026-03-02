import asyncio
import logging
import time
import numpy as np
from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, Container, IoCContainer, INSTANCE
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from framework.abcd.agent import EventBus
from framework.abcd.agent_event import VisionAgentEvent
from moss_in_reachy_mini.camera.camera_worker import CameraWorker
from moss_in_reachy_mini.camera.model import CameraFrame


class HeadTracker:

    def __init__(self, mini: ReachyMini, camera_worker: CameraWorker, container: IoCContainer=None):
        self._mini = mini
        self._container = Container(parent=container)
        self.logger = self._container.get(LoggerItf) or logging.getLogger("HeadTracker")
        self._camera_worker = camera_worker

        self.latest_frame: CameraFrame | None = None
        self.track_lost_start_at = 0
        self.track_lost_threshold = 3  # 3秒
        self._run_task = None

        self.enabled = asyncio.Event()
        self._quit = asyncio.Event()
        
        # Smoothing parameters
        self.min_movement_threshold = 0.05  # Minimum movement to trigger head move

    def set_target_track_name(self, track_name: str):
        self._camera_worker.set_target_track_name(track_name)

    async def run(self):
        while not self._quit.is_set():
            # Adjust loop interval to balance responsiveness and task stability
            loop_interval = 0.08  # 80ms interval - reduces task cancellation frequency
            await asyncio.sleep(loop_interval)
            if not self.enabled.is_set():
                continue

            self.latest_frame = self._camera_worker.get_latest_frame()
            # 有追踪的目标且已经丢失
            if self.latest_frame.track_name and self.latest_frame.track_lost:
                self.track_lost_start_at = time.time()
                # 人脸追踪丢失目标，给脑子发一个event
                if time.time() - self.track_lost_start_at > self.track_lost_threshold:
                    eventbus = self._container.get(EventBus)
                    if eventbus:
                        await eventbus.put(VisionAgentEvent(
                            content=f"人脸跟随的目标{self.latest_frame.track_name}已丢失超过{self.track_lost_threshold}秒，请重新根据下面你看到的画面进行下一步决策",
                            images=[self.latest_frame.to_base64_image()],
                            priority=-1,
                            issuer="HeadTracker",
                        ).to_bytes())
                continue

            self.track_lost_start_at = 0
            # Create target pose from tracking data
            target_pose = create_head_pose(
                x=self.latest_frame.face_tracking_offsets[0],
                y=self.latest_frame.face_tracking_offsets[1],
                z=self.latest_frame.face_tracking_offsets[2],
                roll=self.latest_frame.face_tracking_offsets[3],
                pitch=self.latest_frame.face_tracking_offsets[4],
                yaw=self.latest_frame.face_tracking_offsets[5],
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
