import asyncio
import logging
import time
import numpy as np
from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, Container, IoCContainer, INSTANCE
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import linear_pose_interpolation, delta_angle_between_mat_rot

from framework.abcd.agent import EventBus
from framework.abcd.agent_event import VisionAgentEvent
from moss_in_reachy_mini.camera.camera_worker import CameraWorker
from moss_in_reachy_mini.camera.model import CameraFrame

from scipy.spatial.transform import Rotation as R


class HeadTracker:

    def __init__(self, mini: ReachyMini, camera_worker: CameraWorker, container: IoCContainer=None):
        self._mini = mini
        self._container = Container(parent=container)
        self.logger = self._container.get(LoggerItf) or logging.getLogger("HeadTracker")
        self._camera_worker = camera_worker

        self.latest_frame: CameraFrame | None = None
        self.track_lost_start_at = 0
        self.track_lost_threshold = 10
        self._run_task = None

        self.enabled = asyncio.Event()
        self._quit = asyncio.Event()
        
        # Smoothing parameters
        self.min_movement_threshold = 0.05  # Minimum movement to trigger head move

        # Smoothing parameters
        self.smoothing_alpha = 0.3  # Exponential smoothing factor (0.1-0.3 for smooth tracking)
        self.max_movement_per_frame = 0.4  # Maximum movement per frame to prevent jerky motion
        self.prev_face_position = None  # Previous face position for smoothing

        self.loop_interval = 0.02   # 50帧/秒

    def set_target_track_name(self, track_name: str):
        if not track_name.isalpha() or track_name == "unknown":
            return
        self._camera_worker.set_target_track_name(track_name)

    async def run(self):
        while not self._quit.is_set():
            # Adjust loop interval to balance responsiveness and task stability
            await asyncio.sleep(self.loop_interval)
            if not self.enabled.is_set():
                continue

            self.latest_frame = self._camera_worker.get_latest_frame()
            if self.latest_frame.image is None:
                continue
            # 有追踪的目标且已经丢失
            if self.latest_frame.track_name and self.latest_frame.track_lost:
                if self.track_lost_start_at == 0:
                    self.track_lost_start_at = time.time()
                # 人脸追踪丢失目标，给脑子发一个event
                if time.time() - self.track_lost_start_at > self.track_lost_threshold:
                    # 主动关闭人脸跟随，等待大脑决策是否重新开启
                    self.enabled.clear()
                    self.set_target_track_name("")
                    # 重置时间
                    self.track_lost_start_at = 0
                    eventbus = self._container.get(EventBus)
                    if eventbus:
                        await eventbus.put(VisionAgentEvent(
                            content=f"人脸跟随的目标{self.latest_frame.track_name}已丢失超过{self.track_lost_threshold}秒，请重新根据你的最新视觉进行下一步决策",
                            # images=[self.latest_frame.to_base64_image()],
                            priority=-1,
                            issuer="HeadTracker",
                        ).to_agent_event())
                continue

            self.track_lost_start_at = 0

            # Create target pose from tracking data
            current_head_pose = self._mini.get_current_head_pose()
            current_pos = current_head_pose[:3, 3]
            current_rot = R.from_matrix(current_head_pose[:3, :3]).as_euler('xyz')

            # 目标偏移（假设 face_tracking_offsets 是绝对目标位置+角度）
            target_pos = self.latest_frame.face_tracking_offsets[:3]
            target_rot = self.latest_frame.face_tracking_offsets[3:]

            # Check if movement is needed
            target_pose = create_head_pose(
                x=target_pos[0], y=target_pos[1], z=target_pos[2],
                roll=target_rot[0], pitch=target_rot[1], yaw=target_rot[2],
                degrees=False,
                mm=False,
            )
            movement_magnitude = np.linalg.norm(current_head_pose - target_pose)
            if movement_magnitude < self.min_movement_threshold:
                continue

            # 平滑滤波
            smooth_filter = SmoothFilter(alpha_pos=0.2, alpha_rot=0.25, dt=self.loop_interval)
            smooth_filter.update(current_pos, current_rot)

            target_pos, target_rot = smooth_filter.update(np.asarray(target_pos, dtype=float), np.asarray(target_rot, dtype=float))

            target_pose = create_head_pose(
                x=target_pos[0], y=target_pos[1], z=target_pos[2],
                roll=target_rot[0], pitch=target_rot[1], yaw=target_rot[2],
                degrees=False,
                mm=False,
            )

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


class SmoothFilter:
    def __init__(self, alpha_pos, alpha_rot, dt):
        self.alpha_pos = alpha_pos
        self.alpha_rot = alpha_rot
        self.dt = dt
        self.pos = None
        self.rot = None
        self.pos_vel = np.zeros(3)
        self.rot_vel = np.zeros(3)

    def update(self, target_pos, target_rot):
        if self.pos is None:
            self.pos = target_pos.copy()
            self.rot = target_rot.copy()
            return self.pos, self.rot

        # 计算误差
        pos_error = target_pos - self.pos
        rot_error = self._angle_diff(target_rot, self.rot)

        # 更新速度（使用低通滤波估计速度）
        self.pos_vel = (1 - self.alpha_pos) * self.pos_vel + self.alpha_pos * (pos_error / self.dt)
        self.rot_vel = (1 - self.alpha_rot) * self.rot_vel + self.alpha_rot * (rot_error / self.dt)

        # 预测下一时刻位置（外推）
        pred_pos = self.pos + self.pos_vel * self.dt
        pred_rot = self.rot + self.rot_vel * self.dt

        # 对预测结果再进行一次低通滤波（平滑）
        self.pos = (1 - self.alpha_pos) * pred_pos + self.alpha_pos * target_pos
        self.rot = (1 - self.alpha_rot) * pred_rot + self.alpha_rot * target_rot
        self.rot = self._normalize_angles(self.rot)

        return self.pos, self.rot

    def _angle_diff(self, target, current):
        """计算带环绕处理的角度差"""
        diff = target - current
        diff = (diff + np.pi) % (2 * np.pi) - np.pi  # 映射到 [-π, π]
        return diff

    def _normalize_angles(self, angles):
        """将角度归一化到 [-π, π]"""
        return (angles + np.pi) % (2 * np.pi) - np.pi