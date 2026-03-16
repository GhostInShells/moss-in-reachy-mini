import asyncio
import enum
import logging
from typing import Optional

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Container, IoCContainer, Provider, INSTANCE
from ghoshell_moss import PyChannel, Message, Text
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from moss_in_reachy_mini.camera.camera_worker import CameraWorker
from moss_in_reachy_mini.moves.head_move import HeadMove, BreathingMove
from moss_in_reachy_mini.components.head_tracker import HeadTracker


class IdleMode(enum.Enum):
    hold = "hold"
    breathing = "breathing"
    head_tracking = "head_tracking"


class Head:
    def __init__(
            self,
            mini: ReachyMini,
            head_tracker: HeadTracker,
            camera_worker: CameraWorker,
            logger: LoggerItf=None,
    ):
        self.mini = mini
        self.logger = logger or logging.getLogger("Head")

        self._camera_worker = camera_worker
        self._head_tracker = head_tracker

        self._idle_mode: str = IdleMode.hold.value
        self._last_idle_mode: str = IdleMode.hold.value  # 记录上一个空闲模式，方便恢复

    async def move(
            self,
            x: float = 0,
            y: float = 0,
            z: float = 0,
            roll: float = 0,
            pitch: float = 0,
            yaw: float = 0,
            duration: float = 0.5,
    ):
        """Move to a pose in 6D space (position and orientation).

        Args:
            x (float): X coordinate of the position. range: [-1.5cm, +2.5cm]
            y (float): Y coordinate of the position. range: [-4cm, +4cm]
            z (float): Z coordinate of the position. range: [-4cm, +2.5cm]
            roll (float): Roll angle. range(degree): [-40, +40]
            pitch (float): Pitch angle. range(degree): [-40, +40]
            yaw (float): Yaw angle. range(degree): [-60, +60]
            duration (float): Duration in seconds.
        """
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(x, y, z, roll, pitch, yaw),
            duration=duration
        ))


    def switch_idle_mode(self, mode: str):
        self._last_idle_mode = self._idle_mode
        self._idle_mode = mode

    async def reset(self, duration: float = 0.5, idle_mode: str = IdleMode.hold.value):
        """
        Reset the head, watching forward

        :param duration: 重置时间，单位秒
        :param idle_mode: 重置后保持的空闲模式，可选："hold"（空闲时静止）、"breathing"（空闲时呼吸）、"head_tracking"（空闲时人脸追踪）
        """
        self.switch_idle_mode(idle_mode)
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(),
            duration=duration,
        ))

    async def start_tracking_face(self, name: str):
        """
        启动人脸追踪，会在空闲时间一直看着指定name的用户。

        Args:
            name: 根据你的视觉输入信息，选择你能看到的已注册name填写入参，只能跟随已注册的人脸
        """
        if name == "unknown":
            raise ValueError("unknown表示当前用户未识别，不能作为追踪目标")
        known_faces = self._camera_worker.face_recognizer.known_faces
        if name not in known_faces:
            raise ValueError(f"'{name}'不在已知人脸库中，请先进行人脸录入后再开启跟随。")
        self.switch_idle_mode(IdleMode.head_tracking.value)
        self._head_tracker.set_target_track_name(name)

    async def stop_tracking_face(self):
        """
        停止人脸追踪
        """
        self.switch_idle_mode(self._last_idle_mode)
        self._head_tracker.set_target_track_name("")

    async def _breathing(self):
        _, current_antennas = self.mini.get_current_joint_positions()
        current_head_pose = self.mini.get_current_head_pose()
        breathing_move = BreathingMove(
            interpolation_start_pose=current_head_pose,
            interpolation_start_antennas=current_antennas,
            interpolation_duration=1.0,
        )
        await self.mini.async_play_move(breathing_move)

    async def context_messages(self):
        msg = Message.new(role="user", name="__reachy_mini_head__")

        # 始终从 camera_worker 获取最新帧，确保 LLM 知道当前画面中的人
        frame = self._camera_worker.get_latest_frame()
        recognized_names = [
            pos.name for pos in frame.face_positons
            if pos.is_recognized and pos.name
        ]
        if recognized_names:
            msg.with_content(
                Text(text=f"当前视觉中识别到的已注册用户: {', '.join(recognized_names)}")
            )

        if self._idle_mode == IdleMode.head_tracking.value:
            msg.with_content(
                Text(text="You are keep looking user with head tracking"),
            )
            if frame.track_name:
                msg.with_content(
                    Text(text=f"Current tracking {frame.track_name}")
                )

        msg.with_content(
            Text(text=f"Current head pose is {self.mini.get_current_head_pose()}")
        )

        return [msg]

    async def on_idle(self):
        self.logger.info("Head on-idle entering")
        try:
            if self._idle_mode == IdleMode.hold.value:
                await asyncio.Future()  # 挂起一个空事件，等待被cancel
            if self._idle_mode == IdleMode.head_tracking.value:
                await self._head_tracker.run()
            if self._idle_mode == IdleMode.breathing.value:
                await self._breathing()
        except asyncio.CancelledError:
            self.logger.info("Head on_idle task cancelled successfully")


class HeadProvider(Provider[Head]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        mini = con.force_fetch(ReachyMini)
        head_tracker = con.force_fetch(HeadTracker)
        camera_worker = con.force_fetch(CameraWorker)
        logger = con.get(LoggerItf)
        return Head(mini, head_tracker, camera_worker, logger)
