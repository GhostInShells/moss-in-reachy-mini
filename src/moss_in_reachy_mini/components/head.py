import asyncio
import logging
from typing import Optional

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Container, IoCContainer, Provider, INSTANCE
from ghoshell_moss import PyChannel, Message, Text
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from moss_in_reachy_mini.moves.head_move import HeadMove, BreathingMove
from moss_in_reachy_mini.components.head_tracker import HeadTracker


class Head:
    def __init__(self, mini: ReachyMini, head_tracker: HeadTracker, container: IoCContainer=None):
        self.mini = mini
        self._container = Container(parent=container)
        self.logger = self._container.get(LoggerItf) or logging.getLogger("Head")

        self._head_tracker = head_tracker
        self._tracking_event = asyncio.Event()

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

    async def reset(self, duration: float = 0.5):
        """
        Reset the head, watching forward
        """
        self._tracking_event.clear()
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(),
            duration=duration,
        ))

    async def start_tracking_face(self, name: str):
        """
        启动人脸追踪，会在空闲时间一直看着指定name的用户。

        Args:
            name: 根据你的视觉输入信息，选择你能看到的name填写入参，不能用中文，不能使用你当前视觉里没有的name
        """
        if name == "unknown":
            raise ValueError("unknown表示当前用户未识别，不能作为追踪目标")
        self._tracking_event.set()
        self._head_tracker.enabled.set()
        self._head_tracker.set_target_track_name(name)

    async def stop_tracking_face(self):
        """
        停止人脸追踪
        """
        self._tracking_event.clear()
        self._head_tracker.enabled.clear()
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
        if self._tracking_event.is_set():
            msg.with_content(
                Text(text=f"You are keep looking user with head tracking"),
            )
            if self._head_tracker.latest_frame.track_name:
                msg.with_content(
                    Text(text=f"Current tracking {self._head_tracker.latest_frame.track_name}")
                )
            for pos in self._head_tracker.latest_frame.face_positons:
                if not pos.is_recognized:
                    continue
                msg.with_content(
                    Text(text=f"当前视觉里的人脸 track_id={pos.track_id} name={pos.name}")
                )

        msg.with_content(
            Text(text=f"Current head pose is {self.mini.get_current_head_pose()}")
        )

        return [msg]

    async def on_idle(self):
        self.logger.info("Head on-idle entering")
        try:
            if self._tracking_event.is_set():
                self._head_tracker.enabled.set()
            else:
                await self._breathing()
        except asyncio.CancelledError:
            self._head_tracker.enabled.clear()
            self.logger.info("Head on_idle task cancelled successfully")

    def as_channel(self) -> PyChannel:
        head = PyChannel(name="head", blocking=True)

        head.build.context_messages(self.context_messages)
        head.build.idle(self.on_idle)

        # move
        head.build.command()(self.move)
        head.build.command()(self.reset)
        head.build.command()(self.start_tracking_face)
        head.build.command()(self.stop_tracking_face)

        return head


class HeadProvider(Provider[Head]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        mini = con.force_fetch(ReachyMini)
        head_tracker = con.force_fetch(HeadTracker)
        head = Head(mini, head_tracker, con)
        return head
