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

        self._breathing_task: Optional[asyncio.Task] = None
        self._breathing_event = asyncio.Event()
        self._breathing_event.set()

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
            return "unknown表示当前用户未识别，不能作为追踪目标"
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

    async def start_breathing(self):
        self._breathing_event.set()

    async def stop_breathing(self):
        self.mini.set_target_body_yaw(0.0)
        self._breathing_event.clear()

    async def _breathing(self):
        try:  # 捕获取消异常，确保任务优雅退出
            if self._breathing_event.is_set():
                _, current_antennas = self.mini.get_current_joint_positions()
                current_head_pose = self.mini.get_current_head_pose()
                breathing_move = BreathingMove(
                    interpolation_start_pose=current_head_pose,
                    interpolation_start_antennas=current_antennas,
                    interpolation_duration=1.0,
                )
                await self.mini.async_play_move(breathing_move)
        except asyncio.CancelledError:
            self.logger.info("Breathing task was cancelled")
            raise  # 重新抛出，让外层await能捕获

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

    async def on_policy_run(self):
        self.logger.info(f"Running Head on-policy run")

        if self._tracking_event.is_set():
            self._head_tracker.enabled.set()
        else:
            if not self._breathing_event.is_set():
                return
            # 先取消旧任务（如果存在），避免多任务并发
            if self._breathing_task and not self._breathing_task.done():
                self._breathing_task.cancel()
                try:
                    await self._breathing_task
                except asyncio.CancelledError:
                    pass
            self._breathing_task = asyncio.create_task(self._breathing())

    async def on_policy_pause(self):
        self.logger.info("Running Head on-policy pause")
        self._head_tracker.enabled.clear()

        # 1. 边界检查：任务存在且未完成时才取消
        if self._breathing_task and not self._breathing_task.done():
            self._breathing_task.cancel()
            try:
                # 2. 捕获取消异常，避免程序崩溃
                await self._breathing_task
            except asyncio.CancelledError:
                self.logger.info("Breathing task cancelled successfully")
            finally:
                self._breathing_task = None

    def as_channel(self) -> PyChannel:
        head = PyChannel(name="head", blocking=True)

        head.build.context_messages(self.context_messages)
        head.build.on_policy_run(self.on_policy_run)
        head.build.on_policy_pause(self.on_policy_pause)

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
