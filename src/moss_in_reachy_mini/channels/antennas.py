import asyncio
import json
import logging
from typing import List, Dict, Optional

import numpy as np
from numpy import typing as npt

from reachy_mini import ReachyMini
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import time_trajectory

from ghoshell_moss import PyChannel, Message, Text


class AntennasMove(Move):

    def __init__(
            self,
            current_left: float,
            current_right: float,
            target_left: float = 0,
            target_right: float = 0,
            duration: float = 2.0,
    ):
        self.current_left = current_left
        self.current_right = current_right
        self.target_left = target_left
        self.target_right = target_right
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float) -> tuple[
        npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, float | None
    ]:
        interp_time = time_trajectory(t / self.duration)

        r_rad = np.deg2rad(self.current_right + (self.target_right - self.current_right) * interp_time)
        l_rad = np.deg2rad(self.current_left + (self.target_left - self.current_left) * interp_time)

        return None, np.array([r_rad, l_rad]), None

class Antennas:
    def __init__(self, mini: ReachyMini, logger: logging.Logger):
        self.mini = mini
        self.logger = logger

        self._flapping_task: Optional[asyncio.Task] = None
        self._flapping_event = asyncio.Event()
        self._flapping_event.set()
        self.idle_flapping_params: List[Dict] = [
            {"left": 30, "right": -30, "duration": 1.0},
            {"left": -10, "right": 10, "duration": 1.0},
        ]

    async def move(self, left: float=0, right: float=0, duration: float=2.0):
        """
        Move the antenna to the given position and duration.

        Args:
            left (float): X coordinate of the position. range(degree): [-180, +180]
            right (float): Y coordinate of the position. range(degree): [-180, +180]
            duration (float): Duration of the movement.
        """
        r_degree, l_degree = self._get_current_position()
        await self.mini.async_play_move(AntennasMove(
            current_left=l_degree,
            current_right=r_degree,
            target_left=left,
            target_right=right,
            duration=duration,
        ))

    async def reset(self, duration: float=2.0):
        """
        Reset the antenna to zero in duration.
        """
        r_degree, l_degree = self._get_current_position()
        await self.mini.async_play_move(AntennasMove(
            current_left=l_degree,
            current_right=r_degree,
            duration=duration,
        ))

    async def set_idle_flapping(self, text__: str):
        """
        Give a params to move antennas on idle state

        Args:
            text__ (str): using JSON to implement the cyclic reciprocation of antenna movements, like `[{"left": 30, "right": -30, "duration": 1.0}, {"left": -10, "right": 10, "duration": 1.0}]`
        """

        self.idle_flapping_params = json.loads(text__)

    async def enable_flapping(self, enable: bool=True):
        """
        Enable antenna move on idle state or not.
        """
        if enable:
            self._flapping_event.set()
        else:
            self._flapping_event.clear()

    def _get_current_position(self):
        r_rad, l_rad = self.mini.get_present_antenna_joint_positions()
        r_degree, l_degree = round(np.rad2deg(r_rad), 1), round(np.rad2deg(l_rad), 1)
        return r_degree, l_degree

    async def context_messages(self):
        msg = Message.new(role="user", name="__reachy_mini_antennas__")

        r_degree, l_degree = self._get_current_position()
        msg.with_content(
            Text(text=f"Current antennas right degree:{r_degree}, left degree: {l_degree}")
        )

        if self._flapping_event.is_set():
            msg.with_content(
                Text(text=f"You are flapping antenna on idle with move {self.idle_flapping_params}")
            )

        return [msg]

    async def _flapping(self):
        try:  # 捕获取消异常，确保任务优雅退出
            while self._flapping_event.is_set():
                for params in self.idle_flapping_params:
                    await self.move(**params)
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            self.logger.info("Flapping task was cancelled")
            raise  # 重新抛出，让外层await能捕获

    async def on_policy_run(self):
        self.logger.info("on_policy_run antennas")
        # await self.reset(duration=1.0)
        # 先取消旧任务（如果存在），避免多任务并发
        if self._flapping_task and not self._flapping_task.done():
            self._flapping_task.cancel()
            try:
                await self._flapping_task
            except asyncio.CancelledError:
                pass
        self._flapping_task = asyncio.create_task(self._flapping())

    async def on_policy_pause(self):
        self.logger.info("on_policy_pause antennas")
        # 1. 边界检查：任务存在且未完成时才取消
        if self._flapping_task and not self._flapping_task.done():
            self._flapping_task.cancel()
            try:
                # 2. 捕获取消异常，避免程序崩溃
                await self._flapping_task
            except asyncio.CancelledError:
                self.logger.info("Flapping task cancelled successfully")
            finally:
                self._flapping_task = None
        await self.reset(duration=0.5)

    def as_channel(self) -> PyChannel:
        antennas = PyChannel(name="antennas", description="This channel should only be used when the user explicitly and actively specifies an antenna-related command.", block=True)

        antennas.build.with_context_messages(self.context_messages)
        antennas.build.on_policy_run(self.on_policy_run)
        antennas.build.on_policy_pause(self.on_policy_pause)
        antennas.build.command()(self.move)
        antennas.build.command()(self.reset)
        antennas.build.command()(self.set_idle_flapping)
        antennas.build.command()(self.enable_flapping)

        return antennas