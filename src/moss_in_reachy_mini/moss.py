import asyncio
import io
import logging
import os.path
import pathlib
import time
from functools import partial
from typing import Optional, List

import cv2
from PIL import Image
from ghoshell_common.contracts import LoggerItf, Workspace
from ghoshell_container import IoCContainer
from ghoshell_moss import PyChannel, Message, Base64Image, Text
from reachy_mini import ReachyMini

from moss_in_reachy_mini.channels.antennas import Antennas
from moss_in_reachy_mini.channels.body import Body
from moss_in_reachy_mini.channels.head import Head
from state import AsleepState, BaseState, WakenState, BoringState

class StateLog:
    def __init__(self, from_state: BaseState, to_state: BaseState):
        self.from_state = from_state
        self.to_state = to_state
        self.now = int(time.time())

    def to_message_content(self):
        ago = int(time.time()) - self.now

        if not self.from_state:
            return f"Start state to {self.to_state.NAME} occurred {ago} seconds ago"

        return f"Switch state from {self.from_state.NAME} to {self.to_state.NAME} occurred {ago} seconds ago"

class MossInReachyMini:
    def __init__(self, mini: ReachyMini, container: IoCContainer=None):
        self.mini = mini
        self.logger = container.get(LoggerItf) or logging.getLogger(__name__)
        self._ws = container.force_fetch(Workspace)

        self.body = Body(mini, container)
        self.head = Head(mini, self.logger, container)
        self.antennas = Antennas(mini, self.logger)

        self._state_map = {
            AsleepState.NAME: AsleepState(mini),
            WakenState.NAME: WakenState(
                mini,
                head=self.head,
                antennas=self.antennas,
                turn_to_boring=partial(self._switch_state_to, BoringState.NAME),
            ),
            BoringState.NAME: BoringState(
                mini,
                body=self.body,
                turn_to_asleep=partial(self._switch_state_to, AsleepState.NAME),
                back_to_waken=partial(self._switch_state_to, WakenState.NAME),
            )
        }
        self._state: Optional[BaseState] = None
        self._state_log: List[StateLog] = []
        self._bootstrapped = asyncio.Event()

    @property
    def state(self) -> BaseState:
        return self._state

    def set_proactive_input(self, handle_input):
        self._state_map[WakenState.NAME].set_proactive_input(handle_input)

    async def _switch_state_to(self, state_name: str):
        if state_name not in self._state_map:
            raise ValueError(f'Invalid state name: {state_name}')

        if self._state:
            await self._state.on_exit()

        self.logger.info(f'Switching state from {self._state.NAME if self._state else "initial"} to {state_name}')
        self._state_log.append(StateLog(self._state, self._state_map[state_name]))  # 记录状态切换
        self._state = self._state_map[state_name]
        await self._state.on_enter()

    async def wake_up(self):
        await self._switch_state_to(WakenState.NAME)

    async def goto_sleep(self):
        await self._switch_state_to(AsleepState.NAME)

    async def context_messages(self):
        msg = Message.new(role="user", name="__reachy_mini__")
        appearance_img = Image.open(io.BytesIO(self._ws.assets().get("appearance.png")))
        structure_img = Image.open(io.BytesIO(self._ws.assets().get("structure.png")))
        msg.with_content(
            Text(text="These two images shows your appearance and structure"),
            Base64Image.from_pil_image(appearance_img),
            Base64Image.from_pil_image(structure_img),
            Text(text=f"Your current state is {self._state.NAME}"),
        )

        for state in self._state_log:
            msg.with_content(Text(text=state.to_message_content()))

        self._state_log.clear()
        return [msg]

    async def vision_context_messages(self):
        msg = Message.new(role="user", name="__reachy_mini_vision__")
        # mini vision
        frame = self.mini.media.get_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB
            img_pil = Image.fromarray(frame_rgb)
            # img_pil.save("temp.png")
            msg.with_content(
                Text(text="This image is what you see")
            ).with_content(
                Base64Image.from_pil_image(img_pil)
            )
        else:
            msg.with_content(
                Text(text="No vision available")
            )

        base_msg = await self.context_messages()
        return base_msg + [msg]

    async def integrated_context_messages(self):
        vision_with_base_msg = await self.vision_context_messages()
        head_msg = await self.head.context_messages()
        antenna_msg = await self.antennas.context_messages()
        return vision_with_base_msg + head_msg + antenna_msg

    def as_channel(self) -> PyChannel:
        self.logger.info("MossInReachyMini.as_channel()...")
        assert self._bootstrapped.is_set()

        reachy_mini = PyChannel(name="reachy_mini", block=True)

        # asleep state can see
        asleep_chan = PyChannel(name=AsleepState.NAME, description=f"current state is asleep", block=True)
        asleep_chan.build.command()(self.wake_up)
        asleep_chan.build.with_available()(lambda: self._state.NAME == AsleepState.NAME)
        asleep_chan.build.with_context_messages(self.context_messages)

        # waken state can see
        waken_chan = PyChannel(name=WakenState.NAME, description=f"current state is waken", block=True)
        waken_chan.build.command()(self.goto_sleep)
        waken_chan.build.with_context_messages(self.integrated_context_messages)
        waken_chan.build.command(doc=self.body.dance_docstring)(self.body.dance)
        waken_chan.build.command(doc=self.body.emotion_docstring)(self.body.emotion)
        waken_chan.build.command(name="head_move")(self.head.move)
        waken_chan.build.command(name="head_reset")(self.head.reset)
        waken_chan.build.command()(self.head.start_tracking_face)
        waken_chan.build.command()(self.head.stop_tracking_face)
        waken_chan.build.command()(self.head.start_breathing)
        waken_chan.build.command()(self.head.stop_breathing)
        waken_chan.build.command(name="antennas_move")(self.antennas.move)
        waken_chan.build.command(name="antennas_reset")(self.antennas.reset)
        waken_chan.build.command()(self.antennas.set_idle_flapping)
        waken_chan.build.command()(self.antennas.enable_flapping)
        waken_chan.build.with_available()(lambda: self._state.NAME == WakenState.NAME)

        # boring state can see
        boring_chan = PyChannel(name=BoringState.NAME, description=f"current state is boring", block=True)
        boring_chan.build.command(doc=self.body.emotion_docstring)(self.body.emotion)
        boring_chan.build.command()(self.goto_sleep)
        boring_chan.build.with_context_messages(self.vision_context_messages)
        boring_chan.build.with_available()(lambda: self._state.NAME == BoringState.NAME)

        reachy_mini.import_channels(
            asleep_chan,
            waken_chan,
            boring_chan,
        )

        return reachy_mini

    async def bootstrap(self):
        await self._switch_state_to(AsleepState.NAME)
        await self.head.bootstrap()
        self._bootstrapped.set()

    async def __aenter__(self):
        await self.bootstrap()
        return self

    async def aclose(self):
        await self.head.aclose()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
