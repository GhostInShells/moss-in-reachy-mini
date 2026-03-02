import asyncio
import io
import logging
import time
from functools import partial
from typing import Optional, List

from PIL import Image
from ghoshell_common.contracts import LoggerItf, Workspace
from ghoshell_container import IoCContainer, Provider, INSTANCE, Container
from ghoshell_moss import PyChannel, Message, Base64Image, Text
from reachy_mini import ReachyMini

from framework.abcd.agent_hook import AgentHook, AgentHookState
from moss_in_reachy_mini.components.antennas import Antennas
from moss_in_reachy_mini.components.body import Body
from moss_in_reachy_mini.components.head import Head
from moss_in_reachy_mini.components.vision import Vision
from state import AsleepState, WakenState, BoringState, MiniStateHook, InitialState


class StateLog:
    def __init__(self, from_state: MiniStateHook, to_state: MiniStateHook):
        self.from_state = from_state
        self.to_state = to_state
        self.now = int(time.time())

class MossInReachyMini:
    def __init__(
            self,
            mini: ReachyMini,
            body: Body,
            head: Head,
            antennas: Antennas,
            vision: Vision,
            container: IoCContainer = None,
    ):
        self.mini = mini
        self.logger = container.get(LoggerItf) or logging.getLogger(__name__)
        self._ws = container.force_fetch(Workspace)
        self._container = Container(parent=container)

        self.body = body
        self.head = head
        self.antennas = antennas
        self.vision = vision

        # state
        self._state_map = {
            AsleepState.NAME: AsleepState(mini),
            WakenState.NAME: WakenState(
                mini,
                body=body,
                head=head,
                antennas=antennas,
                vision=vision,
                switch_to=self.switch_to,
                container=self._container,
            ),
            BoringState.NAME: BoringState(
                mini,
                body=body,
                vision=vision,
                switch_to=self.switch_to,
                container=self._container,
            )
        }
        self._state: MiniStateHook = InitialState()
        self._state_log: List[StateLog] = []

        self._bootstrapped = asyncio.Event()

    async def switch_to(self, state_name: str):
        f"""
        切换到指定状态，当前状态为{self._state.NAME}，可选状态有{', '.join([s.NAME for s in self._state_map.values()])}
        
        :param state_name: 目标状态名称
        """
        state_name = state_name.lower()
        if state_name not in self._state_map:
            raise ValueError(f'Invalid state name: {state_name}')

        if self._state:
            await self._state.on_self_exit()

        self.logger.info(f'Switching state from {self._state.NAME} to {state_name}')
        self._state_log.append(StateLog(self._state, self._state_map[state_name]))  # 记录状态切换
        self._state = self._state_map[state_name]
        await self._state.on_self_enter()

    # 交给MainAgent来控制生命周期
    def get_hook(self) -> AgentHook:
        return self._state

    async def context_messages(self):
        msg = Message.new(role="user", name="__reachy_mini__")
        appearance_img = Image.open(io.BytesIO(self._ws.assets().get("appearance.png")))
        structure_img = Image.open(io.BytesIO(self._ws.assets().get("structure.png")))
        msg.with_content(
            Text(text="These two images shows your appearance and structure"),
            Base64Image.from_pil_image(appearance_img),
            Base64Image.from_pil_image(structure_img),
        )

        contents = []
        now = int(time.time())
        for state in self._state_log:
            ago = now - state.now
            if not state.from_state:
                text = f"Start state to {state.to_state.NAME} occurred {ago} seconds ago"
            else:
                text = f"Switch state from {state.from_state.NAME} to {state.to_state.NAME} occurred {ago} seconds ago"
            contents.append(Text(text=text))
        self._state_log.clear()

        msg.with_content(*contents)
        return [msg]

    def as_channel(self) -> PyChannel:
        self.logger.info("MossInReachyMini.as_channel()...")
        assert self._bootstrapped.is_set()

        reachy_mini = PyChannel(name="reachy_mini", block=True)
        reachy_mini.build.command()(self.switch_to)
        reachy_mini.build.with_context_messages(self.context_messages)

        # asleep state
        asleep_chan = self._state_map[AsleepState.NAME].as_channel()
        asleep_chan.build.with_available()(lambda: self._state.NAME == AsleepState.NAME)

        # waken state
        waken_chan = self._state_map[WakenState.NAME].as_channel()
        waken_chan.build.with_available()(lambda: self._state.NAME == WakenState.NAME)

        # boring state
        boring_chan = self._state_map[BoringState.NAME].as_channel()
        boring_chan.build.with_available()(lambda: self._state.NAME == BoringState.NAME)

        reachy_mini.import_channels(
            asleep_chan,
            waken_chan,
            boring_chan,
        )

        return reachy_mini

    async def bootstrap(self):
        await self.head.bootstrap()
        await self.switch_to(AsleepState.NAME)
        self._bootstrapped.set()

    async def __aenter__(self):
        await self.bootstrap()
        return self

    async def aclose(self):
        await self.switch_to(AsleepState.NAME)
        await self.head.aclose()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()


class MossInReachyMiniProvider(Provider[MossInReachyMini]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        mini = con.force_fetch(ReachyMini)
        body = con.force_fetch(Body)
        head = con.force_fetch(Head)
        vision = con.force_fetch(Vision)
        antennas = con.force_fetch(Antennas)
        return MossInReachyMini(mini, body, head, antennas, vision, container=con)
