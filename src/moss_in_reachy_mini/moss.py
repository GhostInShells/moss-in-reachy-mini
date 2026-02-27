import asyncio
import io
import logging
from PIL import Image
from ghoshell_common.contracts import LoggerItf, Workspace
from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss import PyChannel, Message, Base64Image, Text
from reachy_mini import ReachyMini

from framework.abcd.agent_hook import AgentHook
from moss_in_reachy_mini.components.antennas import Antennas
from moss_in_reachy_mini.components.body import Body
from moss_in_reachy_mini.components.head import Head
from moss_in_reachy_mini.components.vision import Vision
from state import AsleepState, WakenState, BoringState, StateManagerHook


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

        self.body = body
        self.head = head
        self.antennas = antennas
        self.vision = vision

        self._state_manager = StateManagerHook(
            mini=self.mini,
            body=self.body,
            head=self.head,
            antennas=self.antennas,
            vision=self.vision,
            logger=self.logger,
        )

        self._bootstrapped = asyncio.Event()

    def hook(self) -> AgentHook:
        return self._state_manager

    async def wake_up(self):
        await self._state_manager.switch_to(WakenState.NAME)

    async def goto_sleep(self):
        await self._state_manager.switch_to(AsleepState.NAME)

    async def context_messages(self):
        msg = Message.new(role="user", name="__reachy_mini__")
        appearance_img = Image.open(io.BytesIO(self._ws.assets().get("appearance.png")))
        structure_img = Image.open(io.BytesIO(self._ws.assets().get("structure.png")))
        msg.with_content(
            Text(text="These two images shows your appearance and structure"),
            Base64Image.from_pil_image(appearance_img),
            Base64Image.from_pil_image(structure_img),
            Text(text=f"Your current state is {self._state_manager.current().NAME}"),
        )

        if self._state_manager.current().NAME == AsleepState.NAME:
            msg.with_content(
                Text(text="You must wake up first"),
            )

        contents = self._state_manager.to_contents()
        self._state_manager.clear_state_log()

        msg.with_content(*contents)
        return [msg]

    async def vision_context_messages(self):
        base_msg = await self.context_messages()
        msg = await self.vision.context_messages()
        return base_msg + msg

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
        asleep_chan.build.with_available()(lambda: self._state_manager.current().NAME == AsleepState.NAME)
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
        waken_chan.build.command()(self.vision.look)
        waken_chan.build.with_available()(lambda: self._state_manager.current().NAME == WakenState.NAME)

        # boring state can see
        boring_chan = PyChannel(name=BoringState.NAME, description=f"current state is boring", block=True)
        boring_chan.build.command(doc=self.body.emotion_docstring)(self.body.emotion)
        boring_chan.build.command()(self.goto_sleep)
        boring_chan.build.command()(self.vision.look)
        boring_chan.build.with_context_messages(self.context_messages)
        boring_chan.build.with_available()(lambda: self._state_manager.current().NAME == BoringState.NAME)

        reachy_mini.import_channels(
            asleep_chan,
            waken_chan,
            boring_chan,
        )

        return reachy_mini

    async def bootstrap(self):
        await self.head.bootstrap()
        await self._state_manager.start()
        self._bootstrapped.set()

    async def __aenter__(self):
        await self.bootstrap()
        return self

    async def aclose(self):
        await self.head.aclose()
        await self._state_manager.close()

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
