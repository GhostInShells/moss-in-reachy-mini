import asyncio
import io
import logging
import os
import time
from typing import List

from PIL import Image
from ghoshell_common.contracts import LoggerItf, Workspace
from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss import PyChannel, Message, Base64Image, Text
from reachy_mini import ReachyMini

from framework.abcd.agent_hook import AgentHook
from moss_in_reachy_mini.state.abcd import MiniStateHook, InitialState
from moss_in_reachy_mini.state.asleep import AsleepState
from moss_in_reachy_mini.state.boring import BoringState
from moss_in_reachy_mini.state.waken import WakenState

try:
    from moss_in_reachy_mini.state.live import LiveState
except Exception:  # optional dependency (douyin_live extras)
    LiveState = None  # type: ignore[assignment]

from moss_in_reachy_mini.video.recorder_channel import VideoRecorder
from moss_in_reachy_mini.video.recorder_worker import VideoRecorderWorker


class StateLog:
    def __init__(self, from_state: MiniStateHook, to_state: MiniStateHook):
        self.from_state = from_state
        self.to_state = to_state
        self.now = int(time.time())

class MossInReachyMini:
    def __init__(
            self,
            mini: ReachyMini,
            *states: MiniStateHook,
            default_state: str = AsleepState.NAME,
            appearance_img: Image.Image,
            structure_img: Image.Image,
            logger: LoggerItf = None,
            recorder: VideoRecorderWorker | None = None,
    ):
        self.mini = mini
        self.logger = logger or logging.getLogger(__name__)

        # state
        self._state_map = { state.NAME: state for state in states }
        self._state: MiniStateHook = InitialState()
        self._state_log: List[StateLog] = []
        self._default_state = default_state

        # img
        self.appearance_img = appearance_img
        self.structure_img = structure_img

        self._recorder = recorder

        self._bootstrapped = asyncio.Event()

    async def switch_state(self, state_name: str, force: bool = False):
        state_name = state_name.lower()
        if state_name not in self._state_map:
            raise ValueError(f'Invalid state name: {state_name}')

        if state_name == self._state.NAME:
            return

        if not self._state.out_switchable and not force:
            raise ValueError(f'Current state {self._state.NAME} is not out switchable')

        if not self._state_map[state_name].in_switchable and not force:
            raise ValueError(f'Current state {state_name} is not in switchable')

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
        msg.with_content(
            Text(text="These two images shows your appearance and structure"),
            Base64Image.from_pil_image(self.appearance_img),
            Base64Image.from_pil_image(self.structure_img),
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

    def as_channel(self, only_context_messages=False) -> PyChannel:
        self.logger.info("MossInReachyMini.as_channel()...")

        reachy_mini = PyChannel(name="reachy_mini", description="reachy mini root channel", blocking=True)

        if only_context_messages:
            reachy_mini.build.context_messages(self.context_messages)
            return reachy_mini

        reachy_mini.build.command(doc=f"""
        切换到指定状态，当前状态为{self._state.NAME}，可选状态有{', '.join([s.NAME for s in self._state_map.values()])}

        :param state_name: 目标状态名称
        :param force: 务必使用默认值False，任何情况都不能设置为True
        """)(self.switch_state)


        channels = []
        for name, state in self._state_map.items():
            chan = state.as_channel()
            chan.build.available(lambda _state=state: self._state.NAME == _state.NAME)
            channels.append(chan)

        if self._recorder is not None:
            recorder_chan = VideoRecorder(self._recorder).as_channel()
            # recorder_chan.build.with_available()(lambda: self._state.NAME != AsleepState.NAME)
            recorder_chan.build.available(lambda: True)
            channels.append(recorder_chan)

        reachy_mini.import_channels(
            *channels,
        )

        reachy_mini.build.start_up(self.bootstrap)
        reachy_mini.build.close(self.aclose)

        return reachy_mini

    async def bootstrap(self):
        self.mini.__enter__()
        await self.switch_state(self._default_state)
        self._bootstrapped.set()

    async def aclose(self):
        self.mini.__exit__(None, None, None)
        await self.switch_state(AsleepState.NAME)


class MossInReachyMiniProvider(Provider[MossInReachyMini]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        mini = con.force_fetch(ReachyMini)
        asleep = con.force_fetch(AsleepState)
        waken = con.force_fetch(WakenState)
        boring = con.force_fetch(BoringState)
        live = None
        if LiveState is not None:
            try:
                live = con.force_fetch(LiveState)
            except Exception as e:
                live = None
        logger = con.get(LoggerItf)

        try:
            recorder = con.get(VideoRecorderWorker)
        except Exception:
            recorder = None

        ws = con.force_fetch(Workspace)
        appearance_img = Image.open(io.BytesIO(ws.assets().get("appearance.png")))
        structure_img = Image.open(io.BytesIO(ws.assets().get("structure.png")))

        # 桌面陪伴模式
        states = [asleep, waken, boring]
        default_state = WakenState.NAME

        # 直播模式下，只使用直播状态，预计未来会增加一个直播讲课状态
        if os.getenv("REACHY_MINI_MODE") == "live":
            if live is None:
                raise RuntimeError(
                    "REACHY_MINI_MODE=live requires optional dependencies. "
                    "Please install project with 'douyin_live' extras."
                )
            states = [asleep, live]
            default_state = LiveState.NAME  # type: ignore[union-attr]

        return MossInReachyMini(
            mini, *states,
            default_state=default_state,
            appearance_img=appearance_img, structure_img=structure_img,
            logger=logger,
            recorder=recorder,
        )
