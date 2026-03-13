import asyncio
import io
import logging
import os
import time
from typing import List, Callable

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
from moss_in_reachy_mini.state.abcd import MiniStateHook, InitialState
from moss_in_reachy_mini.state.asleep import AsleepState
from moss_in_reachy_mini.state.boring import BoringState
from moss_in_reachy_mini.state.teaching import TeachingState
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
            body: Body,
            head: Head,
            antennas: Antennas,
            vision: Vision,
    ):
        self.mini = mini
        self.logger = logger or logging.getLogger(__name__)

        # components
        self.body = body
        self.head = head
        self.antennas = antennas
        self.vision = vision

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
        # outlook message
        messages = [
            Message.new(role="user", name="__reachy_mini_outlook__").with_content(
                Text(text="These two images shows your appearance and structure"),
                Base64Image.from_pil_image(self.appearance_img),
                Base64Image.from_pil_image(self.structure_img),
            )
        ]

        # state context_messages
        state_message = Message.new(role="user", name="__reachy_mini_state__").with_content(
            Text(text=f"You are under {self._state.NAME} state"),
        )
        now = int(time.time())
        for state in self._state_log:
            ago = now - state.now
            if not state.from_state:
                text = f"Start state to {state.to_state.NAME} occurred {ago} seconds ago"
            else:
                text = f"Switch state from {state.from_state.NAME} to {state.to_state.NAME} occurred {ago} seconds ago"
            state_message.with_content(Text(text=text))
        self._state_log.clear()
        messages.append(state_message)

        # components context messages
        head_messages = await self.head.context_messages()
        antenna_messages = await self.antennas.context_messages()
        messages.extend(head_messages)
        messages.extend(antenna_messages)

        return messages

    def is_available_fn(self, *available_states) -> Callable[[], bool]:
        return lambda : self._state.NAME in available_states

    def as_channel(self, only_context_messages=False) -> PyChannel:
        self.logger.info("MossInReachyMini.as_channel()...")

        # name不可改，其他地方有直接使用reach_mini作为路径的ctml事件
        reachy_mini = PyChannel(name="reachy_mini", description="reachy mini root channel", blocking=True)

        if only_context_messages:
            # 只共享当前的reachy mini的全量动态上下文给旁路的Agent
            reachy_mini.build.context_messages(self.context_messages)
            return reachy_mini

        # 支持大模型自主切换reachy mini的仿生状态
        reachy_mini.build.command(doc=f"""
        切换到指定状态，当前状态为{self._state.NAME}，可选状态有{', '.join([s.NAME for s in self._state_map.values()])}

        :param state_name: 目标状态名称
        :param force: 务必使用默认值False，任何情况都不能设置为True
        """)(self.switch_state)

        # recorder（即vlog）独立成轨，后面可以和vision channel合并
        sub_channels = []
        if self._recorder is not None:
            recorder_chan = VideoRecorder(self._recorder).as_channel()
            sub_channels.append(recorder_chan)

        # 视觉独立成轨，不会和其他动作轨打架
        vision_chan = self.vision.as_channel()
        # 目前仅支持Waken和Boring状态可以使用视觉，主要是为了屏蔽掉Live直播状态下的视觉上下文分散注意力
        vision_chan.build.available(self.is_available_fn(WakenState.NAME, BoringState.NAME))
        sub_channels.append(vision_chan)

        # 注册子轨道
        reachy_mini.import_channels(
            *sub_channels,
        )

        # 注册自身command，都是动作
        # 动作单轨化：reachy mini的可执行动作比较简单，拆多轨增加复杂度而且容易导致电机打架抽搐，单轨化的表现力已经足够了
        reachy_mini.build.command(
            doc=self.body.dance_docstring,
            available=self.is_available_fn(WakenState.NAME, LiveState.NAME, TeachingState.NAME),
        )(self.body.dance)

        reachy_mini.build.command(
            doc=self.body.emotion_docstring,
            available=self.is_available_fn(WakenState.NAME, LiveState.NAME, TeachingState.NAME),
        )(self.body.emotion)

        reachy_mini.build.command(
            name="head_move",
            available=self.is_available_fn(WakenState.NAME, LiveState.NAME, TeachingState.NAME),
        )(self.head.move)

        reachy_mini.build.command(
            name="head_reset",
            available=self.is_available_fn(WakenState.NAME, LiveState.NAME, TeachingState.NAME),
        )(self.head.reset)

        reachy_mini.build.command(
            available=self.is_available_fn(WakenState.NAME),
        )(self.head.start_tracking_face)

        reachy_mini.build.command(
            available=self.is_available_fn(WakenState.NAME),
        )(self.head.stop_tracking_face)

        reachy_mini.build.command(
            name="antennas_move",
            available=self.is_available_fn(WakenState.NAME, LiveState.NAME, TeachingState.NAME),
        )(self.antennas.move)

        reachy_mini.build.command(
            name="antennas_reset",
            available=self.is_available_fn(WakenState.NAME, LiveState.NAME, TeachingState.NAME),
        )(self.antennas.reset)

        # 注册idle状态的默认动作
        # 呼吸 或 人脸跟随
        reachy_mini.build.idle(self.head.on_idle)

        # 挂载启动和退出到MOSS生命周期
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
        teaching = con.force_fetch(TeachingState)

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
        states = [asleep, waken, boring, teaching]
        default_state = WakenState.NAME

        # 直播模式下，只使用直播状态，预计未来会增加一个直播讲课状态
        if os.getenv("REACHY_MINI_MODE") == "live":
            if live is None:
                raise RuntimeError(
                    "REACHY_MINI_MODE=live requires optional dependencies. "
                    "Please install project with 'douyin_live' extras."
                )
            states = [asleep, live, teaching]
            default_state = LiveState.NAME  # type: ignore[union-attr]

        # components
        body = con.force_fetch(Body)
        head = con.force_fetch(Head)
        antennas = con.force_fetch(Antennas)
        vision = con.force_fetch(Vision)

        return MossInReachyMini(
            mini, *states,
            default_state=default_state,
            appearance_img=appearance_img, structure_img=structure_img,
            logger=logger,
            recorder=recorder,
            body=body,
            head=head,
            antennas=antennas,
            vision=vision,
        )
