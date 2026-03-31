import asyncio
import bisect
import logging
import os
import time
from collections.abc import Callable
from typing import List

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import INSTANCE, IoCContainer, Provider
from ghoshell_moss import Message, PyChannel, Text
from reachy_mini import ReachyMini

from framework.abcd.agent_hook import AgentHook
from framework.apps.chinese_chess.channel import IS_ENABLE_CHINESE_CHESS
from moss_in_reachy_mini.components.antennas import Antennas
from moss_in_reachy_mini.components.body import Body
from moss_in_reachy_mini.components.head import Head
from moss_in_reachy_mini.components.music import MusicSearch
from moss_in_reachy_mini.components.sound import Sound
from moss_in_reachy_mini.components.vision import Vision
from moss_in_reachy_mini.state.abcd import InitialState, BaseAgentHook
from moss_in_reachy_mini.state.asleep import AsleepState
from moss_in_reachy_mini.state.boring import BoringState
from moss_in_reachy_mini.state.enrolling import EnrollingState
from moss_in_reachy_mini.state.live import LiveState
from moss_in_reachy_mini.state.teaching import TeachingState
from moss_in_reachy_mini.state.waken import WakenState
from moss_in_reachy_mini.state.chess import ChessPlayingState
from moss_in_reachy_mini.video.recorder_channel import VideoRecorder
from moss_in_reachy_mini.video.recorder_worker import VideoRecorderWorker


class StateLog:
    def __init__(self, from_state: BaseAgentHook, to_state: BaseAgentHook):
        self.from_state = from_state
        self.to_state = to_state
        self.now = int(time.time())


class MossInReachyMini:
    def __init__(
        self,
        mini: ReachyMini,
        *states: BaseAgentHook,
        default_state: str = AsleepState.NAME,
        logger: LoggerItf = None,
        recorder: VideoRecorderWorker | None = None,
        body: Body,
        head: Head,
        antennas: Antennas,
        sound: Sound,
        vision: Vision,
        music: MusicSearch,
    ):
        self.mini = mini
        self.logger = logger or logging.getLogger(__name__)

        # components
        self.body = body
        self.head = head
        self.antennas = antennas
        self.sound = sound
        self.vision = vision
        self.music = music

        # state
        self._state_map = {state.NAME: state for state in states}
        self._state: BaseAgentHook = InitialState()
        self._state_log: list[StateLog] = []
        self._default_state = default_state

        self._recorder = recorder

        self._bootstrapped = asyncio.Event()

    async def __aenter__(self) -> "MossInReachyMini":
        # Provide an async context manager for lightweight scripts/tests.
        # The full runtime also wires `bootstrap/aclose` into channel lifecycle.
        await self.bootstrap()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        await self.aclose()

    async def switch_state(self, state_name: str, force: bool = False):
        state_name = state_name.lower()
        if state_name not in self._state_map:
            raise ValueError(f"Invalid state name: {state_name}")

        if state_name == self._state.NAME:
            return

        if not self._state.out_switchable and not force:
            raise ValueError(f"Current state {self._state.NAME} is not out switchable")

        if not self._state_map[state_name].in_switchable and not force:
            raise ValueError(f"Current state {state_name} is not in switchable")

        await self._state.on_self_exit()
        self.logger.info(f"Switching state from {self._state.NAME} to {state_name}")
        self._state_log.append(StateLog(self._state, self._state_map[state_name]))  # 记录状态切换
        self._state = self._state_map[state_name]
        await self._state.on_self_enter()

    # 交给MainAgent来控制生命周期
    def get_hook(self) -> AgentHook:
        return self._state

    async def context_messages(self):
        messages = []

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
        messages.extend(head_messages)

        # music context (suppress speech during playback)
        music_messages = await self.music.context_messages()
        messages.extend(music_messages)

        return messages

    def is_available_fn(self, *available_states) -> Callable[[], bool]:
        return lambda: self._state.NAME in available_states

    # ------------------------------------------------------------------
    # Beat-sync: align dance/emotion to music beats when playing
    # ------------------------------------------------------------------

    async def _wait_for_beat(self):
        """If music is playing, wait until the next beat boundary."""
        song = self.music._current_song
        if song is None:
            return

        start = self.music._playback_start_time
        if start is None:
            return

        beat_times: list[float] = song.get("beat_times", [])
        if not beat_times:
            return

        elapsed = time.monotonic() - start
        idx = bisect.bisect_right(beat_times, elapsed)
        if idx < len(beat_times):
            wait = beat_times[idx] - elapsed
            if 0 < wait < 3.0:
                await asyncio.sleep(wait)

    async def _beat_synced_dance(self, name: str):
        """Dance with beat alignment when music is playing."""
        await self._wait_for_beat()
        return await self.body.dance(name)

    async def _beat_synced_emotion(self, emoji: str):
        """Emotion with beat alignment when music is playing."""
        await self._wait_for_beat()
        return await self.body.emotion(emoji)

    async def _beat_synced_head_move(self, **kwargs):
        """Head move with beat alignment when music is playing."""
        await self._wait_for_beat()
        return await self.head.move(**kwargs)

    async def _beat_synced_antennas_move(self, **kwargs):
        """Antennas move with beat alignment when music is playing."""
        await self._wait_for_beat()
        return await self.antennas.move(**kwargs)

    # ------------------------------------------------------------------

    def as_channel(self, only_context_messages=False) -> PyChannel:
        self.logger.info("MossInReachyMini.as_channel()...")

        # name不可改，其他地方有直接使用reach_mini作为路径的ctml事件
        reachy_mini = PyChannel(name="reachy_mini", description="reachy mini root channel", blocking=True)

        if only_context_messages:
            # 只共享当前的reachy mini的全量动态上下文给旁路的Agent
            reachy_mini.build.context_messages(self.context_messages)
            return reachy_mini

        reachy_mini.build.context_messages(self.context_messages)

        # 支持大模型自主切换reachy mini的仿生状态
        reachy_mini.build.command(
            doc=f"""
        切换到指定状态，当前状态为{self._state.NAME}，可选状态有{", ".join([s.NAME for s in self._state_map.values()])}

        :param state_name: 目标状态名称
        :param force: 务必使用默认值False，任何情况都不能设置为True
        """
        )(self.switch_state)

        # recorder（即vlog）独立成轨，后面可以和vision channel合并
        sub_channels = []
        if self._recorder is not None:
            recorder_chan = VideoRecorder(self._recorder).as_channel()
            sub_channels.append(recorder_chan)

        # 视觉独立成轨，不会和其他动作轨打架

        if not os.getenv("DISABLE_VISION"):
            vision_chan = self.vision.as_channel()
            # 目前仅支持Waken和Boring状态可以使用视觉，主要是为了屏蔽掉Live直播状态下的视觉上下文分散注意力
            vision_chan.build.available(self.is_available_fn(WakenState.NAME, BoringState.NAME))
            sub_channels.append(vision_chan)

        # 语音输出独立成轨，不会和其他动作打架
        sound_chan = self.sound.as_channel()
        sound_chan.build.available(self.is_available_fn(WakenState.NAME, BoringState.NAME))
        sub_channels.append(sound_chan)

        # 注册子轨道
        reachy_mini.import_channels(
            *sub_channels,
        )

        # 注册自身command，都是动作
        # 动作单轨化：reachy mini的可执行动作比较简单，拆多轨增加复杂度而且容易导致电机打架抽搐，单轨化的表现力已经足够了
        # dance/emotion 使用 beat-sync 包装，音乐播放时自动对齐节拍
        reachy_mini.build.command(
            name="dance",
            doc=self.body.dance_docstring,
            available=self.is_available_fn(WakenState.NAME, LiveState.NAME, TeachingState.NAME, ChessPlayingState.NAME),
        )(self._beat_synced_dance)

        reachy_mini.build.command(
            name="emotion",
            doc=self.body.emotion_docstring,
            available=self.is_available_fn(WakenState.NAME, LiveState.NAME, TeachingState.NAME, ChessPlayingState.NAME),
        )(self._beat_synced_emotion)

        reachy_mini.build.command(
            name="head_move",
            available=self.is_available_fn(WakenState.NAME, LiveState.NAME, TeachingState.NAME, ChessPlayingState.NAME),
        )(self._beat_synced_head_move)

        reachy_mini.build.command(
            name="head_reset",
            available=self.is_available_fn(
                WakenState.NAME, LiveState.NAME, TeachingState.NAME, EnrollingState.NAME, ChessPlayingState.NAME
            ),
        )(self.head.reset)

        reachy_mini.build.command(
            available=self.is_available_fn(WakenState.NAME, EnrollingState.NAME),
        )(self.head.start_tracking_face)

        reachy_mini.build.command(
            available=self.is_available_fn(WakenState.NAME, EnrollingState.NAME),
        )(self.head.stop_tracking_face)

        reachy_mini.build.command(
            name="antennas_move",
            available=self.is_available_fn(WakenState.NAME, LiveState.NAME, TeachingState.NAME, ChessPlayingState.NAME),
        )(self._beat_synced_antennas_move)

        reachy_mini.build.command(
            name="antennas_reset",
            available=self.is_available_fn(WakenState.NAME, LiveState.NAME, TeachingState.NAME, ChessPlayingState.NAME),
        )(self.antennas.reset)

        reachy_mini.build.command(
            name="play_music",
            doc=(
                "搜索并播放音乐。"
                "query 必须是具体的「歌手+歌名」或「歌名」，如'周杰伦 晴天'、'光年之外'。"
                "\n**禁止**将用户的模糊描述（如'下雨天适合听的歌'、'欢快的歌'）直接作为 query。"
                "\n当用户给出心情、场景或风格描述时，你需要先根据自己的音乐知识推荐具体歌曲，"
                "然后直接用具体歌名调用本命令，不要询问用户确认。可以边说你的推荐边播放。"
                "\n\n**continuous 参数（必读）**："
                "\n- 播放单首歌：continuous不传或False。播完后系统通知你结束。"
                "\n- 播放多首歌（用户说'播一组'、'播几首'、'连续播放'等）：**必须传 continuous=True**。"
                "  每首歌播完后系统会通知你，届时请评论刚才的歌并继续调用 play_music（**仍带 continuous=True**）播下一首。"
                "\n- 每次调用 play_music 都会自动停掉当前正在播放的歌曲并清空所有旧状态。"
                "\n\nduration 不要传，除非用户在**当前这次请求**中明确说了只听多少秒。"
                "不要沿用之前对话中的duration设置，每次都按用户最新的要求判断。默认播完整首歌。"
                "\n播放后系统会向你发送编舞请求，届时请用dance/emotion/head_move/antennas_move自由编排。"
                "\n停止/暂停/恢复音乐请用 stop_music、pause_music、resume_music。"
            ),
            available=self.is_available_fn(WakenState.NAME),
        )(self.music.play_music)

        reachy_mini.build.command(
            name="stop_music",
            doc="停止音乐播放并停止所有动作。用户说停止音乐、关掉音乐时使用此命令。",
            available=self.is_available_fn(WakenState.NAME),
        )(self.music.stop_music)

        reachy_mini.build.command(
            name="pause_music",
            doc="暂停音乐播放并停止动作。用户说暂停音乐时使用此命令。",
            available=self.is_available_fn(WakenState.NAME),
        )(self.music.pause_music)

        reachy_mini.build.command(
            name="resume_music",
            doc="恢复音乐播放并继续动作编排。用户说继续播放音乐时使用此命令。",
            available=self.is_available_fn(WakenState.NAME),
        )(self.music.resume_music)

        reachy_mini.build.command(
            name="search_music",
            doc="搜索音乐返回结果列表，不自动播放。用于让用户选择。",
            available=self.is_available_fn(WakenState.NAME),
        )(self.music.search_music)

        reachy_mini.build.command(
            doc=(
                "启动人脸注册/录入流程。当用户说'录入人脸'、'注册人脸'、或同意进行人脸注册时，"
                "必须立即调用此指令，不要自己尝试拍照或引导。"
                "调用后系统将全自动引导用户完成拍照和识别，你不需要再生成任何后续动作或语音。"
                "\n\n:param user_name: 用户告诉你的称呼（必须使用用户明确说出的名字，"
                "禁止使用代词或泛称如'用户'、'你'、'朋友'）"
            ),
            available=self.is_available_fn(WakenState.NAME),
        )(self.start_enrolling)

        # 注册idle状态的默认动作
        # 呼吸 或 人脸跟随
        reachy_mini.build.idle(self.head.on_idle)

        # 挂载启动和退出到MOSS生命周期
        reachy_mini.build.start_up(self.bootstrap)
        reachy_mini.build.close(self.aclose)

        return reachy_mini

    async def start_enrolling(self, user_name: str):
        """启动人脸注册流程，调用后系统将自动引导用户完成拍照和识别，你不需要再生成任何后续动作或语音。

        :param user_name: 用户告诉你的称呼（必须使用用户明确说出的名字，禁止使用代词或泛称）
        """
        face_reg = self._state_map.get(EnrollingState.NAME)
        if face_reg is None:
            raise ValueError("EnrollingState is not registered")
        face_reg.target_name = user_name
        await self.switch_state(EnrollingState.NAME, force=True)

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
        enrolling = con.force_fetch(EnrollingState)

        live = con.force_fetch(LiveState)
        logger = con.get(LoggerItf)

        recorder = con.get(VideoRecorderWorker)

        # 桌面陪伴模式
        states: List[BaseAgentHook] = [asleep, waken, boring, teaching, enrolling]
        default_state = WakenState.NAME

        # 直播模式下，只使用直播状态，预计未来会增加一个直播讲课状态
        if os.getenv("REACHY_MINI_MODE") == "live":
            states: List[BaseAgentHook] = [asleep, live, teaching]
            default_state = LiveState.NAME

        if IS_ENABLE_CHINESE_CHESS:
            chess_playing = con.force_fetch(ChessPlayingState)
            states.append(chess_playing)
            default_state = ChessPlayingState.NAME

        # components
        body = con.force_fetch(Body)
        head = con.force_fetch(Head)
        antennas = con.force_fetch(Antennas)
        vision = con.force_fetch(Vision)
        sound = con.force_fetch(Sound)
        music = con.force_fetch(MusicSearch)

        return MossInReachyMini(
            mini,
            *states,
            default_state=default_state,
            logger=logger,
            recorder=recorder,
            body=body,
            head=head,
            antennas=antennas,
            sound=sound,
            vision=vision,
            music=music,
        )
