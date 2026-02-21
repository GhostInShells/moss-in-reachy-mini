import asyncio
import logging
import os
from typing import Optional

import cv2
from PIL import Image
from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Container, get_container, IoCContainer
from ghoshell_moss import PyChannel, Message, Base64Image, Text, Speech
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini.utils import create_head_pose

from audio.player import ReachyMiniStreamPlayer
from channels.antennas import Antennas
from channels.head import Head, HeadMove
from reachy_mini_dances_library import DanceMove
from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
from state import ReachyMiniState, BodyYawMove
from utils import load_instructions, load_emotions

logger = logging.getLogger('reachy_mini_moss')
logger.setLevel(logging.INFO)


class ReachyMiniMoss:
    def __init__(self, mini: ReachyMini, container: IoCContainer=None):
        self.mini = mini
        self.mini.set_target_body_yaw(0.0)
        self._state = ReachyMiniState()
        self._state.twisting.set()

        self._head = Head(mini, self._state, logger, container)
        self._antennas = Antennas(mini, self._state, logger)

        self._emotions_storage, self._emotions = load_emotions(container)

        self._bootstrapped = asyncio.Event()
        self._twisting_task: Optional[asyncio.Task] = None

    async def dance(self, name: str):
        if not AVAILABLE_MOVES.get(name):
            raise ValueError(f'{name} is not a valid dance')
        await self.mini.async_play_move(DanceMove(name))
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(),
        ))

    async def emotion(self, name: str, play_sound: bool = True):
        params = self._emotions.get(name)
        if not params:
            raise ValueError(f"{name} is not a valid emotion")

        sound_path = None
        if play_sound:
            sound_path = f"{self._emotions_storage.abspath()}/{name}.wav"

        await self.mini.async_play_move(RecordedMove(move=params, sound_path=sound_path))
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(),
        ))

    async def wake_up(self):
        self.mini.enable_motors()
        self.mini.wake_up()
        self._state.waken.set()
        await asyncio.sleep(1)

    async def goto_sleep(self):
        self._state.tracking.clear()
        self.mini.goto_sleep()
        self.mini.disable_motors()
        self._state.waken.clear()

    async def context_messages(self):
        msg = Message.new(role="user", name="__reachy_mini__")

        appearance_img = Image.open(".workspace/assets/appearance.png")
        structure_img = Image.open(".workspace/assets/structure.png")
        msg.with_content(
            Text(text="These two images shows your appearance and structure"),
            Base64Image.from_pil_image(appearance_img),
            Base64Image.from_pil_image(structure_img),
        )

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

        if self._state.twisting.is_set():
            msg.with_content(
                Text(text="You are twisting on idle.")
            )

        return [msg]

    async def start_twisting(self):
        self._state.twisting.set()

    async def stop_twisting(self):
        self.mini.set_target_body_yaw(0.0)
        self._state.twisting.clear()

    async def _twisting(self):
        try:  # 捕获取消异常，确保任务优雅退出
            while self._state.waken.is_set() and self._state.twisting.is_set():
                await self.mini.async_play_move(
                    BodyYawMove(self._state.start_body_yaw, 10, 1.5),
                )
                self._state.start_body_yaw = 10
                await self.mini.async_play_move(
                    BodyYawMove(self._state.start_body_yaw, -10, 1.5),
                )
                self._state.start_body_yaw = -10
        except asyncio.CancelledError:
            logger.info("Twisting task was cancelled")
            raise  # 重新抛出，让外层await能捕获

    async def on_policy_run(self):
        if not self._state.waken.is_set():
            return

        # 先取消旧任务（如果存在），避免多任务并发
        if self._twisting_task and not self._twisting_task.done():
            self._twisting_task.cancel()
            try:
                await self._twisting_task
            except asyncio.CancelledError:
                pass
        self._twisting_task = asyncio.create_task(self._twisting())

    async def on_policy_pause(self):
        # 1. 边界检查：任务存在且未完成时才取消
        if self._twisting_task and not self._twisting_task.done():
            self._twisting_task.cancel()
            try:
                # 2. 捕获取消异常，避免程序崩溃
                await self._twisting_task
            except asyncio.CancelledError:
                logger.info("Twisting task cancelled successfully")
            finally:
                self._twisting_task = None
        self.mini.set_target_body_yaw(0.0)

    async def integrated_on_policy_run(self):
        await self.on_policy_run()
        await self._head.on_policy_run()
        await self._antennas.on_policy_run()

    async def integrated_on_policy_pause(self):
        await self.on_policy_pause()
        await self._head.on_policy_pause()
        await self._antennas.on_policy_pause()

    async def integrated_context_messages(self):
        body_msg = await self.context_messages()
        head_msg = await self._head.context_messages()
        antenna_msg = await self._antennas.context_messages()
        return body_msg + head_msg + antenna_msg

    def as_channel(self) -> PyChannel:
        logger.info("as channel")
        assert self._bootstrapped.is_set()

        body = PyChannel(name="reachy_mini", description=f"sleep head pose is {SLEEP_HEAD_POSE}", block=True)

        # lifecycle
        body.build.command()(self.wake_up)
        body.build.command()(self.goto_sleep)

        body.build.command()(self.start_twisting)
        body.build.command()(self.stop_twisting)
        body.build.on_policy_run(self.integrated_on_policy_run)
        body.build.on_policy_pause(self.integrated_on_policy_pause)
        body.build.with_context_messages(self.integrated_context_messages)

        # dance
        dance_docstrings = []
        for name, move in AVAILABLE_MOVES.items():
            func, params, meta = move
            dance_docstrings.append(f"name: {name} description: {meta.get("description", "")} subcycles per beat: {params.get('subcycles_per_beat', 1.0)}")
        body.build.command(doc=f"Dance can be chosen in \n{"\n".join(dance_docstrings)}")(self.dance)

        # emotions
        emotion_docstrings = []
        for name, params in self._emotions.items():
            emotion_docstrings.append(f"name: {name} description: {params.get('description', '')}")
        body.build.command(doc=f"Emotion can be chosen in \n{"\n".join(emotion_docstrings)}")(self.emotion)

        # head
        body.build.command(name="head_move")(self._head.move)
        body.build.command(name="head_reset")(self._head.reset)
        body.build.command()(self._head.start_tracking_face)
        body.build.command()(self._head.stop_tracking_face)

        # antennas
        body.build.command(name="antennas_move")(self._antennas.move)
        body.build.command(name="antennas_reset")(self._antennas.reset)
        body.build.command()(self._antennas.set_idle_flapping)
        body.build.command()(self._antennas.enable_flapping)

        # body.import_channels(
        #     self._head.as_channel(),
        #     self._antennas.as_channel(),
        # )

        return body

    async def bootstrap(self):
        await self._head.bootstrap()
        self._bootstrapped.set()

    async def __aenter__(self):
        await self.bootstrap()
        return self

    async def aclose(self):
        await self._head.aclose()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()


async def run_agent(container, root_dir):
    from ghoshell_moss import new_shell
    from ghoshell_moss_contrib.agent import SimpleAgent, ModelConf
    from ghoshell_moss.transports.zmq_channel import ZMQChannelHub
    from ghoshell_moss.transports.zmq_channel.zmq_hub import ZMQHubConfig, ZMQProxyConfig

    # hub channel
    zmq_hub = ZMQChannelHub(
        config=ZMQHubConfig(
            name="hub",
            description="可以启动指定的子通道并运行.",
            # todo: 当前版本全部基于约定来做. 快速验证.
            root_dir=root_dir,
            proxies={
                "slide": ZMQProxyConfig(
                    script="slide_app.py",
                    description="可以打开你的slide studio gui，通过这个通道你可以呈现并讲述一个slide主题",
                ),
            },
        ),
    )

    with ReachyMini() as _mini:
        async with ReachyMiniMoss(_mini, container) as moss:
            speech = get_speech(_mini, container, default_speaker="saturn_zh_female_keainvsheng_tob")
            shell = new_shell(container=container, speech=speech)
            shell.main_channel.import_channels(
                moss.as_channel(),
                # zmq_hub.as_channel()
            )
            instructions = load_instructions(
                container,
                ["persona.md"],
                "reachy_mini_instructions",
            )
            agent = SimpleAgent(
                instruction=instructions,
                shell=shell,
                speech=speech,
                model=ModelConf(
                    kwargs={
                        "thinking": {
                            "type": "disabled",
                        },
                    },
                ),
                container=container,
            )

            await agent.run()


def get_speech(
    mini: ReachyMini,
    container: Container | None = None,
    default_speaker: str | None = None,
) -> Speech:
    from ghoshell_moss.speech import TTSSpeech
    from ghoshell_moss.speech.mock import MockSpeech
    from ghoshell_moss.speech.volcengine_tts import VolcengineTTS, VolcengineTTSConf

    container = container or get_container()
    use_voice = os.environ.get("USE_VOICE_SPEECH", "no") == "yes"
    if not use_voice:
        return MockSpeech()
    app_key = os.environ.get("VOLCENGINE_STREAM_TTS_APP")
    app_token = os.environ.get("VOLCENGINE_STREAM_TTS_ACCESS_TOKEN")
    resource_id = os.environ.get("VOLCENGINE_STREAM_TTS_RESOURCE_ID", "seed-tts-2.0")
    if not app_key or not app_token:
        raise NotImplementedError(
            "Env $VOLCENGINE_STREAM_TTS_APP or $VOLCENGINE_STREAM_TTS_ACCESS_TOKEN not configured."
            "Maybe examples/.env not set, or you need to set $USE_VOICE_SPEECH `no`"
        )
    tts_conf = VolcengineTTSConf(
        app_key=app_key,
        access_token=app_token,
        resource_id=resource_id,
        sample_rate=mini.media.get_output_audio_samplerate(),
    )
    if default_speaker:
        tts_conf.default_speaker = default_speaker
    return TTSSpeech(player=ReachyMiniStreamPlayer(mini), tts=VolcengineTTS(conf=tts_conf), logger=container.get(LoggerItf))


def main():
    import pathlib
    ws_dir = pathlib.Path(__file__).parent.joinpath(".workspace")
    current_dir = pathlib.Path(__file__).parent
    root_dir = str(current_dir.parent.joinpath("moss_zmq_channels").absolute())

    from ghoshell_moss_contrib.example_ws import workspace_container

    with workspace_container(ws_dir) as container:
        asyncio.run(run_agent(container, root_dir))

if __name__ == "__main__":
    main()


