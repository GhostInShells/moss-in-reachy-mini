import asyncio
import logging
import os

from ghoshell_common.contracts import LoggerItf, Workspace
from ghoshell_container import Container, get_container, Provider, IoCContainer, INSTANCE
from ghoshell_moss import Speech, MOSSShell
from ghoshell_moss import new_shell
from ghoshell_moss.core.shell.main_channel import create_main_channel
from ghoshell_moss.transports.zmq_channel import ZMQChannelHub
from ghoshell_moss.transports.zmq_channel.zmq_hub import ZMQHubConfig, ZMQProxyConfig
from ghoshell_moss_contrib.agent import ConsoleChat
from ghoshell_moss_contrib.agent.chat.base import BaseChat
from reachy_mini import ReachyMini

from framework.abcd.agent import AgentConfig, ModelConf, EventBus, Agent
from framework.abcd.memory import Memory
from framework.agent.broadcaster import ChatBroadcasterProvider
from framework.agent.eventbus import QueueEventBus
from framework.agent.main_agent import MainAgent
from framework.memory.storage_memory import StorageMemory
from framework.agent.utils import run_agent_with_chat
from moss_in_reachy_mini.audio.player import ReachyMiniStreamPlayer
from moss_in_reachy_mini.components.antennas import AntennasProvider
from moss_in_reachy_mini.components.body import BodyProvider
from moss_in_reachy_mini.components.head import HeadProvider
from moss_in_reachy_mini.components.head_tracker import HeadTrackerProvider
from moss_in_reachy_mini.components.vision import VisionProvider
from moss_in_reachy_mini.listener.chat.console_ptt import ConsolePTTChat
from moss_in_reachy_mini.moss import MossInReachyMini, MossInReachyMiniProvider
from moss_in_reachy_mini.utils import load_instructions
from moss_in_reachy_mini.camera.camera_worker import CameraWorkerProvider

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ShellProvider(Provider[MOSSShell]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        mini = con.force_fetch(ReachyMini)
        moss = con.force_fetch(MossInReachyMini)
        memory = con.force_fetch(StorageMemory)
        speech = get_speech(mini, con, default_speaker="saturn_zh_female_keainvsheng_tob")
        shell = new_shell(container=con, speech=speech, main_channel=create_main_channel())
        shell.main_channel.import_channels(
            moss.as_channel(),
            memory.as_channel(),
            # zmq_hub.as_channel()
        )
        return shell

class AgentProvider(Provider[Agent]):
    def __init__(self, config: AgentConfig):
        self._config = config

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        shell = con.force_fetch(MOSSShell)
        memory = con.force_fetch(Memory)
        moss = con.force_fetch(MossInReachyMini)
        return MainAgent(container=con, config=self._config, shell=shell, memory=memory, hook_state=moss)

def providers(container: IoCContainer):
    # Mini
    container.set(ReachyMini, ReachyMini())
    # Agent输入
    container.set(EventBus, QueueEventBus())
    # Agent输出
    container.register(ChatBroadcasterProvider())
    container.set(BaseChat, ConsolePTTChat())
    # dependency registry
    container.register(BodyProvider())
    container.register(HeadProvider())
    container.register(AntennasProvider())
    container.register(VisionProvider())
    container.register(HeadTrackerProvider())
    container.register(CameraWorkerProvider())
    # Agent记忆
    ws = container.force_fetch(Workspace)
    storage = ws.runtime().sub_storage("memory")
    memory = StorageMemory(storage)
    container.set(StorageMemory, memory)
    container.set(Memory, memory)
    container.register(ShellProvider())  # Shell
    # Agent
    instructions = load_instructions(
        container,
        files=["ctml_enrich.md"],
        storage_name="reachy_mini_instructions",
    )
    container.register(AgentProvider(AgentConfig(
        id="reachy_mini",
        name="reachy_mini",
        description="",
        model=ModelConf(
            kwargs={
                "thinking": {
                    "type": "disabled",
                },
            },
        ),
        instructions=instructions,
    )))
    # Moss
    container.register(MossInReachyMiniProvider())


async def run_agent(container, zmq_hub):
    providers(container)
    _mini = container.force_fetch(ReachyMini)
    with _mini:
        moss = container.force_fetch(MossInReachyMini)
        async with moss:
            agent = container.make(Agent)
            chat = container.force_fetch(BaseChat)
            await run_agent_with_chat(agent, chat)


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
    return TTSSpeech(player=ReachyMiniStreamPlayer(mini, logger=container.get(LoggerItf)), tts=VolcengineTTS(conf=tts_conf), logger=container.get(LoggerItf))


def main():
    import pathlib
    ws_dir = pathlib.Path(__file__).parent.joinpath(".workspace")
    current_dir = pathlib.Path(__file__).parent
    root_dir = str(current_dir.parent.joinpath("moss_zmq_channels").absolute())

    from ghoshell_moss_contrib.example_ws import workspace_container

    with workspace_container(ws_dir) as container:
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
        asyncio.run(run_agent(container, zmq_hub))

if __name__ == "__main__":
    main()


