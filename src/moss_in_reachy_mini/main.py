import asyncio
import os

from ghoshell_common.contracts import LoggerItf, Workspace
from ghoshell_container import get_container, Provider, IoCContainer, INSTANCE
from ghoshell_moss import MOSSShell
from ghoshell_moss import new_ctml_shell
from ghoshell_moss.speech import BaseTTSSpeech
from ghoshell_moss.transports.zmq_channel import ZMQChannelHub
from ghoshell_moss.transports.zmq_channel.zmq_hub import ZMQHubConfig, ZMQProxyConfig
from ghoshell_moss_contrib.agent.chat.base import BaseChat
from reachy_mini import ReachyMini

from framework.abcd.agent import AgentConfig, ModelConf, EventBus, Agent
from framework.abcd.memory import Memory
from framework.agent.agent_fastapi import AgentFastAPIProvider, AgentFastAPI
from framework.agent.broadcaster import ChatBroadcasterProvider
from framework.agent.eventbus import QueueEventBus
from framework.agent.main_agent import MainAgent
from framework.agent.utils import run_agent_with_chat
from framework.apps.live.douyin_live import DouyinLiveProvider
from framework.apps.memory.storage_memory import StorageMemory
from framework.apps.news import NewsAPIProvider
from framework.apps.todolist import TodoList, TodoListProvider
from moss_in_reachy_mini.audio.mic_hub import MicHubProvider
from moss_in_reachy_mini.audio.player import ReachyMiniStreamPlayer
from moss_in_reachy_mini.camera.camera_worker import CameraWorkerProvider
from moss_in_reachy_mini.camera.frame_hub import FrameHubProvider
from moss_in_reachy_mini.components.antennas import AntennasProvider
from moss_in_reachy_mini.components.body import BodyProvider
from moss_in_reachy_mini.components.head import HeadProvider
from moss_in_reachy_mini.components.head_tracker import HeadTrackerProvider
from moss_in_reachy_mini.components.vision import VisionProvider
from moss_in_reachy_mini.listener.chat.console_ptt import ConsolePTTChat
from moss_in_reachy_mini.logger import setup_logger
from moss_in_reachy_mini.moss import MossInReachyMini, MossInReachyMiniProvider
from moss_in_reachy_mini.state import AsleepStateProvider, WakenStateProvider, BoringStateProvider, LiveStateProvider
from moss_in_reachy_mini.utils import load_instructions
from moss_in_reachy_mini.video.recorder_worker import VideoRecorderWorker, VideoRecorderWorkerProvider


class ShellProvider(Provider[MOSSShell]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        mini = con.force_fetch(ReachyMini)
        moss = con.force_fetch(MossInReachyMini)
        memory = con.force_fetch(StorageMemory)
        todolist = con.force_fetch(TodoList)
        # news = con.force_fetch(NewsAPI)
        shell = new_ctml_shell(
            container=con,
            speech=get_speech(mini, default_speaker="saturn_zh_female_keainvsheng_tob", logger_=con.get(LoggerItf)),
            experimental=False,
        )
        shell.main_channel.import_channels(
            moss.as_channel(),
            memory.as_channel(),
            todolist.as_channel(),
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
    container.register(FrameHubProvider())
    # Shared microphone capture (avoid multi-stream conflicts)
    container.register(MicHubProvider())
    container.register(AgentFastAPIProvider())
    # Agent输入
    container.set(EventBus, QueueEventBus())
    # Agent输出
    container.register(ChatBroadcasterProvider())
    container.set(BaseChat, ConsolePTTChat(container=container))
    # dependency registry
    container.register(BodyProvider())
    container.register(HeadProvider())
    container.register(AntennasProvider())
    container.register(VisionProvider())
    container.register(HeadTrackerProvider())
    container.register(CameraWorkerProvider())
    container.register(VideoRecorderWorkerProvider())
    # Agent记忆
    ws = container.force_fetch(Workspace)
    storage_name = os.getenv("REACHY_MINI_MEMORY_STORAGE", "memory")
    logger = container.get(LoggerItf)
    if logger:
        logger.info(f"Reachy Mini memory storage set to '{storage_name}'")
    storage = ws.runtime().sub_storage(storage_name)
    memory = StorageMemory(storage)
    container.set(StorageMemory, memory)
    container.set(Memory, memory)
    # TodoList
    container.register(TodoListProvider())
    # News
    container.register(NewsAPIProvider())
    # Douyin Live
    container.register(DouyinLiveProvider())
    # Shell
    container.register(ShellProvider())
    # Agent
    instructions = load_instructions(
        container,
        files=["ctml_enrich.md", "speech.md", "websearch.md", "news.md"],
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
                "extra_body": {
                    "enable_web_search": True
                }
            },
        ),
        instructions=instructions,
    )))
    # Moss
    container.register(MossInReachyMiniProvider())
    # Moss State
    container.register(AsleepStateProvider())
    container.register(WakenStateProvider())
    container.register(BoringStateProvider())
    container.register(LiveStateProvider())


async def run_agent(container, zmq_hub):
    providers(container)
    _mini = container.force_fetch(ReachyMini)
    with _mini:
        moss = container.force_fetch(MossInReachyMini)
        async with moss:
            agent = container.make(Agent)
            chat = container.make(BaseChat)
            agent_fastapi = container.make(AgentFastAPI)

            await asyncio.gather(
                run_agent_with_chat(agent, chat),
                agent_fastapi.run()
            )


def get_speech(
    mini: ReachyMini,
    default_speaker: str | None = None,
    logger_: LoggerItf=None,
    container: IoCContainer=None,
) -> BaseTTSSpeech:
    from ghoshell_moss.speech.volcengine_tts import VolcengineTTS, VolcengineTTSConf

    container = container or get_container()
    try:
        recorder = container.get(VideoRecorderWorker)
    except Exception:
        recorder = None
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
    return BaseTTSSpeech(
        tts=VolcengineTTS(conf=tts_conf),
        player=ReachyMiniStreamPlayer(mini, logger=container.get(LoggerItf), recorder=recorder),
    )


def main():
    import pathlib

    ws_dir = pathlib.Path(__file__).parent.joinpath(".workspace")
    current_dir = pathlib.Path(__file__).parent
    root_dir = str(current_dir.parent.joinpath("moss_zmq_channels").absolute())

    from ghoshell_moss_contrib.example_ws import workspace_container

    with workspace_container(ws_dir) as container:
        logger = setup_logger(str(ws_dir.joinpath("runtime/logs/moss_demo.log").absolute()),)
        container.set(LoggerItf, logger)

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
