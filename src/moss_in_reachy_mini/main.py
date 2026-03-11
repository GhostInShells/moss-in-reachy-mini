import asyncio
import os

from ghoshell_common.contracts import LoggerItf, Workspace
from ghoshell_container import get_container, Provider, IoCContainer, INSTANCE, Container
from ghoshell_moss import MOSSShell
from ghoshell_moss import new_ctml_shell
from ghoshell_moss.speech import BaseTTSSpeech
from reachy_mini import ReachyMini

from framework.abcd.agent import AgentConfig, ModelConf
from framework.abcd.agent_hub import EventBus, AgentHub
from framework.abcd.session import Session
from framework.agent.agent_fastapi import AgentFastAPIProvider, AgentFastAPI
from framework.agent.agent_hub import AgentHubImpl
from framework.agent.broadcaster import ChatBroadcasterProvider
from framework.agent.eventbus import QueueEventBus
from framework.agent.main_agent import MainAgent
from framework.agent.utils import setup_chat
from framework.apps.memory.storage_memory import StorageMemory
from framework.apps.session.storage_session import StorageSession
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
        shell = new_ctml_shell(
            container=con,
            speech=get_speech(mini, default_speaker="saturn_zh_female_keainvsheng_tob", container=con),
            experimental=False,
        )
        shell.main_channel.import_channels(
            moss.as_channel(),
            memory.as_channel(),
            todolist.as_channel(),
        )
        return shell


class MainAgentProvider(Provider[MainAgent]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: Container) -> MainAgent:
        moss = con.force_fetch(MossInReachyMini)
        instructions = load_instructions(
            con,
            files=["ctml_enrich.md", "websearch.md", "news.md"],
            storage_name="main_agent_instructions",
        )
        main_agent = MainAgent.new(
            container=con,
            config=AgentConfig(
                id="main",
                name="main",
                description="",
                model=ModelConf(
                    kwargs={
                        "extra_body": {
                            "thinking": {
                                "type": "disabled",
                            },
                            "enable_web_search": True
                        }
                    },
                    temperature=float(os.getenv("MOSS_LLM_TEMPERATURE", "0.7")),
                ),
                instructions=instructions,
            ),
        )
        main_agent.set_state_hook(moss)
        return main_agent


class AgentHubProvider(Provider[AgentHub]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: Container) -> AgentHub:
        main_agent = con.force_fetch(MainAgent)
        eventbus = con.force_fetch(EventBus)
        agent_hub = AgentHubImpl(
            main_agent_id=main_agent.info().id,
            agents=[main_agent],
            eventbus=eventbus,
            logger=con.get(LoggerItf),
        )
        return agent_hub


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
    storage = ws.runtime().sub_storage("memory")
    memory = StorageMemory(storage)
    session = StorageSession(storage)
    container.set(StorageMemory, memory)
    container.set(Session, session)
    # TodoList
    container.register(TodoListProvider())
    # Shell
    container.register(ShellProvider())
    # Moss
    container.register(MossInReachyMiniProvider())
    # Moss State
    container.register(AsleepStateProvider())
    container.register(WakenStateProvider())
    container.register(BoringStateProvider())
    container.register(LiveStateProvider())


async def run(container):
    container = Container(parent=container, name="main_agent")
    providers(container)
    agent_fastapi = container.make(AgentFastAPI)
    agent_hub = container.make(AgentHub)
    await asyncio.gather(
        agent_hub.bootstrap(),
        setup_chat(agent_hub.eventbus(), ConsolePTTChat(container=container)),
        agent_fastapi.run(),
    )


def get_speech(
    mini: ReachyMini,
    default_speaker: str | None = None,
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
    speech = BaseTTSSpeech(
        tts=VolcengineTTS(conf=tts_conf),
        player=ReachyMiniStreamPlayer(mini, logger=container.get(LoggerItf), recorder=recorder),
    )
    speech.commands = lambda: []
    return speech


async def main():
    import pathlib

    ws_dir = pathlib.Path(__file__).parent.joinpath(".workspace")

    from ghoshell_moss_contrib.example_ws import workspace_container

    with workspace_container(ws_dir) as container:
        logger = setup_logger(str(ws_dir.joinpath("runtime/logs/moss_demo.log").absolute()),)
        container.set(LoggerItf, logger)

        await run(container)

if __name__ == "__main__":
    asyncio.run(main())
