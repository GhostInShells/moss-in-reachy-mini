import asyncio
import os

from ghoshell_common.contracts import LoggerItf, Workspace
from ghoshell_container import Container, IoCContainer, get_container
from ghoshell_moss import MOSSShell, new_ctml_shell
from ghoshell_moss.speech import BaseTTSSpeech
from ghoshell_moss.transports.zmq_channel import ZMQChannelProxy
from ghoshell_moss_contrib.agent.chat.base import BaseChat
from reachy_mini import ReachyMini

from framework.abcd.agent import AgentConfig, ModelConf
from framework.abcd.agent_hub import AgentHub, EventBus
from framework.abcd.session import Session
from framework.agent.agent_fastapi import AgentFastAPI, AgentFastAPIProvider
from framework.agent.agent_hub import AgentHubImpl
from framework.agent.broadcaster import ChatBroadcasterProvider, LogBroadcasterProvider
from framework.agent.eventbus import QueueEventBus
from framework.agent.main_agent import MainAgent
from framework.agent.utils import setup_chat
from framework.apps.agent_task import AgentTaskChannel, AgentTaskChannelProvider
from framework.apps.live.douyin_live import DouyinLive, DouyinLiveProvider
from framework.agent.decision_agent import DecisionAgent, DecisionSession, DecisionAgentHookProvider
from framework.apps.memory.storage_memory import StorageMemory
from framework.apps.session.storage_session import StorageSession
from framework.apps.utils import AgentConsoleChat
from framework.apps.volc_websearch import VolcWebsearchChannel
from framework.listener.chat.console_ptt import ConsolePTTChat
from framework.moss_contrib.ctml_repo import CtmlRepo, CtmlRepoProvider
from moss_in_reachy_mini.audio.mic_hub import MicHubProvider
from moss_in_reachy_mini.audio.player import ReachyMiniStreamPlayer
from moss_in_reachy_mini.audio.mixer import AudioMixerProvider
from moss_in_reachy_mini.camera.camera_worker import CameraWorkerProvider
from moss_in_reachy_mini.camera.frame_hub import FrameHubProvider
from moss_in_reachy_mini.components.antennas import AntennasProvider
from moss_in_reachy_mini.components.body import BodyProvider
from moss_in_reachy_mini.components.head import HeadProvider
from moss_in_reachy_mini.components.head_tracker import HeadTrackerProvider
from moss_in_reachy_mini.components.music import MusicSearchProvider
from moss_in_reachy_mini.components.sound import SoundProvider
from moss_in_reachy_mini.components.vision import VisionProvider
from moss_in_reachy_mini.logger import setup_logger
from moss_in_reachy_mini.moss import MossInReachyMini, MossInReachyMiniProvider
from moss_in_reachy_mini.state import AsleepStateProvider, BoringStateProvider, LiveStateProvider, WakenStateProvider
from moss_in_reachy_mini.state.enrolling import EnrollingStateProvider
from moss_in_reachy_mini.state.teaching import TeachingState, TeachingStateProvider
from moss_in_reachy_mini.utils import load_instructions
from moss_in_reachy_mini.video.recorder_worker import VideoRecorderWorker, VideoRecorderWorkerProvider

MEMORY = os.getenv("REACHY_MINI_MEMORY", "memory")


# 决策脑：分析当前所有事件然后给主脑递小纸条
def build_decision_agent(parent: Container, agent_id: str) -> DecisionAgent:
    container = Container(parent=parent, name=agent_id)

    # chat
    container.set(BaseChat, AgentConsoleChat(agent_id=agent_id))
    container.register(ChatBroadcasterProvider())

    # memory
    memory = container.force_fetch(StorageMemory)

    # session
    ws = container.force_fetch(Workspace)
    storage = ws.runtime().sub_storage(MEMORY).sub_storage("decision_agent_sessions")
    decision_session = DecisionSession(storage)
    container.set(DecisionSession, decision_session)

    # moss_in_reachy_mini
    moss = container.force_fetch(MossInReachyMini)

    # agent task
    agent_task_chan = container.force_fetch(AgentTaskChannel)

    # decision agent hook
    container.register(DecisionAgentHookProvider(decision_agent_id=agent_id))

    # reflex proxy
    reflex_gui_proxy = ZMQChannelProxy(
        name="gui",
        address="tcp://127.0.0.1:9527",
        recv_timeout=3,
        send_timeout=3,
    )

    # websearch
    websearch_chan = container.force_fetch(VolcWebsearchChannel)

    # shell
    shell = new_ctml_shell(
        name=f"{agent_id}_shell",
        container=container,
        experimental=False,
    )
    shell.main_channel.import_channels(
        memory.as_channel(),
        decision_session.as_channel(),
        # moss.as_channel(only_context_messages=True),
        agent_task_chan,
        reflex_gui_proxy,
        websearch_chan,
    )
    if os.getenv("REACHY_MINI_MODE") == "live":
        # douyin_live
        douyin_live = container.force_fetch(DouyinLive)
        shell.main_channel.import_channels(douyin_live.as_channel())

    container.set(MOSSShell, shell)
    instructions = load_instructions(
        container,
        files=[
            "decision_agent/instructions.md",
            "decision_agent/give_cues_ctml_guideline.md",
            "decision_agent/gui_ctml_guideline.md",
        ],
        storage_name="instructions",
    )
    decision_agent = DecisionAgent.new(
        container,
        AgentConfig(
            id=agent_id,
            name=agent_id,
            description="",
            model=ModelConf(
                base_url="$DECISION_LLM_BASE_URL",
                model="$DECISION_LLM_MODEL",
                api_key="$DECISION_LLM_API_KEY",
                kwargs={
                    "extra_body": {
                        "thinking": {
                            "type": "enabled",
                        },
                    }
                },
                temperature=float(os.getenv("DECISION_LLM_TEMPERATURE", "0.7")),
            ),
            instructions=instructions,
        ),
    )

    return decision_agent


# 主脑：交互
async def build_main_agent(parent: Container, agent_id: str) -> MainAgent:
    container = Container(parent=parent, name=agent_id)

    # Agent输出
    container.register(ChatBroadcasterProvider())

    # Agent记忆
    memory = container.force_fetch(StorageMemory)

    # 会话记录
    session = container.force_fetch(StorageSession)

    # websearch
    websearch_chan = container.force_fetch(VolcWebsearchChannel)

    # agent task
    agent_task_chan = container.force_fetch(AgentTaskChannel)

    # Shell
    mini = container.force_fetch(ReachyMini)
    moss = container.force_fetch(MossInReachyMini)
    shell = new_ctml_shell(
        container=container,
        speech=get_speech(
            mini,
            default_speaker="可爱女生",
            container=container,
        ),
        experimental=False,
    )

    shell.main_channel.import_channels(
        moss.as_channel(),
        memory.as_channel(),
        session.as_channel(),
        websearch_chan,
        agent_task_chan,
    )
    if os.getenv("REACHY_MINI_MODE") == "live":
        # douyin_live
        douyin_live = container.force_fetch(DouyinLive)
        shell.main_channel.import_channels(douyin_live.as_channel())

    ctml_repo = container.force_fetch(CtmlRepo)
    shell.main_channel.build.command(
        available=moss.is_available_fn(TeachingState.NAME),  # 只允许示教模式来用这个command
    )(ctml_repo.save_ctml)
    shell.main_channel.build.command(
        doc=ctml_repo.execute_ctml_docstring,  # 动态加载docstring
    )(ctml_repo.execute_ctml)

    container.set(MOSSShell, shell)

    # Agent
    instructions = load_instructions(
        container,
        files=[
            # "memory_rules.md",
            "system_rules.md",
            # "main_agent/instructions.md",
        ],
        storage_name="instructions",
    )
    main_agent = MainAgent.new(
        container=container,
        config=AgentConfig(
            id=agent_id,
            name=agent_id,
            description="",
            model=ModelConf(
                kwargs={
                    "extra_body": {
                        "thinking": {
                            "type": "disabled",
                        },
                    }
                },
                temperature=float(os.getenv("MOSS_LLM_TEMPERATURE", "0.7")),
            ),
            instructions=instructions,
        ),
    )
    main_agent.set_state_hook(moss)
    return main_agent


def common_dependencies(container: IoCContainer):
    container.register(AgentFastAPIProvider())
    # AgentHub Eventbus
    container.set(EventBus, QueueEventBus())
    # Agent共同记忆
    ws = container.force_fetch(Workspace)
    storage = ws.runtime().sub_storage(MEMORY)
    memory = StorageMemory(storage)
    container.set(StorageMemory, memory)

    # 主Agent会话
    ws = container.force_fetch(Workspace)
    storage = ws.runtime().sub_storage(MEMORY).sub_storage("main_agent_sessions")
    session = StorageSession(storage)
    container.set(StorageSession, session)
    container.set(Session, session)

    # 默认Agent输出
    chat = ConsolePTTChat(container=container)
    container.set(BaseChat, chat)

    # DouyinLive
    container.register(DouyinLiveProvider())

    # Websearch
    container.set(VolcWebsearchChannel, VolcWebsearchChannel(
        name="websearch",
        description="A channel for web search",
        api_key=os.environ["VOLC_WEBSEARCH_API_KEY"],
    ))

    # agent task
    container.register(AgentTaskChannelProvider(
        name="task",
        description="擅长处理复杂任务，所有复杂任务都可以代理给这个channel来异步完成",
        instructions="你是一个任务助手，你需要通过给你的目标，规划路径并拆解子任务，最终完成这个目标，你的每个任务内容都需要用文件存储起来",
        agent_id="decision",
    ))

    # ctml repo
    container.register(CtmlRepoProvider())

    # Mini
    container.set(ReachyMini, ReachyMini())

    # dependency registry
    container.register(BodyProvider())
    container.register(HeadProvider())
    container.register(AntennasProvider())
    container.register(VisionProvider())
    container.register(HeadTrackerProvider())
    container.register(SoundProvider())
    container.register(MusicSearchProvider())
    container.register(CameraWorkerProvider())
    container.register(VideoRecorderWorkerProvider())
    container.register(FrameHubProvider())
    # Shared microphone capture (avoid multi-stream conflicts)
    container.register(MicHubProvider())

    # Shared audio output mixer (avoid multi-producer conflicts between TTS and play_sound)
    # Mixer is lazy-started; safe to register at boot.
    container.register(AudioMixerProvider())
    # Moss State
    container.register(AsleepStateProvider())
    container.register(WakenStateProvider())
    container.register(BoringStateProvider())
    container.register(LiveStateProvider())
    container.register(TeachingStateProvider())
    container.register(EnrollingStateProvider())

    # Moss
    container.register(MossInReachyMiniProvider())


async def run(container):
    # 公共依赖
    common_dependencies(container)

    agents = []
    # 主Agent
    main_agent = await build_main_agent(container, "main")
    agents.append(main_agent)

    # 决策Agent
    # decision_agent = build_decision_agent(container, "decision")
    # agents.append(decision_agent)

    # AgentHub
    eventbus = container.force_fetch(EventBus)
    agent_hub = AgentHubImpl(
        main_agent_id=main_agent.info().id,
        agents=agents,
        eventbus=eventbus,
        logger=container.get(LoggerItf),
    )
    container.set(AgentHub, agent_hub)

    # HTTP API
    agent_fastapi = container.make(AgentFastAPI)
    await asyncio.gather(
        agent_hub.bootstrap(),
        setup_chat(eventbus, container.force_fetch(BaseChat)),
        agent_fastapi.run(),
    )


def get_speech(
    mini: ReachyMini,
    default_speaker: str | None = None,
    container: IoCContainer = None,
) -> BaseTTSSpeech:
    from ghoshell_moss.speech.volcengine_tts import VolcengineTTS, VolcengineTTSConf

    container = container or get_container()
    try:
        from moss_in_reachy_mini.audio.mixer import AudioMixer

        mixer = container.force_fetch(AudioMixer)
    except Exception:
        mixer = None
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
        player=ReachyMiniStreamPlayer(
            mini,
            logger=container.get(LoggerItf),
            recorder=recorder,
            mixer=mixer,
        ),
    )
    # speech.commands = lambda: []
    return speech


async def main():
    import pathlib

    ws_dir = pathlib.Path(__file__).parent.joinpath(".workspace")

    from ghoshell_moss_contrib.example_ws import workspace_container

    with workspace_container(ws_dir) as container:
        logger = setup_logger(
            str(ws_dir.joinpath("runtime/logs/moss_demo.log").absolute()),
        )
        container.set(LoggerItf, logger)

        await run(container)


if __name__ == "__main__":
    asyncio.run(main())
