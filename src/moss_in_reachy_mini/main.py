import asyncio
import logging
import os

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Container, get_container
from ghoshell_moss import Speech, MOSSShell
from ghoshell_moss import new_shell
from ghoshell_moss.core.shell.main_channel import create_main_channel
from ghoshell_moss.transports.zmq_channel import ZMQChannelHub
from ghoshell_moss.transports.zmq_channel.zmq_hub import ZMQHubConfig, ZMQProxyConfig
from ghoshell_moss_contrib.agent import ConsoleChat
from ghoshell_moss_contrib.agent.chat.base import BaseChat
from reachy_mini import ReachyMini

from framework.abcd.agent import AgentConfig, ModelConf, EventBus
from framework.abcd.memory import Memory
from framework.agent.broadcaster import ChatBroadcasterProvider
from framework.agent.eventbus import QueueEventBus
from framework.agent.main_agent import MainAgent
from framework.memory.storage_memory import new_ws_storage_memory
from framework.agent.utils import run_agent_with_chat
from framework.memory.storage_memory_server import StorageMemoryAPIServer
from moss_in_reachy_mini.audio.player import ReachyMiniStreamPlayer
from moss_in_reachy_mini.moss import MossInReachyMini
from moss_in_reachy_mini.utils import load_instructions

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def run_agent(container, zmq_hub):
    with ReachyMini() as _mini:
        async with MossInReachyMini(_mini, container) as moss:
            speech = get_speech(_mini, container, default_speaker="saturn_zh_female_keainvsheng_tob")
            # Agent记忆
            memory = new_ws_storage_memory(container)
            container.set(Memory, memory)
            # Shell
            shell = new_shell(container=container, speech=speech, main_channel=create_main_channel())
            shell.main_channel.import_channels(
                moss.as_channel(),
                memory.as_channel(),
                # zmq_hub.as_channel()
            )
            container.set(MOSSShell, shell)
            # Agent输入
            container.set(EventBus, QueueEventBus())
            # Agent输出
            container.register(ChatBroadcasterProvider())
            chat = ConsoleChat()
            container.set(BaseChat, chat)

            instructions = load_instructions(
                container,
                files=["ctml_enrich.md"],
                storage_name="reachy_mini_instructions",
            )
            agent = MainAgent(
                container=container,
                config=AgentConfig(
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
                ),
                shell=shell,
                memory=memory,
                hook=moss.hook(),
            )
            await asyncio.gather(
                StorageMemoryAPIServer(memory).run(),
                run_agent_with_chat(agent, chat),
            )


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


