import asyncio
import logging
import os
import pathlib

from ghoshell_common.contracts import LoggerItf
from ghoshell_common.contracts.storage import MemoryStorage
from ghoshell_container import Container
from ghoshell_moss import MOSSShell
from ghoshell_moss import new_ctml_shell
from ghoshell_moss_contrib.agent.chat.base import BaseChat
from ghoshell_moss_contrib.example_ws import workspace_container, get_example_speech

from framework.abcd.agent import AgentConfig, ModelConf
from framework.abcd.agent_hub import EventBus, AgentHub
from framework.abcd.session import Session
from framework.agent.agent_fastapi import AgentFastAPI
from framework.agent.agent_hub import AgentHubImpl
from framework.agent.broadcaster import ChatBroadcasterProvider
from framework.agent.eventbus import QueueEventBus
from framework.agent.main_agent import MainAgent
from framework.apps.live.douyin_live import DouyinLiveProvider, DouyinLive
from framework.apps.live.live_agent import LiveAgent
from framework.apps.memory.storage_memory import StorageMemory
from framework.apps.session.storage_session import StorageSession
from framework.apps.utils import AgentConsoleChat
from moss_in_reachy_mini.utils import load_instructions


async def build_main_agent(parent: Container) -> MainAgent:
    container = Container(parent=parent, name="main_agent")

    # broadcaster
    container.set(BaseChat, AgentConsoleChat(agent_id="main"))
    container.register(ChatBroadcasterProvider())

    # memory
    memory = container.force_fetch(StorageMemory)

    # session
    session = StorageSession(MemoryStorage(dir_="session"))
    container.set(Session, session)

    # douyin_live
    douyin_live = container.force_fetch(DouyinLive)

    # shell
    shell = new_ctml_shell(container=container, speech=get_example_speech(container, default_speaker="可爱女生"))
    shell.main_channel.import_channels(
        memory.as_channel(),
        session.as_channel(),
        douyin_live.as_channel(),
    )
    container.set(MOSSShell, shell)

    agent = MainAgent.new(
        container=container,
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
                temperature=0.6
            ),
            instructions="不要使用memory的工具啦，直接输出互动文本",
        ),
    )
    return agent


async def build_live_agent(parent: Container) -> LiveAgent:
    container = Container(parent=parent, name="live_agent")

    # chat
    container.set(BaseChat, AgentConsoleChat(agent_id="live"))
    container.register(ChatBroadcasterProvider())

    # memory
    memory = container.force_fetch(StorageMemory)

    # session
    session = StorageSession(MemoryStorage(dir_="live_session"))
    container.set(Session, session)

    # douyin_live
    douyin_live = container.force_fetch(DouyinLive)

    # shell
    shell = new_ctml_shell(
        name="live_shell",
        container=container,
    )
    shell.main_channel.import_channels(
        memory.as_channel(),
        session.as_channel(),
        douyin_live.as_channel(is_live_agent=True),
    )
    container.set(MOSSShell, shell)
    instructions = load_instructions(
        container,
        files=["ctml_enrich.md", "live_agent_persona.md"],
        storage_name="instructions",
    )
    live_agent = LiveAgent.new(container, AgentConfig(
        id="live",
        name="live",
        description="",
        model=ModelConf(
            base_url="$LIVE_LLM_BASE_URL",
            model="$LIVE_LLM_MODEL",
            api_key="$LIVE_LLM_API_KEY",
            kwargs={
                "extra_body": {
                    "thinking": {
                        "type": "enabled",
                    },
                    "enable_web_search": True
                }
            },
            temperature=float(os.getenv("LIVE_LLM_TEMPERATURE", "0.7")),
        ),
        instructions=instructions,
    ))

    return live_agent


async def main() -> None:
    ws_dir = pathlib.Path(__file__).parent.parent.joinpath("moss_in_reachy_mini/.workspace")
    with workspace_container(ws_dir) as container:
        container.set(LoggerItf, logging.getLogger())
        logging.basicConfig(level=logging.WARNING)

        # 公共的依赖
        container.set(StorageMemory, StorageMemory(MemoryStorage(dir_="memory")))
        eventbus = QueueEventBus()
        container.set(EventBus, eventbus)
        container.register(DouyinLiveProvider())

        main_agent = await build_main_agent(parent=container)
        live_agent = await build_live_agent(parent=container)

        agent_hub = AgentHubImpl(
            main_agent_id="main",
            agents=[
                main_agent,
                live_agent,
            ],
            eventbus=container.force_fetch(EventBus),
            logger=container.get(LoggerItf),
        )
        container.set(AgentHub, agent_hub)

        await agent_hub.bootstrap()

        server = AgentFastAPI(eventbus=eventbus)
        await server.run()


if __name__ == '__main__':
    asyncio.run(main())