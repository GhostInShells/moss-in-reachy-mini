import asyncio
import logging
import os
import pathlib

from ghoshell_common.contracts import LoggerItf
from ghoshell_common.contracts.storage import MemoryStorage
from ghoshell_container import Container
from ghoshell_moss import MOSSShell
from ghoshell_moss import new_ctml_shell
from ghoshell_moss.transports.zmq_channel import ZMQChannelProxy
from ghoshell_moss_contrib.agent.chat.base import BaseChat
from ghoshell_moss_contrib.example_ws import workspace_container, get_example_speech

from framework.abcd.agent import AgentConfig, ModelConf
from framework.abcd.agent_hub import EventBus, AgentHub
from framework.abcd.session import Session
from framework.agent.agent_fastapi import AgentFastAPI
from framework.agent.agent_hub import AgentHubImpl
from framework.agent.broadcaster import ChatBroadcasterProvider, LogBroadcasterProvider
from framework.agent.cognition_agent import CognitionAgent, CognitionSession
from framework.agent.eventbus import QueueEventBus
from framework.agent.main_agent import MainAgent
from framework.agent.utils import setup_chat
from framework.apps.live.douyin_live import DouyinLiveProvider, DouyinLive
from framework.agent.decision_agent import DecisionAgent
from framework.apps.memory.storage_memory import StorageMemory
from framework.apps.news import NewsAPI, NewsAPIProvider
from framework.apps.session.storage_session import StorageSession
from framework.apps.todolist import TodoList, TodoListProvider
from framework.listener.chat.console_ptt import ConsolePTTChat
from moss_in_reachy_mini.utils import load_instructions


# 认知脑：自驱完成一个主题，记录状态
def build_cognition_agent(parent: Container):
    container = Container(parent=parent, name="cognition_agent")

    # chat
    container.register(LogBroadcasterProvider())

    # memory
    memory = container.force_fetch(StorageMemory)

    # session
    cognition_session = CognitionSession(MemoryStorage(dir_="cognition_session"))
    container.set(CognitionSession, cognition_session)

    # todolist
    todolist = container.force_fetch(TodoList)
    news = container.force_fetch(NewsAPI)

    # shell
    shell = new_ctml_shell(
        name="cognition_shell",
        container=container,
        experimental=False,
    )
    shell.main_channel.import_channels(
        memory.as_channel(),
        cognition_session.as_channel(),
        todolist.as_channel(is_main_agent=False),
        news.as_channel(),
    )
    container.set(MOSSShell, shell)

    agent = CognitionAgent.new(
        container=container,
        config=AgentConfig(
            id="cognition",
            name="cognition",
            description="",
            model=ModelConf(
                base_url="$COGNITION_LLM_BASE_URL",
                model="$COGNITION_LLM_MODEL",
                api_key="$COGNITION_LLM_API_KEY",
                kwargs={
                    "extra_body": {
                        "thinking": {
                            "type": "enabled",
                        },
                    },
                },
                temperature=0.6
            ),
            instructions="你是一个旁路在运行任务的agent，你可以看到主agent的历史对话上下文，同时你也有自己独立的历史上下文，你的任务是根据todolist的任务进行处理，每次处理完后要通过mark_as_done将任务结果标记为结束同时将任务的细节告诉给主agent",
        ),
    )
    return agent


async def build_main_agent(parent: Container) -> MainAgent:
    container = Container(parent=parent, name="main_agent")

    # broadcaster
    container.register(ChatBroadcasterProvider())

    # memory
    memory = container.force_fetch(StorageMemory)

    # session
    session = container.force_fetch(StorageSession)

    # douyin_live
    douyin_live = container.force_fetch(DouyinLive)

    reflex_proxy = ZMQChannelProxy(
        name="reflex",
        address="tcp://127.0.0.1:9527",
    )

    # todolist
    todolist = container.force_fetch(TodoList)

    # shell
    shell = new_ctml_shell(container=container, speech=get_example_speech(container, default_speaker="可爱女生"))
    shell.main_channel.import_channels(
        memory.as_channel(),
        session.as_channel(),
        # douyin_live.as_channel(),
        reflex_proxy,
        todolist.as_channel(is_main_agent=True),
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
    agent.ctml_candidates = [
        # "<say>我正在听</say>"
    ]
    return agent


async def build_decision_agent(parent: Container) -> DecisionAgent:
    container = Container(parent=parent, name="decision_agent")

    # chat
    # container.set(BaseChat, AgentConsoleChat(agent_id="live"))
    container.register(LogBroadcasterProvider())

    # memory
    memory = container.force_fetch(StorageMemory)

    # session
    session = StorageSession(MemoryStorage(dir_="decision_session"))
    container.set(Session, session)

    # douyin_live
    douyin_live = container.force_fetch(DouyinLive)

    # shell
    shell = new_ctml_shell(
        name="decision_shell",
        container=container,
    )
    shell.main_channel.import_channels(
        memory.as_channel(),
        session.as_channel(),
        douyin_live.as_channel(is_main_agent=False),
    )
    container.set(MOSSShell, shell)
    instructions = load_instructions(
        container,
        files=["ctml_enrich.md", "decision_agent_persona.md"],
        storage_name="instructions",
    )
    decision_agent = DecisionAgent.new(container, AgentConfig(
        id="decision",
        name="decision",
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

    return decision_agent


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
        # session
        session = StorageSession(MemoryStorage(dir_="session"))
        container.set(StorageSession, session)
        container.set(Session, session)
        # todolist
        container.register(TodoListProvider())
        # news
        container.register(NewsAPIProvider())

        container.set(BaseChat, ConsolePTTChat(eventbus=eventbus))

        main_agent = await build_main_agent(parent=container)
        # decision_agent = await build_decision_agent(parent=container)
        cognition_agent = build_cognition_agent(parent=container)

        agent_hub = AgentHubImpl(
            main_agent_id="main",
            agents=[
                main_agent,
                # decision_agent,
                cognition_agent,
            ],
            eventbus=container.force_fetch(EventBus),
            logger=container.get(LoggerItf),
        )
        container.set(AgentHub, agent_hub)

        server = AgentFastAPI(eventbus=eventbus)

        await asyncio.gather(
            agent_hub.bootstrap(),
            setup_chat(eventbus, container.force_fetch(BaseChat)),
            server.run()
        )


if __name__ == '__main__':
    asyncio.run(main())