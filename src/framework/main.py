import asyncio
import logging
import os
import pathlib

from ghoshell_common.contracts import LoggerItf
from ghoshell_common.contracts.storage import MemoryStorage
from ghoshell_container import Container
from ghoshell_moss import MOSSShell, Message, Text
from ghoshell_moss import new_ctml_shell
from ghoshell_moss.transports.zmq_channel import ZMQChannelProxy
from ghoshell_moss_contrib.agent.chat.base import BaseChat
from ghoshell_moss_contrib.example_ws import workspace_container, get_example_speech

from framework.abcd.agent import AgentConfig, ModelConf
from framework.abcd.agent_event import ProgramInputAgentEvent
from framework.abcd.agent_hook import AgentHook
from framework.abcd.agent_hub import EventBus, AgentHub
from framework.abcd.session import Session
from framework.agent.agent_fastapi import AgentFastAPI
from framework.agent.agent_hook import BaseAgentHook
from framework.agent.agent_hub import AgentHubImpl
from framework.agent.broadcaster import ChatBroadcasterProvider, LogBroadcasterProvider
from framework.agent.eventbus import QueueEventBus
from framework.agent.main_agent import MainAgent
from framework.agent.utils import setup_chat
from framework.apps.agent_task import AgentTaskChannelProvider, AgentTaskChannel
from framework.apps.live.douyin_live import DouyinLiveProvider, DouyinLive
from framework.agent.decision_agent import DecisionAgent, DecisionSession, DecisionAgentHookProvider
from framework.apps.memory.storage_memory import StorageMemory
from framework.apps.session.storage_session import StorageSession
from framework.apps.utils import AgentConsoleChat
from framework.apps.volc_websearch import VolcWebsearchChannel
from framework.listener.chat.console_ptt import ConsolePTTChat
from moss_in_reachy_mini.utils import load_instructions


class DriveSelfState(BaseAgentHook):

    NAME = "live"
    out_switchable = False

    def __init__(
            self,
            eventbus: EventBus,
            douyin: DouyinLive,
            logger: LoggerItf=None,
    ):
        super().__init__()
        self.logger = logger or logging.getLogger("LiveState")
        self.eventbus = eventbus
        self.douyin = douyin

    def get_hook(self) -> AgentHook:
        return self

    async def on_self_enter(self):
        pass

    async def on_self_exit(self):
        pass

    async def _run_idle_move(self):
        if self._idle_move_duration > 5:
            message = Message.new(role="user", name="__drive_self__")
            message.with_content(
                Text(text=self.douyin.config.idle_think_prompt)
            )
            if not message.is_empty():
                await self.eventbus.put(ProgramInputAgentEvent(
                    message=message,
                    agent_id="", # 默认主脑
                    priority=0,  # 普通队列，可被高优事件打断
                    overdue=20,
                ))


async def build_main_agent(parent: Container, agent_id: str) -> MainAgent:
    container = Container(parent=parent, name="main_agent")

    # broadcaster
    container.register(ChatBroadcasterProvider())

    # memory
    memory = container.force_fetch(StorageMemory)

    # session
    session = container.force_fetch(StorageSession)

    # douyin_live
    douyin_live = container.force_fetch(DouyinLive)

    # websearch
    websearch_chan = container.force_fetch(VolcWebsearchChannel)

    # agent task
    agent_task_chan = container.force_fetch(AgentTaskChannel)

    # shell
    shell = new_ctml_shell(
        container=container,
        # speech=get_example_speech(container, default_speaker="可爱女生"),
        experimental=False,
    )
    shell.main_channel.import_channels(
        memory.as_channel(),
        session.as_channel(),
        douyin_live.as_channel(),
        websearch_chan,
        agent_task_chan
    )
    container.set(MOSSShell, shell)

    instructions = load_instructions(
        container,
        files=["main_agent/instructions.md", "memory_rules.md"],
        storage_name="instructions",
    )

    agent = MainAgent.new(
        container=container,
        config=AgentConfig(
            id=agent_id,
            name="main",
            description="",
            model=ModelConf(
                kwargs={
                    "extra_body": {
                        "thinking": {
                            "type": "disabled",
                        },
                    }
                },
                temperature=0.6
            ),
            instructions=instructions,
        ),
    )
    eventbus = container.force_fetch(EventBus)
    agent.set_state_hook(DriveSelfState(eventbus=eventbus, douyin=douyin_live))
    agent.ctml_candidates = [
        # "<say>我正在听</say>"
    ]
    return agent


async def build_decision_agent(parent: Container, agent_id: str) -> DecisionAgent:
    container = Container(parent=parent, name="decision_agent")

    # chat
    container.set(BaseChat, AgentConsoleChat(agent_id="decision"))
    container.register(ChatBroadcasterProvider())

    # memory
    memory = container.force_fetch(StorageMemory)

    # session
    decision_session = DecisionSession(MemoryStorage(dir_="decision_session"))
    container.set(DecisionSession, decision_session)

    # douyin_live
    douyin_live = container.force_fetch(DouyinLive)

    # websearch
    websearch_chan = container.force_fetch(VolcWebsearchChannel)

    # agent task
    agent_task_chan = container.force_fetch(AgentTaskChannel)

    # decision agent hook
    container.register(DecisionAgentHookProvider(decision_agent_id=agent_id))

    reflex_proxy = ZMQChannelProxy(
        name="reflex",
        address="tcp://127.0.0.1:9527",
    )

    # shell
    shell = new_ctml_shell(
        name="decision_shell",
        container=container,
        experimental=False,
    )
    shell.main_channel.import_channels(
        memory.as_channel(),
        decision_session.as_channel(),
        douyin_live.as_channel(),
        websearch_chan,
        agent_task_chan,
        reflex_proxy,
    )
    container.set(MOSSShell, shell)
    instructions = load_instructions(
        container,
        files=["decision_agent/instructions.md", "decision_agent/give_cues_ctml_guideline.md"],
        storage_name="instructions",
    )
    decision_agent = DecisionAgent.new(container, AgentConfig(
        id=agent_id,
        name="decision",
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
    ))

    return decision_agent


async def main() -> None:

    main_agent_id = "main"
    decision_agent_id = "decision"

    ws_dir = pathlib.Path(__file__).parent.parent.joinpath("moss_in_reachy_mini/.workspace")
    with workspace_container(ws_dir) as container:
        container.set(LoggerItf, logging.getLogger())
        logging.basicConfig(level=logging.WARNING)

        # 公共的依赖
        container.set(StorageMemory, StorageMemory(MemoryStorage(dir_="memory")))
        eventbus = QueueEventBus()
        container.set(EventBus, eventbus)
        container.register(DouyinLiveProvider())

        # main session
        session = StorageSession(MemoryStorage(dir_="session"))
        container.set(StorageSession, session)
        container.set(Session, session)

        # websearch
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
            agent_id=decision_agent_id,
        ))

        # 控制台语音交互
        container.set(BaseChat, ConsolePTTChat(eventbus=eventbus))

        main_agent = await build_main_agent(parent=container, agent_id=main_agent_id)
        decision_agent = await build_decision_agent(parent=container, agent_id=decision_agent_id)

        agent_hub = AgentHubImpl(
            main_agent_id=main_agent_id,
            agents=[
                main_agent,
                # decision_agent,
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