import asyncio
import logging
from typing import Dict, Optional, List

from ghoshell_common.contracts import LoggerItf

from framework.abcd.agent import Agent, AgentStateName
from framework.abcd.agent_event import AgentEventModel
from framework.abcd.agent_hub import AgentHub, EventBus
from framework.agent.eventbus import QueueEventBus


class AgentHubImpl(AgentHub):
    def __init__(
            self,
            *,
            main_agent_id: str,
            agents: List[Agent],
            eventbus: EventBus=None,
            logger: LoggerItf=None,
    ):
        self._logger = logger or logging.getLogger("AgentHubImpl")

        # main agent
        self._main_agent_id = main_agent_id
        self._event_queue = asyncio.Queue()

        if eventbus is None:
            eventbus: EventBus = QueueEventBus()
        self.eventbus = eventbus

        # 每个 Agent Process 都需要有一个主 Agent.
        self._agent_instances: Dict[str, Agent] = {
            a.info().id: a for a in agents
        }

        self._bootstrapping: bool = False
        self._shutdown_event = asyncio.Event()
        self._main_event_loop_task: Optional[asyncio.Task] = None
        self._bootstrapped: bool = False
        self._shutting_down: bool = False

    def eventbus(self) -> EventBus:
        return self.eventbus

    def main_agent_id(self) -> str:
        return self._main_agent_id

    def get_agent(self, agent_id: str = "") -> Agent:
        self._validate_bootstrapped()

        if agent_id == "":
            agent_id = self._main_agent_id

        agent_ins = self._agent_instances[agent_id]
        if not agent_ins:
            raise ValueError(f"agent {agent_id} not found")

        return agent_ins

    async def add_event(self, event: AgentEventModel) -> bool:
        self._logger.info(f'AGENT HUB ADD EVENT: {event.model_dump()}')
        agent_id = event.agent_id
        agent = self.get_agent(agent_id)
        if agent is None:
            return False
        self._logger.info(
            'add event type %s id %s to agent %s',
            event.event_type, event.event_id, agent.info().id,
        )
        await agent.add_event(event)
        return True

    async def _main_event_loop(self) -> None:
        """
        不断从 eventbus 里接受消息, 然后通过 add_event 分发.
        """
        while not self._shutdown_event.is_set():
            try:
                event = await self.eventbus.get()
                if event is None:
                    continue
                self._logger.info("AgentHub receive async event %s for agent %s", event.event_id, event.agent_id)
                await self.add_event(event)
            except Exception as e:
                # 记录异常但不中断.
                self._logger.exception(e)

    def _validate_bootstrapped(self):
        if not self._bootstrapped:
            raise RuntimeError(f'AgentHub does not bootstrap')

    async def bootstrap(self) -> None:
        if self._bootstrapping:
            return
        self._bootstrapping = True

        for _, agent in self._agent_instances.items():
            await agent.start(auto_shutdown=False)

        self._main_event_loop_task = asyncio.create_task(self._main_event_loop())
        self._bootstrapped = True

    async def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._shutdown_event.set()
        # 先下达异步关闭指令.
        self._logger.info(f"Agent Hub {self._main_agent_id} shutdown")
        for agent in self._agent_instances.values():
            await agent.close()
        # 阻塞等待所有的 Agent 退出.
        for agent in self._agent_instances.values():
            await agent.wait_until_close(shutdown=False)
        self._main_event_loop_task.cancel()
        self._logger.info(f"Agent Hub {self._main_agent_id} quit")


