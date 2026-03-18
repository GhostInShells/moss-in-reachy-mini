from typing import Union, Self

from ghoshell_container import IoCContainer
from ghoshell_moss import MOSSShell

from framework.abcd.agent import AgentConfig
from framework.abcd.agent_event import AgentEvent
from framework.abcd.agent_hook import AgentHook
from framework.abcd.session import Session
from framework.agent.main_agent import BaseMainAgent


class DecisionAgent(BaseMainAgent):
    """
    Decision Agent.
    """
    def _parse_event(self, event: AgentEvent) -> Union[AgentEvent, None]:
        return event

    @classmethod
    def new(cls, container: IoCContainer, config: AgentConfig) -> Self:
        session = container.force_fetch(Session)
        shell = container.force_fetch(MOSSShell)
        return cls(container=container, config=config, shell=shell, session=session)
