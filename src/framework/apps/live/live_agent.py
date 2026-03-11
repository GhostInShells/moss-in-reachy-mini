from typing import Union, Self

from ghoshell_container import IoCContainer
from ghoshell_moss import MOSSShell

from framework.abcd.agent import AgentConfig
from framework.abcd.agent_event import AgentEventModel
from framework.abcd.session import Session
from framework.agent.main_agent import BaseMainAgent


class LiveAgent(BaseMainAgent):
    """
    Live Agent.
    """
    def _parse_event(self, event: AgentEventModel) -> Union[AgentEventModel, None]:
        return event

    @classmethod
    def new(cls, container: IoCContainer, config: AgentConfig) -> Self:
        session = container.force_fetch(Session)
        shell = container.force_fetch(MOSSShell)
        return cls(container=container, config=config, shell=shell, session=session)
