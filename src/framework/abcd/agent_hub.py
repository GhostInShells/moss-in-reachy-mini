from abc import abstractmethod, ABC
from typing import Union

from framework.abcd.agent import Agent
from framework.abcd.agent_event import AgentEvent, AgentEventModel


class EventBus(ABC):
    @abstractmethod
    async def put(self, event: AgentEvent | AgentEventModel) -> None:
        """
        将 event 入队.
        """
        pass

    @abstractmethod
    async def get(self, timeout: Union[float, None] = None) -> Union[AgentEvent, None]:
        """
        吐出 event.
        :param timeout: 超时时间, 并不会抛出异常, 如果拿不到会返回 None. 如果超时时间为 None, 则是 get_nowait
        """
        pass

    @abstractmethod
    def on_get(self, callback) -> None:
        pass



class AgentHub(ABC):

    @abstractmethod
    def eventbus(self) -> EventBus:
        pass

    @abstractmethod
    def main_agent_id(self) -> str:
        pass

    @abstractmethod
    def get_agent(self, agent_id: str = "") -> Agent:
        """
        :param agent_id: 为空表示取出主 Agent.
        """
        pass

    @abstractmethod
    async def add_event(self, event: AgentEvent) -> bool:
        """
        同步调用推送事件给 Agent.
        或者用 EventBus, 都一样.
        """
        pass

    @abstractmethod
    async def bootstrap(self):
        """
        初始化主 Agent.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        同步阻塞关闭.
        """
        pass