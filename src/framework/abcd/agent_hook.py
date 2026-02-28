import abc
from typing import Protocol


class AgentHook(abc.ABC):

    @abc.abstractmethod
    async def on_idle(self):
        pass

    @abc.abstractmethod
    async def on_responding(self):
        pass


class AgentHookState(Protocol):

    def get_hook(self) -> AgentHook:
        ...
