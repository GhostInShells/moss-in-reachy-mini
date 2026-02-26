import abc

class AgentHook(abc.ABC):

    @abc.abstractmethod
    async def on_idle(self):
        pass

    @abc.abstractmethod
    async def on_responding(self):
        pass
