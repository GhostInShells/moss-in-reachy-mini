import asyncio
from typing import Union

from framework.abcd.agent import EventBus
from framework.abcd.agent_event import AgentEvent


class QueueEventBus(EventBus):

    def __init__(self) -> None:
        self.queue = asyncio.Queue()

    async def put(self, event: AgentEvent) -> None:
        await self.queue.put(event)

    async def get(self, timeout: Union[float, None] = None) -> Union[AgentEvent, None]:
        try:
            item = await asyncio.wait_for(self.queue.get(), timeout)
            self.queue.task_done()
            return item
        except asyncio.TimeoutError:
            return None
