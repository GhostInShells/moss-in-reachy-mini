import asyncio
from typing import Union, TypedDict

from framework.abcd.agent_hub import EventBus
from framework.abcd.agent_event import AgentEventModel


class QueueEventBus(EventBus):

    def __init__(self) -> None:
        self.queue = asyncio.Queue()
        self.get_callback = None

    async def put(self, event: AgentEventModel) -> None:
        await self.queue.put(event)

    async def get(self, timeout: Union[float, None] = None) -> Union[AgentEventModel, None]:
        try:
            item = await asyncio.wait_for(self.queue.get(), timeout)
            self.queue.task_done()
            if self.get_callback:
                asyncio.create_task(self.get_callback(item))
            return item
        except asyncio.TimeoutError:
            return None

    def on_get(self, callback) -> None:
        self.get_callback = callback

