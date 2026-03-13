import asyncio
from typing import Union, TypedDict

from framework.abcd.agent_hub import EventBus
from framework.abcd.agent_event import AgentEvent, AgentEventModel


class QueueEventBus(EventBus):

    def __init__(self) -> None:
        self.queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self.get_callback = None

    async def put(self, event: AgentEvent | AgentEventModel) -> None:
        if isinstance(event, AgentEventModel):
            event = event.to_agent_event()
        await self.queue.put(event)

    async def get(self, timeout: Union[float, None] = None) -> Union[AgentEvent, None]:
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

