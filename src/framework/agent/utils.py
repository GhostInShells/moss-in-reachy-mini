import asyncio
from typing import Optional, Self

from ghoshell_moss import Message, Text, ContentModel, Delta, DeltaModel
from ghoshell_moss_contrib.agent.chat.base import BaseChat

from framework.abcd.agent import Agent
from framework.abcd.agent_event import AgentEvent, UserInputAgentEvent, InterruptAgentEvent, AgentEventModel
from framework.abcd.agent_hub import EventBus


def get_event(queue: asyncio.Queue) -> Optional[AgentEventModel]:
    """消费单个队列的一个任务（非阻塞取任务）"""
    try:
        task = queue.get_nowait()
        queue.task_done()
        return task
    except asyncio.QueueEmpty:
        return None

def clear_queue(queue: asyncio.Queue) -> None:
    while not queue.empty():
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            break

async def setup_chat(eventbus: EventBus, chat: BaseChat) -> None:
    loop = asyncio.get_running_loop()

    def _callback(user_input):
        asyncio.run_coroutine_threadsafe(eventbus.put(UserInputAgentEvent(
            message=Message.new(role="user").with_content(Text(text=user_input)),
        )), loop)

    def _interrupt():
        asyncio.run_coroutine_threadsafe(
            eventbus.put(InterruptAgentEvent()),
            loop
        )
    async def _on_get(event: AgentEventModel):
        if event.agent_id not in ["main", ""]:
            return
        if user_input := UserInputAgentEvent.from_agent_event_model(event):

            message_strings = []
            for content in user_input.message.contents:
                if text := Text.from_content(content):
                    message_strings.append(text.text)

            chat.add_user_message("\n".join(message_strings))

    eventbus.on_get(_on_get)

    chat.set_input_callback(_callback)
    chat.set_interrupt_callback(_interrupt)
    await chat.run()  # block


class InterruptedContent(ContentModel):
    """
    完成打断
    """

    CONTENT_TYPE = "interrupted"

    def buffer_delta(self, delta: Delta | DeltaModel) -> bool:
        return False

    @classmethod
    def from_delta(cls, delta: Delta | DeltaModel) -> Self | None:
        return None
