import asyncio
from typing import Optional

from ghoshell_moss import Message, Text, ContentModel
from ghoshell_moss_contrib.agent.chat.base import BaseChat

from framework.abcd.agent import Agent
from framework.abcd.agent_event import AgentEvent, UserInputAgentEvent, InterruptAgentEvent


def get_event(queue: asyncio.Queue) -> Optional[AgentEvent]:
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

async def run_agent_with_chat(agent: Agent, chat: BaseChat) -> None:
    def _callback(user_input):
        asyncio.create_task(agent.eventbus().put(UserInputAgentEvent(
            message=Message.new(role="user").with_content(Text(text=user_input)),
        ).to_agent_event()))

    def _interrupt():
        asyncio.create_task(
            agent.eventbus().put(InterruptAgentEvent().to_agent_event())
        )

    chat.set_input_callback(_callback)
    chat.set_interrupt_callback(_interrupt)
    await agent.start(auto_shutdown=False)
    await chat.run()  # block


class InterruptedContent(ContentModel):
    """
    完成打断
    """
    CONTENT_TYPE = "interrupted"