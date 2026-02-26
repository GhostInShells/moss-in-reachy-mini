import asyncio
import logging
from typing import Union

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Container, Provider, IoCContainer, INSTANCE
from ghoshell_moss import Message, TextDelta
from ghoshell_moss_contrib.agent.chat.base import BaseChat

from framework.abcd.agent import Broadcaster
from framework.agent.utils import InterruptedContent


class LogBroadcaster(Broadcaster):
    def __init__(self, container: Container):
        self.logger = container.get(LoggerItf) or logging.getLogger("LogBroadcaster")

    async def broadcast(self, agent_id: str, message: Union[Message, None]) -> None:
        self.logger.info(f"Agent(id={agent_id}) broadcast message={message}")

    def bootstrap(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class ChatBroadcaster(Broadcaster):
    def __init__(self, chat: BaseChat):
        self._chat = chat

    async def broadcast(self, agent_id: str, message: Union[Message, None]) -> None:
        if message.role == "assistant" and message.seq == "head":
            self._chat.start_ai_response()
        if message.role == "assistant" and message.seq == "delta":
            chunk = TextDelta.from_delta(message.delta)
            if chunk:
                self._chat.update_ai_response(chunk=chunk.content)
        if message.role == "assistant" and message.is_done():
            self._chat.finalize_ai_response()

        if message.role == "system":
            for content in message.contents:
                # 打断完成
                if InterruptedContent.from_content(content):
                    self._chat.finalize_ai_response()

    def bootstrap(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class ChatBroadcasterProvider(Provider[Broadcaster]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        chat = con.force_fetch(BaseChat)
        return ChatBroadcaster(chat=chat)
