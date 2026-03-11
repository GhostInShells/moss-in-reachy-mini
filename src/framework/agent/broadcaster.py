import logging
from typing import Union

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer, INSTANCE
from ghoshell_moss import Message, TextDelta, Text, MessageStage
from ghoshell_moss_contrib.agent.chat.base import BaseChat

from framework.abcd.agent import Broadcaster
from framework.agent.utils import InterruptedContent


class LogBroadcaster(Broadcaster):
    def __init__(self, logger: LoggerItf=None):
        self.logger = logger or logging.getLogger("LogBroadcaster")

    async def broadcast(self, agent_id: str, message: Union[Message, None]) -> None:
        self.logger.debug(f"Agent(id={agent_id}) broadcast message={message}")
        if message.is_done():
            for content in message.contents:
                if text := Text.from_content(content):
                    self.logger.info(f"\n{message.role}: {text.text}\n")

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
                is_thinking = message.meta.stage == MessageStage.REASONING.value
                self._chat.update_ai_response(chunk=chunk.content, is_thinking=is_thinking)
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

    def factory(self, con: IoCContainer) -> Broadcaster:
        chat = con.force_fetch(BaseChat)
        return ChatBroadcaster(chat=chat)


class LogBroadcasterProvider(Provider[Broadcaster]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> Broadcaster:
        logger = con.get(LoggerItf)
        return LogBroadcaster(logger=logger)
