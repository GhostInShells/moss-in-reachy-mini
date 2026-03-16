import logging
import re
from typing import Union

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer, INSTANCE
from ghoshell_moss import Message, TextDelta, Text, MessageStage
from ghoshell_moss_contrib.agent.chat.base import BaseChat

from framework.abcd.agent import Broadcaster
from framework.agent.utils import InterruptedContent

# CTML 标签中需要从终端输出中隐藏的 channel 前缀（say 标签的文本内容保留显示）
_HIDDEN_CTML_CHANNELS = {"memory", "reachy_mini", "douyin_live"}


class _CtmlDisplayFilter:
    """流式 CTML 过滤器：将不需要展示给用户的 CTML 标签从终端输出中过滤掉，只保留 <say> 标签内的文本。

    由于 LLM 输出是流式 delta chunk（可能在标签中间断开），需要用简单的状态机跟踪：
    - 在 <say>...</say> 内部 → 输出文本
    - 在其他 CTML 标签内部 → 不输出
    - 在标签外部的纯文本 → 输出
    """

    def __init__(self):
        self._buf = ""  # 累积未完成的 chunk
        self._inside_hidden = False  # 是否在隐藏标签内部

    def reset(self):
        self._buf = ""
        self._inside_hidden = False

    def feed(self, chunk: str) -> str:
        """输入一个 delta chunk，返回应该显示给用户的文本。"""
        self._buf += chunk
        output = []

        while self._buf:
            if self._inside_hidden:
                # 寻找闭合 > （包括自闭合 />) 来结束隐藏
                close_idx = self._buf.find(">")
                if close_idx == -1:
                    # 还没看到闭合，整段都隐藏
                    self._buf = ""
                    break
                # 闭合了，跳过这段
                self._buf = self._buf[close_idx + 1:]
                self._inside_hidden = False
                continue

            # 不在隐藏标签内，寻找下一个 <
            lt_idx = self._buf.find("<")
            if lt_idx == -1:
                # 没有标签，全部输出
                output.append(self._buf)
                self._buf = ""
                break

            # < 之前的纯文本输出
            if lt_idx > 0:
                output.append(self._buf[:lt_idx])
                self._buf = self._buf[lt_idx:]

            # 检查是否有完整标签头部（到第一个空格或 > 或 />）
            # 需要至少看到标签名才能判断是否隐藏
            m = re.match(r"</?([a-zA-Z_][a-zA-Z0-9_:]*)", self._buf)
            if not m:
                # 可能是不完整的 <，等待更多 chunk
                break

            tag_name = m.group(1)
            # 检查是否是需要隐藏的 CTML channel 标签
            channel = tag_name.split(":")[0] if ":" in tag_name else tag_name
            if channel in _HIDDEN_CTML_CHANNELS:
                self._inside_hidden = True
                continue
            elif tag_name == "say" or tag_name == "/say":
                # <say> 和 </say> 标签本身隐藏，内容保留
                close_idx = self._buf.find(">")
                if close_idx == -1:
                    break  # 等待更多 chunk
                self._buf = self._buf[close_idx + 1:]
                continue
            else:
                # 未知标签，原样输出
                output.append("<")
                self._buf = self._buf[1:]
                continue

        return "".join(output)


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
        self._display_filter = _CtmlDisplayFilter()

    async def broadcast(self, agent_id: str, message: Union[Message, None]) -> None:
        if message.role == "assistant" and message.seq == "head":
            self._display_filter.reset()
            self._chat.start_ai_response()
        if message.role == "assistant" and message.seq == "delta":
            chunk = TextDelta.from_delta(message.delta)
            if chunk:
                is_thinking = message.meta.stage == MessageStage.REASONING.value
                display_text = chunk.content if is_thinking else self._display_filter.feed(chunk.content)
                if display_text:
                    self._chat.update_ai_response(chunk=display_text, is_thinking=is_thinking)
        if message.role == "assistant" and message.is_done():
            self._display_filter.reset()
            self._chat.finalize_ai_response()

        if message.role == "system":
            for content in message.contents:
                # 打断完成
                if InterruptedContent.from_content(content):
                    self._display_filter.reset()
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
