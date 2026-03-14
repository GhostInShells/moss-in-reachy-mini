import asyncio
import json
import logging
import random
from typing import List, Optional, AsyncIterator, AsyncIterable, Dict

import litellm
from ghoshell_common.contracts import LoggerItf
from ghoshell_moss import Message, MOSSShell, Text, MessageMeta, MessageStage, TextDelta, Addition
from ghoshell_moss.message.adapters.openai_adapter import parse_messages_to_params
from pydantic import Field

from framework.abcd.agent import Response, ModelConf
from framework.abcd.agent_event import AgentEventModel, ReactAgentEvent
from framework.abcd.agent_hub import EventBus


class AgentEventAddition(Addition):

    event_id: str = Field(description="event id")
    event_type: str = Field(description="event type")

    @classmethod
    def keyword(cls) -> str:
        return "agent_event"


class MOSShellResponse(Response):
    def __init__(
            self,
            shell: MOSSShell,
            agent_id: str,
            *,
            event: AgentEventModel,
            inputs: List[Message],
            model: Optional[ModelConf] = None,
            prompts: List[Message] = None,
            eventbus: EventBus = None,
            logger: Optional[LoggerItf] = None,
    ):
        self.shell = shell
        self.model = model
        self.prompts = prompts
        self.inputs = inputs
        self.eventbus = eventbus
        self.agent_id = agent_id

        self._logger = logger or logging.getLogger()
        self.event = event
        self._event_addition = AgentEventAddition(
            event_id=event.event_id,
            event_type=event.event_type,
        )
        self.response_id = event.event_id

        # buffered 一定是完整的尾包
        self._buffered: List[Message] = []  # 初始化空列表，避免None导致append报错
        self._interrupted = asyncio.Event()
        self._interrupted_done = asyncio.Event()
        self._interrupt_msg_id = None  # 用于保持消息一致id，防止出现多次消息框
        # 新增：存储生成器任务，用于上下文管理
        self._stream_task: Optional[asyncio.Task] = None

    @property
    def logger(self) -> LoggerItf:
        return self._logger

    # 核心修正：将异步生成器改为实例属性/方法，确保返回AsyncIterable
    def stream_messages(self) -> AsyncIterable[Message]:
        """返回异步可迭代对象，用于遍历消息流（类型正确版）"""

        # 内部封装异步生成器协程
        async def _generator() -> AsyncIterator[Message]:
            if not self.model:
                self.logger.error("Model configuration is missing for stream generation")
                return

            try:
                params = self.model.generate_litellm_params()
                async with self.shell.interpreter_in_ctx() as interpreter:
                    reasoning = False

                    # 构建请求消息列表
                    merged = interpreter.merge_messages(
                        self.prompts,
                        self.inputted()
                    )

                    messages = parse_messages_to_params(merged)
                    with open(f"{self.agent_id}_temp.json", "w", encoding="utf-8") as f:
                        json.dump(messages, f, ensure_ascii=False, indent=4)

                    # 设置流式参数
                    params.update({
                        "messages": messages,
                        "stream": True
                    })

                    # 生成首包
                    head_msg = Message(
                        meta=MessageMeta(stage=MessageStage.RESPONSE.value, role="assistant")
                    ).with_additions(
                        self._event_addition,
                    ).as_head()
                    self._buffered.append(head_msg)
                    yield head_msg

                    # 调用litellm流式接口
                    response_stream = await litellm.acompletion(**params)
                    async for chunk in response_stream:
                        # 检查中断信号
                        if self._interrupted.is_set():
                            self.logger.info("Message stream interrupted by user")
                            break

                        delta = chunk.choices[0].delta
                        self.logger.debug(f"Received delta: {delta}")

                        # 处理推理内容
                        if "reasoning_content" in delta:
                            if not reasoning:
                                reasoning = True
                            reasoning_msg = Message(
                                meta=MessageMeta(stage=MessageStage.REASONING.value, role="assistant")
                            ).with_additions(
                                self._event_addition,
                            ).as_delta(TextDelta(content=delta.reasoning_content))
                            self._buffered.append(reasoning_msg)
                            yield reasoning_msg
                            continue

                        # 处理普通内容
                        content = delta.content
                        if not content:
                            continue
                        # 生成间包
                        delta_msg = Message(
                            meta=MessageMeta(stage=MessageStage.RESPONSE.value, role="assistant")
                        ).with_additions(
                            self._event_addition,
                        ).as_delta(TextDelta(content=content))
                        self._buffered.append(delta_msg)
                        yield delta_msg
                        interpreter.feed(content)

                    # 解释器完成
                    interpreter.commit()
                    interpretation = await interpreter.wait_stopped()
                    # 处理尾包
                    completed_msg = Message(
                        meta=MessageMeta(stage=MessageStage.RESPONSE.value, role="assistant")
                    ).with_content(
                        Text(text=interpreter.executed_tokens()),
                    ).with_additions(
                        self._event_addition,
                    ).as_completed()
                    self._buffered.append(completed_msg)
                    yield completed_msg
                    # 需要被客户端看到的消息
                    for message in interpretation.output_messages():
                        self._buffered.append(message)
                        yield message
                    # 处理观察消息
                    for message in interpretation.execution_messages():
                        self._buffered.append(message)
                    if interpretation.observe:
                        await self.eventbus.put(ReactAgentEvent(
                            messages=interpretation.execution_messages(),
                            priority=1,  # 高优事件
                            agent_id=self.agent_id,
                        ))
            except asyncio.CancelledError:
                self.logger.info("Message stream generator cancelled")
            except Exception as e:
                self.logger.error(f"Error in message stream generator: {e}", exc_info=True)
                raise

        # 返回异步生成器（关键：直接返回生成器对象，而非协程）
        return _generator()

    async def start(self):
        pass

    async def close(self):
        """上下文管理器退出方法"""
        # 清理中断状态
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                self.logger.info("Stream task cancelled")

        # 确保中断完成事件被设置
        self._interrupted_done.set()

    def buffered(self) -> List[Message]:
        """获取缓存的消息列表"""
        return self._buffered.copy()  # 返回副本，避免外部修改

    async def interrupt(self) -> None:
        """中断消息流处理"""
        # 防止重入
        if self._interrupted.is_set():
            return
        try:
            self._interrupted.set()
            self.interrupted = True
            await self.shell.clear()  # 清空shell
            interrupt_msg = Message.new(
                role="system"
            ).with_additions(
                AgentEventAddition(event_id=self.event.event_id, event_type=self.event.event_type),
            ).with_content(
                Text(text="[Interrupt]")
            )
            self._buffered.append(interrupt_msg)
            self._logger.info(f"MOSShellResponse {self.response_id} interrupted")
        finally:
            self._interrupted_done.set()


class CTMLResponse(Response):
    def __init__(
            self,
            shell: MOSSShell,
            agent_id: str,
            *,
            ctml: str,
            event: AgentEventModel,
            eventbus: EventBus = None,
            logger: Optional[LoggerItf] = None,
    ):
        self.shell = shell
        self.agent_id = agent_id
        self.ctml = ctml
        self.eventbus = eventbus
        self.inputs = []

        self._logger = logger or logging.getLogger()
        self.event = event
        self._event_addition = AgentEventAddition(
            event_id=event.event_id,
            event_type=event.event_type,
        )
        self.response_id = event.event_id

        # buffered 一定是完整的尾包
        self._buffered: List[Message] = []  # 初始化空列表，避免None导致append报错
        self._interrupted = asyncio.Event()
        self._interrupted_done = asyncio.Event()
        self._interrupt_msg_id = None  # 用于保持消息一致id，防止出现多次消息框
        # 新增：存储生成器任务，用于上下文管理
        self._stream_task: Optional[asyncio.Task] = None

    @property
    def logger(self) -> LoggerItf:
        return self._logger

    # 核心修正：将异步生成器改为实例属性/方法，确保返回AsyncIterable
    def stream_messages(self) -> AsyncIterable[Message]:
        """返回异步可迭代对象，用于遍历消息流（类型正确版）"""

        # 内部封装异步生成器协程
        async def _generator() -> AsyncIterator[Message]:
            try:
                async with self.shell.interpreter_in_ctx() as interpreter:
                    interpreter.feed(self.ctml)
                    # 解释器完成
                    interpreter.commit()
                    interpretation = await interpreter.wait_stopped()
                    # 处理尾包
                    completed_msg = Message(
                        meta=MessageMeta(stage=MessageStage.RESPONSE.value, role="assistant")
                    ).with_content(
                        Text(text=interpreter.executed_tokens()),
                    ).with_additions(
                        self._event_addition,
                    ).as_completed()
                    self._buffered.append(completed_msg)
                    yield completed_msg
                    # 需要被客户端看到的消息
                    for message in interpretation.output_messages():
                        self._buffered.append(message)
                        yield message
                    # 处理观察消息
                    if interpretation.observe:
                        for message in interpretation.execution_messages():
                            self._buffered.append(message)
                        await self.eventbus.put(ReactAgentEvent(
                            messages=interpretation.execution_messages(),
                            priority=1,  # 高优事件
                            agent_id=self.agent_id,
                        ))
            except asyncio.CancelledError:
                self.logger.info("Message stream generator cancelled")
            except Exception as e:
                self.logger.error(f"Error in message stream generator: {e}", exc_info=True)
                raise

        # 返回异步生成器（关键：直接返回生成器对象，而非协程）
        return _generator()

    async def start(self):
        pass

    async def close(self):
        """上下文管理器退出方法"""
        # 清理中断状态
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                self.logger.info("Stream task cancelled")

        # 确保中断完成事件被设置
        self._interrupted_done.set()

    def buffered(self) -> List[Message]:
        """获取缓存的消息列表"""
        return self._buffered.copy()  # 返回副本，避免外部修改

    async def interrupt(self) -> None:
        """中断消息流处理"""
        # 防止重入
        if self._interrupted.is_set():
            return
        try:
            self._interrupted.set()
            self.interrupted = True
            await self.shell.clear()  # 清空shell
            interrupt_msg = Message.new(
                role="system"
            ).with_additions(
                AgentEventAddition(event_id=self.event.event_id, event_type=self.event.event_type),
            ).with_content(
                Text(text="[Interrupt] User input")
            )
            self._buffered.append(interrupt_msg)
            self._logger.info(f"MOSShellResponse {self.response_id} interrupted")
        finally:
            self._interrupted_done.set()


class QuickResponse(CTMLResponse):
    def __init__(
            self,
            shell: MOSSShell,
            agent_id: str,
            ctml_candidates: List[str],
            *,
            event: AgentEventModel,
            eventbus: EventBus = None,
            logger: Optional[LoggerItf] = None,
    ):
        super().__init__(
            shell=shell,
            agent_id=agent_id,
            ctml=random.choice(ctml_candidates),
            event=event,
            eventbus=eventbus,
            logger=logger,
        )
