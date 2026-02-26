import asyncio
import logging
from typing import List, Optional, AsyncIterator, AsyncIterable

import litellm
from ghoshell_common.contracts import LoggerItf
from ghoshell_moss import Message, MOSSShell, Text, MessageMeta, MessageStage, TextDelta, ContentModel, Addition
from ghoshell_moss.message.adapters.openai_adapter import parse_messages_to_params
from pydantic import Field

from framework.abcd.agent import Response, ModelConf
from framework.abcd.agent_event import UserInputAgentEvent, AgentEventModel


class CTMLResult(ContentModel):
    CONTENT_TYPE = "function_call"

    ctml: str = Field(description="ctml")
    result: str = Field(description="ctml result. ")

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
            *,
            event: AgentEventModel,
            inputs: List[Message],
            model: Optional[ModelConf] = None,
            prompts: List[Message] = None,
            logger: Optional[LoggerItf] = None,
    ):
        self.shell = shell
        self.model = model
        self.prompts = prompts
        self.inputs = inputs

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
                    messages = []
                    moss_instruction = interpreter.moss_instruction()
                    if moss_instruction:
                        messages.append({"role": "system", "content": moss_instruction})

                    # 拼接各类消息
                    messages.extend(parse_messages_to_params(self.prompts))
                    context = interpreter.context_messages()
                    if context:
                        messages.extend(parse_messages_to_params(context))
                    messages.extend(parse_messages_to_params(self.inputted()))

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
                        interpreter.feed(content)

                        # 生成间包
                        delta_msg = Message(
                            meta=MessageMeta(stage=MessageStage.RESPONSE.value, role="assistant")
                        ).with_additions(
                            self._event_addition,
                        ).as_delta(TextDelta(content=content))
                        self._buffered.append(delta_msg)
                        yield delta_msg

                    # 处理完成后的尾包（未中断时）
                    if not self._interrupted.is_set():
                        interpreter.commit()
                        results = await interpreter.results()
                        result_contents = [
                            CTMLResult(ctml=_ctml, result=_result)
                            for _ctml, _result in results.items()
                        ]

                        completed_msg = Message(
                            meta=MessageMeta(stage=MessageStage.RESPONSE.value, role="assistant")
                        ).with_content(
                            Text(text=interpreter.executed_tokens()),
                            *result_contents
                        ).with_additions(
                            self._event_addition,
                        ).as_completed()
                        self._buffered.append(completed_msg)
                        yield completed_msg

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

    def inputted(self) -> List[Message]:
        return self.inputs

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
                AgentEventAddition(event_id=self.event.event_id)
            ).with_content(
                Text(text="[Interrupt] User input")
            )
            self._buffered.append(interrupt_msg)
            self._logger.info(f"MOSShellResponse {self.response_id} interrupted")
        finally:
            self._interrupted_done.set()