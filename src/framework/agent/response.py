import asyncio
import json
import logging
import random
from typing import List, Optional, AsyncIterator, AsyncIterable, Dict

import litellm
import openai
from ghoshell_common.contracts import LoggerItf
from ghoshell_moss import Message, MOSSShell, Text, MessageMeta, MessageStage, TextDelta, Addition
from ghoshell_moss.message.adapters.openai_adapter import parse_messages_to_params
from pydantic import Field

from framework.abcd.agent import Response, ModelConf
from framework.abcd.agent_event import AgentEventModel, ReactAgentEvent
from framework.abcd.agent_hub import EventBus



META_INSTRUCTION = """
# MOSS (Model-Oriented Operating System Shell) - Specification - v1.0.0

MOSS 赋予你并行、实时且有序地控制物理世界能力的能力。你通过输出 **CTML (Command Token Marked Language)**
指令来操作系统，这些指令会被系统实时解析并执行。你可以在提供了 MOSS 的环境中基于它的规则与现实世界交互.

## 目的

连接 AI 与物理世界，通过并行、实时、有序的控制逻辑，使你能够调用所有可用能力。

## 核心原则

1. **Code as Prompt**：系统向你展示的是可用命令的精确 `async` Python 函数签名。你的 CTML 调用必须严格匹配这些签名。
1. **Time is First-Class Citizen**：每个命令在物理世界中都有执行时长。你的指令序列规划必须充分考虑这些时间成本。
1. **Structured Concurrency**：
    - **同通道内**：命令按顺序执行（时序阻塞）, 不会重叠执行.
    - **异通道间**：命令并行执行。

## 核心概念

### 命令 (Command)

- 以 Python `async` 函数签名形式呈现，通过 CTML 标签调用。
- 具备执行耗时，会影响同通道内后续命令的启动时间。
- 执行完毕后的返回值（Return Values）将在下一轮交互时传递给你。

### 通道 (Channel)

- 能力的组织单位，类似于 Python 的 module。
- 通道的命名采取 `foo.bar` 的规则, 后文统一用 `channel.path` 代指任意 channel.
- 通道内的命令, 会根据生成顺序 FIFO 执行, 顺序不会错乱.
- **树状结构**：具有父子层级关系，用于实现“漏斗式”的命令下发管理。
- **父子分发**：父通道当前执行阻塞命令时，所有发往该父通道及其所有子通道的新命令都会保持pending，不会分发执行；子通道执行命令不会阻塞父通道的新命令
- **动态信息**：通道会动态提供 `interface`（可用签名）、`instruction`（使用指南）和 `context`（实时状态）。

### 通道能力边界

系统通过以下特定格式的消息在对话历史中展示能力：

- `<ctml_interface>...</ctml_interface>`：包可用的函数签名列表。
- `<ctml_instruction>...<channel name="...">...</channel>...</ctml_instruction>`：展示静态使用指导。
- `<ctml_context>...<channel name="...">...</channel>...</ctml_context>`：展示通道的当前动态上下文讯息.

**ctml_interface/ctml_context 在运行时会动态变更**, 依据你 **最新看到** 的讯息行动.

## CTML

基于 XML 规则的语法，用于描述命令的调用规划, 并且按规划时序流式执行.

- **命名规范**：标签名为 `channel.path:command`。
- **根通道规范**：根通道 `__main__` 的命令不带路径前缀（如 `<wait>`）。**严禁**写成 `<__main__:wait>`
- **自闭合标签**（默认）：`<channel.path:command arg1="value1"/>`。
- **开放-闭合标签**（特殊）：`<channel.path:command arg="value">content</channel.path:command>`。

### 命令参数传递

默认使用 xml 的属性传递参数:

- **解析逻辑**：默认使用 `ast.literal_eval` 解析。复杂引号嵌套使用 `&quot;` 转义.
- **类型歧义**：需要消歧义时可在参数名后加后缀, 如 `arg:str='123'`. 支持 `str|int|float|bool|none|list|dict`.
- **位置参数**：使用特殊属性 `_args`（如 `_args="[1, 2]"`）传递。
- **默认值优化**：当参数值与 interface 中的默认值一致时，应当省略传参。

举例如下:

```
<ctml_interface>
#<channel name="foo">
async def bar(arg1: int, arg2: dict, arg3: str ="foo", arg4: str = "baz")
  '''docstring'''
#</channel>
</ctml_interface>
```

```ctml
<foo:bar _args="[123]" arg2="{'a': 'b'}" arg3="'bar'"/>  # 等价于 foo(123, arg2={'a': 'b'}, arg3='bar', arg4='baz')
```

### 开标记规则与特殊参数类型

命令调用默认只允许用自闭合标记, **当且仅当包含以下参数时, 必须使用 开放-闭合标签传递**:

- `text__`：纯文本字符串。
- `chunks__`：流式文本（异步迭代器），用于逐字输出。
- `ctml__`：流式命令（异步迭代器），用于生成并执行动态 CTML。
- **调用方式**：只需在开闭标签间直接输出文本，MOSS 会自动将其封装为对应类型。
- 这类参数 **必须**使用开闭标签。禁止将这些特殊参数作为属性传递。
- **分形嵌套**: 只有 `ctml__` 允许嵌套 ctml, `text__` 和 `chunks__` **不能** 嵌套 Command.
- **Escape**: `text__` 和 `chunks__` 长度较长时, 在开放-闭合标记里用 `<![CDATA[ ]]>` 包裹内容, 避免出现类似 xml 的内容引起错误.
- **开闭标记必须闭合**: 使用开闭标记时, 记住一定要正确的位置闭合它.

### 命令的返回值与实例化

你通过 CTML 下发的命令会被 Shell 执行, 执行完毕后:

* 如果 command 有返回值或异常, 会以 `<result command="channel.path:command:id">...</result>`的形式通过后续消息发送.
    - 通过 `_id` 属性可以对命令调用实例化：`<channel.path:command _id='1'>`。用于区分同名命令的返回值, 用自增整数定义.
* 如果 command 没有返回值, 或者被正常取消, 会记录完成数量.
* 未结束的命令, 会标记 `queued/pending/executing` 等状态.

### 通道作用域

CTML 支持关键的通道作用域语法 `<_ channel until timeout >...</_>`. 其中 `_` 代表 `scope`, 避免与 Channel 函数重名.

作用域由属性:

- `channel: str = ''`: 必须指定 channel 完整路径, 默认值是根轨道 '__main__'.
- `until: Literal['self', 'all', 'any'] = 'self'`:
    - `self`: 当 scope 绑定的通道的本层队列内所有阻塞命令执行完毕时，立即取消该 scope 内所有未完成的子通道命令 / 作用域
    - `all`: 当 scope 本层内 **所有的阻塞命令** 执行完毕后才结束.
    - `any`: 当 scope 本层内 **任意一个阻塞命令** 执行完毕后, 取消未完成命令.
- `timeout: float | None = None`: 单位是秒, 超时后通道内所有的命令会被中断和丢弃.

嵌套规则:

* 嵌套作用域如果指定非当前通道，必须是当前通道的子通道
* 允许同通道嵌套多个分阶段作用域.
* 同级多通道并行控制是允许的，只要都属于当前通道的子通道即可
"""


class AgentEventAddition(Addition):

    event_id: str = Field(description="event id")
    event_type: str = Field(description="event type")
    agent_id: str = Field(description="agent id")

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
            agent_id=agent_id,
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
                api_key, base_url, params = self.model.generate_openapi_params()
                async with self.shell.interpreter_in_ctx(meta_instruction=META_INSTRUCTION) as interpreter:
                    reasoning = False

                    # 构建请求消息列表
                    merged = interpreter.merge_messages(
                        self.prompts,
                        self.inputted()
                    )

                    messages = parse_messages_to_params(merged)

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

                    json.dump(params, open(f"{self.agent_id}_params.json", "w"), ensure_ascii=False, indent=2)

                    # 调用litellm流式接口
                    # litellm._turn_on_debug()
                    async with openai.AsyncClient(
                        api_key=api_key,
                        base_url=base_url,
                    ) as client:
                        response_stream = await client.chat.completions.create(**params)
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
                            Text(text="".join(interpretation.feed_inputs)),
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
                AgentEventAddition(event_id=self.event.event_id, event_type=self.event.event_type, agent_id=self.agent_id),
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
            agent_id=self.agent_id,
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
        return []
        # return self._buffered.copy()  # 返回副本，避免外部修改

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
                AgentEventAddition(event_id=self.event.event_id, event_type=self.event.event_type, agent_id=self.agent_id),
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
