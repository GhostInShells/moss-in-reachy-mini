import asyncio
import datetime
import logging
import time
from abc import ABC, abstractmethod
from typing import Union, Optional, Self, List

from ghoshell_common.contracts.logger import LoggerItf
from ghoshell_common.contracts.storage import MemoryStorage
from ghoshell_container import Container
from ghoshell_moss import Message, Text, MOSSShell

from framework.abcd.agent import (
    Agent, Identifier, Broadcaster, AgentStateName, AgentConfig, Response, EventBus, ModelConf, AgentId
)
from framework.abcd.agent_event import InterruptAgentEvent, ShutdownAgentEvent, AgentEvent, \
    UserInputAgentEvent, ReactAgentEvent
from framework.abcd.memory import Memory
from framework.agent.eventbus import QueueEventBus
from framework.agent.response import MOSShellResponse, CTMLResult
from framework.agent.utils import get_event, clear_queue, run_agent_with_chat, InterruptedContent



class BaseMainAgent(Agent, ABC):
    """
    具备完整生命周期和AgentHook的一个Agent，未来要被moss.Ghost取代
    """

    def __init__(
            self,
            container: Container,
            config: AgentConfig,
            shell: MOSSShell,
            memory: Memory,
    ):
        self.shell = shell
        self.memory = memory

        self.config = config
        self._id = config.id
        self._state: AgentStateName = AgentStateName.CRATED
        self._container = container
        self._logger = container.get(LoggerItf) or logging.getLogger("MainAgent")

        self._broadcaster = container.get(Broadcaster)
        self._eventbus = container.get(EventBus)
        self._idling_time: float = 0.0
        self._error_time: int = 0

        self._halt_event = asyncio.Event() # 停止所有事件的运行
        self._shutdown_event = asyncio.Event() # 关闭整个 agent 运行

        self._add_event_queue: asyncio.Queue[AgentEvent]= asyncio.Queue()

        self._preempt_event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue() # 抢占式调度的事件队列
        self._queued_event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue() # 普通的事件队列
        self._low_event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue() # 低优先级的事件队列

        self._auto_shutdown: bool = False

        # 生命周期管理.
        self._running_response: Union[Response, None] = None
        self._main_loop_task: Optional[asyncio.Task] = None
        self._add_event_task: Optional[asyncio.Task] = None

    @property
    def logger(self):
        return self._logger

    async def make_prompts(self) -> List[Message]:
        """
        语法糖, 用来快速定义 prompt 对象.
        """
        system_prompt = Message.new(role="system").with_content(
            # instructions
            Text(text=self.config.instructions),
            # env datetime
            Text(text=f"Current datetime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",),
        )
        return [system_prompt]

    @abstractmethod
    def _parse_event(self, event: AgentEvent) -> Union[AgentEvent, None]:
        """
        重写这个事件, 可以做 agent 级别的拦截.
        """
        pass

    async def _inform_error(self, reason: str) -> None:
        error = Message.new(role="system").with_content(
            Text(text=f"Error Occurred: {reason}"),
        )
        await self._broadcaster.broadcast(self._id, error)

    async def _inform_system(self, reason: str) -> None:
        message = Message.new(role="system").with_content(
            Text(text=f"System Notification: {reason}"),
        )
        await self._broadcaster.broadcast(self._id, message)

    def info(self) -> Identifier:
        return Identifier(
            id=self._id,
            name=self.config.name,
            description=self.config.description,
        )

    def state(self) -> AgentStateName:
        return self._state

    async def halt(self, toggle: bool) -> None:
        self._validate_state()
        if toggle:
            await self._interrupt()
            self._halt_event.set()
            self._state = AgentStateName.HALT
            await self._inform_system("agent is halting")
        else:
            self._halt_event.clear()
            self._state = AgentStateName.IDLE
            await self._inform_system("agent recover")
        await self._broadcaster.broadcast(self._id, None)
        self._error_time = 0

    def broadcaster(self) -> Broadcaster:
        return self._broadcaster

    def eventbus(self) -> EventBus:
        return self._eventbus

    async def _idle(self, duration: Union[float, None]) -> None:
        """
        闲置一段时间, 并且会增加闲置状态.
        """
        if duration is None:
            self._idling_time = 0.0
            return
        await asyncio.sleep(duration)
        self._idling_time += duration
        self._state = AgentStateName.IDLE

    async def _handle_event(self, event: AgentEvent) -> Optional[Response]:
        prompts = await self.make_prompts()
        if user_input := UserInputAgentEvent.from_agent_event(event):
            now = time.time()
            # 不处理过期事件.
            if user_input.is_overdue(now):
                self._logger.info(f"agent receive event overdue: {event}")
                return None
            return MOSShellResponse(
                shell=self.shell,
                response_id=user_input.event_id,
                inputs=[user_input.message],
                model=self.config.model,
                prompts=prompts,
                logger=self.logger,
            )

        if react := ReactAgentEvent.from_agent_event(event):
            return MOSShellResponse(
                shell=self.shell,
                response_id=react.event_id,
                inputs=react.messages,
                model=self.config.model,
                prompts=prompts,
                logger=self.logger,
            )

        return None

    def _is_running_response(self, response_id: str) -> bool:
        return self._running_response is not None and self._running_response.response_id == response_id

    async def _handle_response(self, response: Response) -> None:
        self._state = AgentStateName.RESPONDING
        try:
            async with response:
                has_first = False
                async for item in response.stream_messages():
                    if self._shutdown_event.is_set():
                        break
                    # 如果当前正在执行中的 event 被中断了, 则
                    if not has_first:
                        pass
                    has_first = True
                    if not self._is_running_response(response.response_id):
                        break
                    # 广播所有发送的消息. 但有可能有
                    if item is not None:
                        await self._broadcaster.broadcast(self._id, item)
                    self._logger.debug("handling item %s", item)
        except asyncio.CancelledError:
            self._logger.error("handling response cancelled")
        finally:
            await self._finish_response(response)
            self._state = AgentStateName.IDLE
            self._clear_running_response(response.response_id)

    async def _finish_response(self, response: Response) -> None:
        inputs = response.inputted()
        outputs = response.buffered()
        # 判断 outputs 不为空, 就再次保存.
        if inputs or outputs:
             await self.memory.save_turn(inputs, outputs)

        # 如果response output里有拿到ctml执行的返回值，直接触发ReactEvent给Agent继续处理
        if len(outputs) > 0 and outputs[-1].is_completed():
            ctml_results: List[CTMLResult] = []
            for content in outputs[-1].contents:
                if result := CTMLResult.from_content(content):
                    ctml_results.append(result)
            if ctml_results:
                await self.eventbus().put(ReactAgentEvent(
                    messages=[Message.new(role="assistant").with_content(*ctml_results)]
                ).to_agent_event())

    def _clear_running_response(self, response_id: str) -> None:
        if self._running_response and self._running_response.response_id == response_id:
            self._running_response = None

    async def _main_loop(self) -> None:
        """
        执行主循环, 接受所有的事件并处理.
        """
        event_interval = self.config.event_interval
        max_error_time = self.config.max_error_time
        while not self._shutdown_event.is_set():
            # 如果当前是暂停状态, 不会消费任何事件.
            if self._halt_event.is_set():
                await asyncio.sleep(1)
                continue
            try:
                if self._auto_shutdown and 0 < self.config.max_idle_time <= self._idling_time:
                    # 超时自动退出. 可以用于 multi agent 实现.
                    self._shutdown_event.is_set()
                    break
                # 超过异常次数, 停止运行.
                elif self._error_time > max_error_time:
                    await self.halt(True)
                    await self._inform_error("halt because too much errors")
                    continue

                if self._running_response is not None:
                    # 有事件正在运行中, 继续等待.
                    # 这个循环在不断等待事件结束, 或者被中断.
                    await asyncio.sleep(event_interval)
                    continue

                # 高优队列里找
                event = get_event(self._preempt_event_queue)
                if not event:
                    # 普通队列
                    event = get_event(self._queued_event_queue)
                if not event:
                    # 低优队列
                    event = get_event(self._low_event_queue)
                if not event:
                    # 进入下一次 looping.
                    await self._idle(event_interval)
                    continue

                if event:
                    # 为 None 清空等待时间.
                    await self._idle(None)
                    response = await self._handle_event(event)
                    if response is None:
                        continue
                    # 记录运行事件.
                    self._running_response = response
                    await self._handle_response(response)
            except asyncio.QueueEmpty:
                # 队列全部为空?
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.exception(e)
                self._error_time += 1

    async def _add_event_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                e = await self._eventbus.get()
                await self._add_event(e)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.exception(e)

    async def _add_event(self, event: AgentEvent) -> None:
        # 系统级别的事件, 强制关闭 agent 运行.
        if ShutdownAgentEvent.from_agent_event(event):
            await self.close()
            return

        # 中断事件，强制中断当前响应生成
        if InterruptAgentEvent.from_agent_event(event):
            await self._interrupt()
            return

        # 进行事件拦截逻辑.
        parsed = self._parse_event(event)
        if parsed is None:
            return

        if parsed["priority"] < 0:
            # 低优先级事件, 进入普通队列.
            await self._low_event_queue.put(parsed)
            return

        elif parsed["priority"] == 0:
            # 普通事件, 进入正常队列.
            await self._queued_event_queue.put(parsed)
            return

        _running_response = self._running_response
        if _running_response is not None:
            # 相同级别的事件在运行, 也会触发中断.
            if parsed["priority"] >= _running_response.event.priority:
                await self._preempt_event_queue.put(parsed)
                # 清空普通事件队列.
                clear_queue(self._queued_event_queue)
                # 还是要快速中断当前运行的任务.
                # 保存完被中断逻辑.
                await self._interrupt()
                return
            else:
                # 拒绝接受高优先级事件.
                return
        # 高优先级事件如果优先级不够高, 会被忽略掉. 也就是 system is busy.
        await self._preempt_event_queue.put(parsed)
        return

    async def _interrupt(self) -> None:
        """
        同步阻塞的 interrupt 逻辑.
        """
        if self._running_response is not None:
            response = self._running_response
            await response.interrupt()
            self._clear_running_response(response.response_id)
            self._logger.info("interrupt response set")

        # 广播打断完成消息
        await self._broadcaster.broadcast(self._id, Message.new(role="system").with_content(InterruptedContent()))

    def _validate_state(self):
        if not AgentStateName.is_available(self._state):
            raise RuntimeError(f'Agent is not available: {self._state}')

    async def start(self, auto_shutdown: bool) -> None:
        if self._state == AgentStateName.SHUTDOWN:
            raise RuntimeError('Agent is shutting down')
        if self._state == AgentStateName.BOOTSTRAP:
            raise RuntimeError('Agent is already bootstrapped')
        self._state = AgentStateName.BOOTSTRAP
        self._auto_shutdown = auto_shutdown
        self._state = AgentStateName.IDLE

        await self.shell.start()

        self._main_loop_task = asyncio.create_task(self._main_loop())
        self._add_event_task = asyncio.create_task(self._add_event_loop())

        self._logger.info("Agent is now running.")

    async def close(self) -> None:
        if self._state == AgentStateName.SHUTDOWN:
            return
        self._state = AgentStateName.SHUTDOWN
        await self._interrupt()
        self._shutdown_event.set()

        if self._main_loop_task and not self._main_loop_task.done():
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        if self._add_event_task and not self._add_event_task.done():
            self._add_event_task.cancel()
            try:
                await self._add_event_task
            except asyncio.CancelledError:
                pass

        await self.shell.close()

        await self._inform_system("Agent is shutting down")

    async def wait_until_close(self, shutdown: bool) -> None:
        if shutdown:
            await self.close()

        await self._shutdown_event.wait()
        self._logger.info(f"Agent {self._id} quit")


class MainAgent(BaseMainAgent):
    """
    主 Agent.
    """

    def _parse_event(self, event: AgentEvent) -> Union[AgentEvent, None]:
        return event

    @classmethod
    def new(cls, container: Container, config: AgentConfig) -> Self:
        shell = container.force_fetch(MOSSShell)
        memory = container.force_fetch(Memory)
        return cls(container=container, config=config, shell=shell, memory=memory)


async def main(container: Container) -> None:
    agent = MainAgent.new(
        container=container,
        config=AgentConfig(
            id="reachy_mini",
            name="reachy_mini",
            description="",
            model=ModelConf(
                kwargs={
                    "thinking": {
                        "type": "disabled",
                    },
                },
            ),
            instructions=""
        ),
    )
    chat = container.force_fetch(BaseChat)
    await run_agent_with_chat(agent, chat)


if __name__ == '__main__':
    from ghoshell_moss import new_shell
    from framework.agent.storage_memory import StorageMemory
    from framework.agent.broadcaster import ChatBroadcasterProvider
    from ghoshell_moss_contrib.agent.chat.base import BaseChat
    from ghoshell_moss_contrib.agent import ConsoleChat
    from ghoshell_moss.speech import MockSpeech
    _container = Container()
    _container.set(LoggerItf, logging.getLogger())
    logging.basicConfig(level=logging.INFO)

    _memory = StorageMemory(MemoryStorage(dir_=""))
    _container.set(Memory, _memory)
    _shell = new_shell(container=_container, speech=MockSpeech(typing_sleep=0.1))
    _shell.main_channel.import_channels(
        _memory.as_channel()
    )
    _container.set(MOSSShell, _shell)
    _container.set(EventBus, QueueEventBus())
    _container.register(ChatBroadcasterProvider())
    _container.set(BaseChat, ConsoleChat())
    asyncio.run(main(container=_container))