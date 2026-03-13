import asyncio
import datetime
import logging
import time
from abc import ABC, abstractmethod
from typing import Union, Optional, Self, List

from ghoshell_common.contracts.logger import LoggerItf
from ghoshell_container import Container, Provider, IoCContainer, INSTANCE
from ghoshell_moss import Message, Text, MOSSShell

from framework.abcd.agent import (
    Agent, Identifier, Broadcaster, AgentStateName, AgentConfig, Response, ModelConf
)
from framework.abcd.agent_event import InterruptAgentEvent, ShutdownAgentEvent, \
    UserInputAgentEvent, ReactAgentEvent, VisionAgentEvent, CTMLAgentEvent, AgentEventModel, ResumeAgentEvent
from framework.abcd.agent_hook import AgentStateHook
from framework.abcd.agent_hub import EventBus
from framework.abcd.session import Session
from framework.agent.response import MOSShellResponse, CTMLResponse
from framework.agent.utils import get_event, InterruptedContent


class BaseMainAgent(Agent, ABC):
    """
    具备完整生命周期和AgentHook的一个Agent，未来要被moss.Ghost取代
    """

    def __init__(
            self,
            container: IoCContainer,
            config: AgentConfig,
            shell: MOSSShell,
            session: Session,
            state_hook: AgentStateHook=None,
    ):
        self.shell = shell
        self.session = session
        self.state_hook = state_hook

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

        self._add_event_queue: asyncio.Queue[AgentEventModel]= asyncio.Queue()

        self._preempt_event_queue: asyncio.Queue[AgentEventModel] = asyncio.Queue() # 抢占式调度的事件队列
        self._queued_event_queue: asyncio.Queue[AgentEventModel] = asyncio.Queue() # 普通的事件队列
        self._low_event_queue: asyncio.Queue[AgentEventModel] = asyncio.Queue() # 低优先级的事件队列

        self._auto_shutdown: bool = False

        # 生命周期管理.
        self._running_response: Union[Response, None] = None
        self._main_loop_task: Optional[asyncio.Task] = None
        self._add_event_task: Optional[asyncio.Task] = None

    @property
    def logger(self):
        return self._logger

    def set_state_hook(self, state_hook: AgentStateHook):
        self.state_hook = state_hook

    # 异步方法替代 setter，明确标识异步操作
    async def set_state(self, value: AgentStateName):
        if self._state == value:
            return

        self._state = value

        if not self.state_hook:
            return
        # 执行异步 hook
        if value == AgentStateName.RESPONDING:
            await self.state_hook.get_hook().on_responding()
        if value == AgentStateName.IDLE:
            await self.state_hook.get_hook().on_idle()

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
    def _parse_event(self, event: AgentEventModel) -> Union[AgentEventModel, None]:
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
            await self.set_state(AgentStateName.HALT)
            await self._inform_system("agent is halting")
        else:
            self._halt_event.clear()
            await self.set_state(AgentStateName.IDLE)
            await self._inform_system("agent recover")
        await self._broadcaster.broadcast(self._id, None)
        self._error_time = 0

    def broadcaster(self) -> Broadcaster:
        return self._broadcaster

    async def _idle(self, duration: Union[float, None]) -> None:
        """
        闲置一段时间, 并且会增加闲置状态.
        """
        if duration is None:
            self._idling_time = 0.0
            return
        await asyncio.sleep(duration)
        self._idling_time += duration
        await self.set_state(AgentStateName.IDLE)

    async def _handle_event(self, event: AgentEventModel) -> Optional[Response]:
        prompts = await self.make_prompts()
        if user_input := UserInputAgentEvent.from_agent_event_model(event):
            now = time.time()
            # 不处理过期事件.
            if user_input.is_overdue(now):
                self._logger.info(f"agent receive event overdue: {event}")
                return None
            return MOSShellResponse(
                shell=self.shell,
                agent_id=self._id,
                event=user_input,
                inputs=[user_input.message],
                model=self.config.model,
                prompts=prompts,
                logger=self.logger,
                eventbus=self._eventbus,
            )

        if react := ReactAgentEvent.from_agent_event_model(event):
            return MOSShellResponse(
                shell=self.shell,
                agent_id=self._id,
                event=react,
                inputs=react.messages,
                model=self.config.model,
                prompts=prompts,
                logger=self.logger,
                eventbus=self._eventbus,
            )

        if vision := VisionAgentEvent.from_agent_event_model(event):
            inputs = [Message.new(role="system").with_content(
                Text(text=vision.content),
                *vision.images,
            )]
            return MOSShellResponse(
                shell=self.shell,
                agent_id=self._id,
                event=vision,
                inputs=inputs,
                model=self.config.model,
                prompts=prompts,
                logger=self.logger,
                eventbus=self._eventbus,
            )

        if ctml := CTMLAgentEvent.from_agent_event_model(event):
            return CTMLResponse(
                shell=self.shell,
                agent_id=self._id,
                ctml=ctml.ctml,
                event=ctml,
                eventbus=self._eventbus,
                logger=self.logger,
            )

        if resume := ResumeAgentEvent.from_agent_event_model(event):
            return MOSShellResponse(
                shell=self.shell,
                agent_id=self._id,
                event=resume,
                inputs=[resume.message],
                model=self.config.model,
                prompts=prompts,
                logger=self.logger,
                eventbus=self._eventbus,
            )

        return None

    def _is_running_response(self, response_id: str) -> bool:
        return self._running_response is not None and self._running_response.response_id == response_id

    async def _handle_response(self, response: Response) -> None:
        await self.set_state(AgentStateName.RESPONDING)
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
            await self.set_state(AgentStateName.IDLE)
            await self._clear_running_response(response.response_id)

    async def _finish_response(self, response: Response) -> None:
        inputs = response.inputted()
        outputs = response.buffered()
        # 判断 outputs 不为空, 就再次保存.
        if inputs or outputs:
             await self.session.save_turn(inputs, outputs)

        # 如果response output里有拿到ctml执行的返回值，直接触发ReactEvent给Agent继续处理
        # if len(outputs) > 0 and outputs[-1].is_completed():
        #     ctml_results: List[CTMLResult] = []
        #     for content in outputs[-1].contents:
        #         if result := CTMLResult.from_content(content):
        #             ctml_results.append(result)
        #     if ctml_results:
        #         await self.eventbus().put(ReactAgentEvent(
        #             event_id=response.response_id,  # 保持同一个会话
        #             messages=[Message.new(role="assistant").with_content(*ctml_results)]
        #         ))

        # if len(outputs) > 0 and outputs[-1].is_completed():
        #     ctml_results: List[CTMLResult] = []
        #     for content in outputs[-1].contents:
        #         if result := CTMLResult.from_content(content):
        #             ctml_results.append(result)
        #
        #     def _should_auto_react(ctml: str) -> bool:
        #         """Whether CTML execution results should trigger a follow-up ReactEvent.
        #
        #         Some CTML commands are side-effect tools (e.g. start/stop background
        #         workers) where an automatic second LLM turn adds noise (extra UI
        #         prompts) and increases chances of user interrupts.
        #         """
        #
        #         ctml_stripped = (ctml or "").strip()
        #         no_react_prefixes = (
        #             "<reachy_mini.video_recorder:start_recording",
        #             "<reachy_mini.video_recorder:stop_recording",
        #             "<reachy_mini.video_recorder:status",
        #         )
        #         return not any(ctml_stripped.startswith(prefix) for prefix in no_react_prefixes)
        #
        #     reactable_results = [r for r in ctml_results if _should_auto_react(r.ctml)]
        #
        #     if reactable_results:
        #         await self.eventbus().put(ReactAgentEvent(
        #             event_id=response.response_id,  # 保持同一个会话
        #             messages=[Message.new(role="assistant").with_content(*reactable_results)]
        #         ))

    async def _clear_running_response(self, response_id: str) -> None:
        if self._running_response and self._running_response.response_id == response_id:
            await self._running_response.interrupt()
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

    async def add_event(self, event: AgentEventModel):
        await self._add_event_queue.put(event)

    async def _add_event_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                e = await self._add_event_queue.get()
                await self._add_event(e)
                self._add_event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.exception(e)

    async def _add_event(self, event: AgentEventModel) -> None:
        # 系统级别的事件, 强制关闭 agent 运行.
        if event.event_type == ShutdownAgentEvent.event_type:
            await self.close()
            return

        # 中断事件，强制中断当前响应生成
        if event.event_type == InterruptAgentEvent.event_type:
            await self._interrupt()
            return

        # 进行事件拦截逻辑.
        parsed = self._parse_event(event)
        if parsed is None:
            return

        if parsed.priority < 0:
            # 低优先级事件, 进入普通队列.
            await self._low_event_queue.put(parsed)
            return

        elif parsed.priority == 0:
            # 普通事件, 进入正常队列.
            await self._queued_event_queue.put(parsed)
            return

        elif parsed.priority > 0:
            await self._preempt_event_queue.put(parsed)
            _running_response = self._running_response
            if _running_response is not None:
                # 更高级别的事件也会触发中断.
                if parsed.priority > _running_response.event.priority:
                    await self._interrupt()
                    return
        return

    async def _interrupt(self) -> None:
        """
        同步阻塞的 interrupt 逻辑.
        """
        if self._running_response is not None:
            response = self._running_response
            await response.interrupt()
            await self._clear_running_response(response.response_id)
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
        await self.set_state(AgentStateName.BOOTSTRAP)
        self._auto_shutdown = auto_shutdown
        await self.set_state(AgentStateName.IDLE)

        await self.shell.start()

        self._main_loop_task = asyncio.create_task(self._main_loop())
        self._add_event_task = asyncio.create_task(self._add_event_loop())

        self._logger.info("Agent is now running.")

    async def close(self) -> None:
        if self._state == AgentStateName.SHUTDOWN:
            return
        await self.set_state(AgentStateName.SHUTDOWN)
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

    def _parse_event(self, event: AgentEventModel) -> Union[AgentEventModel, None]:
        return event

    @classmethod
    def new(cls, container: Container, config: AgentConfig) -> Self:
        shell = container.force_fetch(MOSSShell)
        session = container.force_fetch(Session)
        return cls(container=container, config=config, shell=shell, session=session)
