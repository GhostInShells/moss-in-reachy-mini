from typing import Union, Self, List, Optional

from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss import MOSSShell, Message, Text, PyChannel

from framework.abcd.agent import AgentConfig, Response
from framework.abcd.agent_event import AgentEvent, ProgramInputAgentEvent
from framework.abcd.agent_hook import AgentHook
from framework.abcd.agent_hub import EventBus
from framework.abcd.session import Session
from framework.agent.agent_hook import BaseAgentHook
from framework.agent.main_agent import BaseMainAgent
from framework.agent.response import AgentEventAddition
from framework.apps.live.douyin_live import DouyinLive
from framework.apps.session.storage_session import StorageSession, TurnAddition


class DecisionSession(StorageSession):
    pass

class DecisionAgent(BaseMainAgent):
    """
    Decision Agent.
    """
    def __init__(
            self,
            container: IoCContainer,
            config: AgentConfig,
            shell: MOSSShell,
            main_session: Session,
            decision_session: DecisionSession,
    ):
        super().__init__(container, config, shell, main_session)

        self.decision_session = decision_session

    def _parse_event(self, event: AgentEvent) -> Union[AgentEvent, None]:
        return event

    async def give_cues(self, text__):
        """
        给主Agent提供决策建议
        :param text__:
        """
        await self._eventbus.put(ProgramInputAgentEvent(
            message=Message.new(role="user", name="__decision_agent_cues__").with_content(
                Text(text=CUES_PROMPT),
                Text(text=text__)
            ),
            agent_id="",  # 默认主agent
            priority=0,
            overdue=30,
        ))

    @classmethod
    def new(cls, container: IoCContainer, config: AgentConfig) -> Self:
        main_session = container.force_fetch(Session)
        decision_session = container.force_fetch(DecisionSession)
        hook = container.force_fetch(DecisionAgentHook)
        shell = container.force_fetch(MOSSShell)
        ins = cls(
            container=container,
            config=config,
            shell=shell,
            main_session=main_session,
            decision_session=decision_session,
        )
        ins.set_state_hook(hook)
        shell.main_channel.build.command()(ins.give_cues)
        return ins

    async def _finish_response(self, response: Response) -> None:
        """
        仅和主脑共享session，但是自己不主动更新session
        """
        inputs = response.inputted()
        outputs = response.buffered()
        # 判断 outputs 不为空, 就再次保存.
        if inputs or outputs:
            await self.decision_session.save_turn(inputs, outputs)

    async def make_prompts(self) -> List[Message]:
        """
        决策Agent需要知道哪些是自己的上下文，哪些是主Agent的上下文.
        """
        super_prompts = await super().make_prompts()
        decision_prompts = await self.decision_session.get_session_history()
        messages = super_prompts + decision_prompts
        for message in messages:
            addition = AgentEventAddition.read(message)
            if not addition or not addition.agent_id:
                continue
            for i, content in enumerate(message.contents):
                if text := Text.from_content(content):
                    text.text = f"[{addition.agent_id} agent] {text.text}"
                    message.contents[i] = text.to_content()
        return messages

CUES_PROMPT = """
## 使用说明
1. 此事件会作为上下文消息插入到你的下一次交互中
2. 你应像处理灵感闪现一样自然参考这些建议
3. 建议可以有选择性地采纳，不必全部执行
4. 所有建议都应转化为你自己的语言和风格
5. 对外保持连续性，不暴露系统来源

## 处理指南
当看到此事件时，请：
1. 快速阅读建议内容
2. 评估当前情境与建议的匹配度
3. 选择最合适的建议自然融入
4. 保持互动流畅，不突兀转折
5. 可以混合多个建议或只采用部分想法
6. 避免单独重复响应已处理的互动事件，可以对未处理事件响应时联想已处理事件。

**关键原则**：让建议成为你自然思考过程的一部分，而不是外部指令。保持互动深度和连贯性比表面欢迎更重要。

## 下面是 DecisionAgent 本轮的建议
"""

class DecisionAgentHook(BaseAgentHook):
    def __init__(
            self,
            main_session: Session,
            eventbus: EventBus,
            decision_agent_id: str,
            douyin_live: DouyinLive,
    ):
        super().__init__()
        self.main_session = main_session
        self.eventbus = eventbus
        self.decision_agent_id = decision_agent_id
        self.douyin_live = douyin_live
        self.latest_message: Optional[Message] = None

    def get_hook(self) -> AgentHook:
        return self

    async def on_self_enter(self):
        pass

    async def on_self_exit(self):
        pass

    async def _run_idle_move(self):
        # ============ 添加主脑上下文更新信息 ============
        message = Message.new(role="user", name="__decision_agent__")
        session_history = await self.main_session.get_session_history()
        if session_history:
            latest = session_history[-1]
            now_turn = TurnAddition.read(latest)
            latest_turn = TurnAddition.read(self.latest_message)
            if not latest_turn or now_turn.turn_id != latest_turn.turn_id:
                # 更新最新的上下文锚点
                self.latest_message = latest
                message.with_content(
                    Text(text="====== 主Agent上下文更新事件 start ======"),
                    Text(text="主Agent做出了新的回复，这个回复有可能是用户输入或者其他事件触发，也可能是上一轮你的`give_cues`触发。"
                              "所以你需要根据上下文充分理解当前的现状后，判断是否需要补充行动、使用`give_cues`给出进一步的建议。"
                              "**原则**："
                              "1. 不可以滥用`give_cues`，你的建议会让main agent做出新的回复导致又回到你这个思维起点，你必须谨慎使用`give_cues`"
                              "**严禁**："
                              "1. 重复上一轮的建议；"
                              "2. 重复main agent的行动；"
                              "3. 在明确问题上和main agent都不行动；"),
                    Text(text="====== 主Agent上下文更新事件 end ======"),
                )

        # ============ 添加直播间的事件更新信息 ============
        recent_unprocessed_events = await self.douyin_live.get_unprocessed_events()
        if recent_unprocessed_events:
            message.with_content(
                Text(text="====== 抖音直播间事件 start ======"),
                Text(text=self.douyin_live.config.idle_task_prompt),
                Text(text=f"\n发现{len(recent_unprocessed_events)}个未处理事件，请分析："),
                Text(text=f"当前在线人数：{self.douyin_live.current_users}"),
                Text(text=f"事件类型分布："),
                *[Text(text=f"- {event.to_natural()}") for event in recent_unprocessed_events],
                Text(text=f"\n请分析这些事件并提供互动建议。"),
                Text(text="====== 抖音直播间事件 end ======"),
            )

        if not message.is_empty():
            await self.eventbus.put(ProgramInputAgentEvent(
                message=message,
                agent_id=self.decision_agent_id,
                priority=0,  # 普通队列，可被高优事件打断
            ))


class DecisionAgentHookProvider(Provider[DecisionAgentHook]):
    def __init__(self, decision_agent_id: str):
        self.decision_agent_id = decision_agent_id

    def singleton(self) -> bool:
        return True

    def factory(self, container: IoCContainer) -> DecisionAgentHook:
        main_session = container.force_fetch(Session)
        eventbus = container.force_fetch(EventBus)
        douyin_live = container.force_fetch(DouyinLive)
        return DecisionAgentHook(
            main_session=main_session,
            eventbus=eventbus,
            decision_agent_id=self.decision_agent_id,
            douyin_live=douyin_live,
        )
