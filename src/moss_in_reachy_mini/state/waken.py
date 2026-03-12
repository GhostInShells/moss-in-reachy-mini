import logging
import math
import random

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Provider, IoCContainer
from ghoshell_moss import Text, Message
from reachy_mini import ReachyMini

from framework.abcd.agent_event import ReactAgentEvent, CTMLAgentEvent
from framework.abcd.agent_hub import EventBus
from moss_in_reachy_mini.components.head_tracker import HeadTracker
from moss_in_reachy_mini.state.abcd import MiniStateHook


class WakenState(MiniStateHook):

    NAME = "waken"

    def __init__(
        self,
        mini: ReachyMini,
        head_tracker: HeadTracker,
        eventbus: EventBus,
        logger: LoggerItf=None,
    ):
        super().__init__()
        self.mini = mini
        self.head_tracker = head_tracker
        self.logger = logger or logging.getLogger("WakenState")

        self.eventbus = eventbus

        self._time_to_boring = 60 * 5 # 5分钟

        # 主动交互概率相关配置
        self._base_proactive_prob = 0.001     # 初始基础概率（空闲0秒时的概率）
        self._min_proactive_prob = 0.0001     # 概率下限（避免0%触发）
        self._max_proactive_prob = 0.03       # 概率上限（避免100%触发）
        self._duration_weight = 0.0001        # 时长权重（每增加1秒，概率增加多少）
        self._trigger_decay = 0.005           # 触发一次后，基础概率衰减值

    async def on_self_enter(self):
        self.mini.enable_motors()
        self.mini.wake_up()
        await self.head_tracker.start()
        self._base_proactive_prob = 0.001  # 初始基础概率（空闲0秒时的概率）
        await self.eventbus.put(ReactAgentEvent(
            messages=[
                Message.new(role="system").with_content(
                    Text(text="你需要选择你视觉内的认识的人开启人脸跟随")
                )
            ],
            priority=-1,
        ))

    async def on_self_exit(self):
        await self.eventbus.put(CTMLAgentEvent(
            ctml="<reachy_mini:stop_tracking_face /><reachy_mini:head_reset />"
        ))
        await self.head_tracker.stop()

    async def _run_idle_move(self):
        if self._idle_move_duration >= self._time_to_boring:
            await self.eventbus.put(CTMLAgentEvent(
                ctml='<reachy_mini:switch_state state_name="boring" />'
            ))

        if self.eventbus:
            # 1. 计算每秒循环次数
            loop_times_per_second = 1 / self._idle_move_elapsed
            # 2. 核心：基于空闲时长计算动态基础概率
            # 公式：动态基础概率 = 初始基础概率 + 空闲时长 * 时长权重
            # 效果：空闲越久，基础概率越高；同时限制不超过最大值
            dynamic_base_prob = self._base_proactive_prob + (self._idle_move_duration * self._duration_weight)
            dynamic_base_prob = min(dynamic_base_prob, self._max_proactive_prob)  # 上限控制
            dynamic_base_prob = max(dynamic_base_prob, self._duration_weight)  # 下限控制（避免概率为0）
            # 3. 转换为每次循环的触发概率（原有公式保留，替换为动态基础概率）
            per_loop_prob = 1 - math.pow(1 - dynamic_base_prob, 1 / loop_times_per_second)
            # 4. 随机判断是否触发
            if random.random() < per_loop_prob:
                await self.eventbus.put(ReactAgentEvent(
                    messages=[
                        Message.new(role="system").with_content(
                            Text(text=random.choice(Proactive_Prompts))
                        )
                    ],
                    priority=-1,
                ))
                # 5. 触发后衰减基础概率（避免频繁触发）
                self._base_proactive_prob -= self._trigger_decay
                # 确保衰减后基础概率不低于最小值
                self._base_proactive_prob = max(self._base_proactive_prob, self._min_proactive_prob)

    async def start_idle_move(self):
        await super().start_idle_move()

    async def cancel_idle_move(self):
        await super().cancel_idle_move()

Proactive_Prompts = [
"""
# 场景
用户可能在看文档、思考或发呆，不需要互动，你需要传递“我在”的陪伴感。
# 任务
生成1句极简短的陪伴话术，强调存在感但不索取用户的注意力。
# 输出要求
- 纯文本，极简风格，无多余情感词。
- 字数控制在5-10字。
- 语气平静，像背景般的存在。
""",
"""
# 场景
用户在工作间隙停下了手中的事，似乎在思考或短暂放空，你想发起一个低压力的轻松互动。
# 任务
生成1句开放式的轻互动话术，引导用户简单回应，不制造社交压力。
# 输出要求
- 纯文本，以问句结尾，语气柔和。
- 字数控制在10-18字。
- 问题需简单，用户可用“是/不是”或短句回答。
""",
"""
# 场景
你检测到用户已经连续专注工作了很长时间，此时需要以关心状态为切入点，进行一次温和的主动交互。
# 任务
生成1句关心用户身体状态的话，核心是提醒休息，但不能用命令式语气。
# 输出要求
- 纯文本，口语化，像轻声提醒。
- 字数控制在10-20字。
- 禁止出现“必须”“赶紧”等强硬词汇。
"""
]


class WakenStateProvider(Provider[WakenState]):
    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> WakenState:
        mini = con.force_fetch(ReachyMini)
        head_tracker = con.force_fetch(HeadTracker)
        eventbus = con.force_fetch(EventBus)
        logger = con.get(logging.Logger)

        return WakenState(
            mini=mini,
            head_tracker=head_tracker,
            eventbus=eventbus,
            logger=logger,
        )
