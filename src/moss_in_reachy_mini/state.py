import abc
import asyncio
import math
import random
import time
from functools import partial
from typing import Optional, List

from ghoshell_common.contracts import LoggerItf
from ghoshell_moss import Text, Message
from reachy_mini import ReachyMini

from framework.abcd.agent_hook import AgentHook
from moss_in_reachy_mini.channels.antennas import Antennas
from moss_in_reachy_mini.channels.body import Body
from moss_in_reachy_mini.channels.head import Head


class QuitIdleMove(Exception):
    pass


class MiniStateHook(AgentHook, abc.ABC):
    NAME = ""

    def __init__(self):
        self._run_idle_move_task: Optional[asyncio.Task] = None
        self._idle_move_duration = 0
        self._idle_move_elapsed = 0.1

    @abc.abstractmethod
    async def on_self_enter(self):
        pass

    @abc.abstractmethod
    async def on_self_exit(self):
        pass

    @abc.abstractmethod
    async def _run_idle_move(self):
        pass

    async def run_idle_move(self):
        start = int(time.time())
        try:
            while True:
                await asyncio.sleep(self._idle_move_elapsed)
                now = int(time.time())
                self._idle_move_duration = now - start

                await self._run_idle_move()

        except asyncio.CancelledError:
            raise
        except QuitIdleMove:
            pass
        finally:
            self._idle_move_duration = 0

    async def start_idle_move(self):
        await self.cancel_idle_move()
        self._run_idle_move_task = asyncio.create_task(self.run_idle_move())

    async def cancel_idle_move(self):
        if self._run_idle_move_task is not None and not self._run_idle_move_task.done():
            self._run_idle_move_task.cancel()
            try:
                await self._run_idle_move_task
            except asyncio.CancelledError:
                pass

    async def on_idle(self):
        await self.start_idle_move()

    async def on_responding(self):
        await self.cancel_idle_move()


class AsleepState(MiniStateHook):

    NAME = "asleep"

    def __init__(self, mini: ReachyMini):
        super().__init__()
        self.mini = mini

    async def on_self_enter(self):
        self.mini.set_target_body_yaw(0.0)
        self.mini.goto_sleep()
        self.mini.disable_motors()

    async def on_self_exit(self):
        pass

    async def _run_idle_move(self):
        pass


class WakenState(MiniStateHook):

    NAME = "waken"

    def __init__(self, mini: ReachyMini, head: Head, antennas: Antennas, turn_to_boring):
        super().__init__()
        self.mini = mini
        self.head = head
        self.antennas = antennas
        self.turn_to_boring = turn_to_boring

        self._time_to_boring = 60 * 5 # 5分钟
        self._proactive_input = None

        # 主动交互概率相关配置
        self._base_proactive_prob = 0.001     # 初始基础概率（空闲0秒时的概率）
        self._min_proactive_prob = 0.0001     # 概率下限（避免0%触发）
        self._max_proactive_prob = 0.03       # 概率上限（避免100%触发）
        self._duration_weight = 0.0001        # 时长权重（每增加1秒，概率增加多少）
        self._trigger_decay = 0.005           # 触发一次后，基础概率衰减值

    async def on_self_enter(self):
        self.mini.enable_motors()
        self.mini.wake_up()
        self._base_proactive_prob = 0.001  # 初始基础概率（空闲0秒时的概率）

    async def on_self_exit(self):
        await self.cancel_idle_move()
        await self.head.reset()

    async def _run_idle_move(self):
        if self._idle_move_duration >= self._time_to_boring:
            await self.turn_to_boring()
            raise QuitIdleMove

        if self._proactive_input:
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
                self._proactive_input(random.choice(Proactive_Prompts))
                # 5. 触发后衰减基础概率（避免频繁触发）
                self._base_proactive_prob -= self._trigger_decay
                # 确保衰减后基础概率不低于最小值
                self._base_proactive_prob = max(self._base_proactive_prob, self._min_proactive_prob)

    async def start_idle_move(self):
        await super().start_idle_move()
        await self.head.on_policy_run()
        await self.antennas.on_policy_run()

    async def cancel_idle_move(self):
        await super().cancel_idle_move()
        await self.head.on_policy_pause()
        await self.antennas.on_policy_pause()

    def set_proactive_input(self, handle_input):
        self._proactive_input = handle_input


class BoringState(MiniStateHook):

    NAME = "boring"

    def __init__(self, mini: ReachyMini, body: Body, turn_to_asleep, back_to_waken):
        super().__init__()
        self.mini = mini
        self.body = body

        self.turn_to_asleep = turn_to_asleep
        self.back_to_waken = back_to_waken

        self._time_to_sleep = 30 # 30秒
        self._emotion_prob = 0.03 # 目标：每秒有3%的概率触发函数

    async def on_self_enter(self):
        # Boring只能靠自己来触发idle move
        await self.start_idle_move()

    async def on_self_exit(self):
        pass

    async def _run_idle_move(self):
        if self._idle_move_duration >= self._time_to_sleep:
            await self.turn_to_asleep()
            raise QuitIdleMove

        loop_times_per_second = 1 / self._idle_move_elapsed  # 每秒循环的次数
        per_loop_prob = 1 - math.pow(1 - self._emotion_prob, 1 / loop_times_per_second)  # 每次循环的概率
        if per_loop_prob >= 0 and  random.random() < per_loop_prob:
            emotion = random.choice(["sleep1", "boredom1", "boredom2"])
            await self.body.emotion(emotion)
            self._emotion_prob -= 0.003  # 每次触发后降低概率

    async def cancel_idle_move(self):
        await super().cancel_idle_move()
        await self.back_to_waken()  # cancel_idle_move由Agent触发
        self._emotion_prob = 0.03  # 重置触发概率


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


class StateLog:
    def __init__(self, from_state: MiniStateHook, to_state: MiniStateHook):
        self.from_state = from_state
        self.to_state = to_state
        self.now = int(time.time())

class StateManagerHook(AgentHook):
    def __init__(self, mini: ReachyMini, body: Body, head: Head, antennas: Antennas, logger: LoggerItf):
        self._state_map = {
            AsleepState.NAME: AsleepState(mini),
            WakenState.NAME: WakenState(
                mini,
                head=head,
                antennas=antennas,
                turn_to_boring=partial(self.switch_to, BoringState.NAME),
            ),
            BoringState.NAME: BoringState(
                mini,
                body=body,
                turn_to_asleep=partial(self.switch_to, AsleepState.NAME),
                back_to_waken=partial(self.switch_to, WakenState.NAME),
            )
        }
        self._state: Optional[MiniStateHook] = None
        self._state_log: List[StateLog] = []
        self.logger = logger

    async def switch_to(self, state_name: str):
        if state_name not in self._state_map:
            raise ValueError(f'Invalid state name: {state_name}')

        if self._state:
            await self._state.on_self_exit()

        self.logger.info(f'Switching state from {self._state.NAME if self._state else "initial"} to {state_name}')
        self._state_log.append(StateLog(self._state, self._state_map[state_name]))  # 记录状态切换
        self._state = self._state_map[state_name]
        await self._state.on_self_enter()

    def current(self) -> MiniStateHook:
        return self._state

    async def on_idle(self):
        await self._state.on_idle()

    async def on_responding(self):
        await self._state.on_responding()

    def to_contents(self):
        contents = []
        now = int(time.time())
        for state in self._state_log:
            ago = now - state.now
            if not state.from_state:
                text = f"Start state to {state.to_state.NAME} occurred {ago} seconds ago"
            else:
                text = f"Switch state from {state.from_state.NAME} to {state.to_state.NAME} occurred {ago} seconds ago"
            contents.append(Text(text=text))
        return contents

    def clear_state_log(self):
        self._state_log.clear()

    async def start(self):
        await self.switch_to(AsleepState.NAME)

    async def close(self):
        await self.switch_to(AsleepState.NAME)