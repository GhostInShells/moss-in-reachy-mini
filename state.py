import abc
import asyncio
import math
import random
import time
from typing import Optional

from reachy_mini import ReachyMini

from channels.antennas import Antennas
from channels.body import Body
from channels.head import Head


class QuitIdleMove(Exception):
    pass


class BaseState(abc.ABC):
    NAME = ""

    def __init__(self):
        self._run_idle_move_task: Optional[asyncio.Task] = None
        self._idle_move_duration = 0
        self._idle_move_elapsed = 0.1

    @abc.abstractmethod
    async def on_enter(self):
        pass

    @abc.abstractmethod
    async def on_exit(self):
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
        await self._cancel_run_idle_move_task()
        self._run_idle_move_task = asyncio.create_task(self.run_idle_move())

    async def cancel_idle_move(self):
        await self._cancel_run_idle_move_task()

    async def _cancel_run_idle_move_task(self):
        if self._run_idle_move_task is not None and not self._run_idle_move_task.done():
            self._run_idle_move_task.cancel()
            try:
                await self._run_idle_move_task
            except asyncio.CancelledError:
                pass


class AsleepState(BaseState):

    NAME = "asleep"

    def __init__(self, mini: ReachyMini):
        super().__init__()
        self.mini = mini

    async def on_enter(self):
        self.mini.set_target_body_yaw(0.0)
        self.mini.goto_sleep()
        self.mini.disable_motors()

    async def on_exit(self):
        pass

    async def _run_idle_move(self):
        pass


class WakenState(BaseState):

    NAME = "waken"

    def __init__(self, mini: ReachyMini, head: Head, antennas: Antennas, turn_to_boring):
        super().__init__()
        self.mini = mini
        self.head = head
        self.antennas = antennas
        self.turn_to_boring = turn_to_boring

        self._time_to_boring = 60 * 5 # 5分钟
        self._proactive_input = None
        self._proactive_prob = 0.03 # 每秒有3%的概率触发函数

    async def on_enter(self):
        self.mini.enable_motors()
        self.mini.wake_up()

    async def on_exit(self):
        await self.cancel_idle_move()
        await self.head.reset()

    async def _run_idle_move(self):
        if self._idle_move_duration >= self._time_to_boring:
            await self.turn_to_boring()
            raise QuitIdleMove

        if self._proactive_input:
            loop_times_per_second = 1 / self._idle_move_elapsed  # 每秒循环的次数
            per_loop_prob = 1 - math.pow(1 - self._proactive_prob, 1 / loop_times_per_second)  # 每次循环的概率
            if random.random() < per_loop_prob:
                self._proactive_input(random.choice(Proactive_Prompts))
                self._proactive_prob -= 0.01  # 降低触发概率

    async def start_idle_move(self):
        await super().start_idle_move()
        await self.head.on_policy_run()
        await self.antennas.on_policy_run()

    async def cancel_idle_move(self):
        await super().cancel_idle_move()
        await self.head.on_policy_pause()
        await self.antennas.on_policy_pause()
        self._proactive_prob = 0.03  # 重置触发概率

    def set_proactive_input(self, handle_input):
        self._proactive_input = handle_input


class ListeningState(BaseState):
    NAME = "listening"

    def __init__(self, mini: ReachyMini, head, antennas, back_to_waken):
        super().__init__()
        self.mini = mini
        self.head = head
        self.antennas = antennas
        self.back_to_waken = back_to_waken

    async def on_enter(self):
        # listening动作
        pass

    async def on_exit(self):
        await self.head.reset()
        await self.back_to_waken()

    async def _run_idle_move(self):
        pass


class BoringState(BaseState):

    NAME = "boring"

    def __init__(self, mini: ReachyMini, body: Body, turn_to_asleep, back_to_waken):
        super().__init__()
        self.mini = mini
        self.body = body

        self.turn_to_asleep = turn_to_asleep
        self.back_to_waken = back_to_waken

        self._time_to_sleep = 60 * 5 # 5分钟
        self._emotion_prob = 0.03 # 目标：每秒有3%的概率触发函数

    async def on_enter(self):
        # Boring只能靠自己来触发idle move
        await self.start_idle_move()

    async def on_exit(self):
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