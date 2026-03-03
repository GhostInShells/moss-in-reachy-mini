import abc
import asyncio
import logging
import math
import random
import time
from typing import Optional

from ghoshell_common.contracts import LoggerItf
from ghoshell_container import IoCContainer
from ghoshell_moss import Text, Message, PyChannel
from reachy_mini import ReachyMini

from framework.abcd.agent import EventBus
from framework.abcd.agent_event import ReactAgentEvent
from framework.abcd.agent_hook import AgentHook
from framework.live.douyin_live_channel import DouyinLiveChannel
from moss_in_reachy_mini.components.antennas import Antennas
from moss_in_reachy_mini.components.body import Body
from moss_in_reachy_mini.components.head import Head
from moss_in_reachy_mini.components.vision import Vision


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


class InitialState(MiniStateHook):
    NAME = "initial"

    def __init__(self):
        super().__init__()

    async def on_self_enter(self):
        pass

    async def on_self_exit(self):
        pass

    async def _run_idle_move(self):
        pass


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

    async def context_messages(self):
        msg = Message.new(role="system").with_content(
            Text(text="你现在处于asleep状态，你必须得先切换到waken状态才能继续和用户进行交互"),
        )
        return [msg]

    def as_channel(self):
        chan = PyChannel(name=AsleepState.NAME, description=f"current state is asleep", block=True)
        chan.build.with_context_messages(self.context_messages)
        return chan

class WakenState(MiniStateHook):

    NAME = "waken"

    def __init__(self, mini: ReachyMini, body: Body, head: Head, antennas: Antennas, vision: Vision, switch_to, container: IoCContainer):
        super().__init__()
        self.mini = mini
        self.body = body
        self.head = head
        self.antennas = antennas
        self.vision = vision
        self.switch_to = switch_to
        self.container = container
        self.logger = container.get(LoggerItf) or logging.getLogger("WakenState")

        self._eventbus: Optional[EventBus] = container.get(EventBus)

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
        self._base_proactive_prob = 0.001  # 初始基础概率（空闲0秒时的概率）
        if self._eventbus:
            await self._eventbus.put(ReactAgentEvent(
                messages=[
                    Message.new(role="system").with_content(
                        Text(text="你现在进入Waken状态了，可以选择你眼前的人进行人脸跟随")
                    )
                ],
                priority=-1,
            ).to_agent_event())

    async def on_self_exit(self):
        await self.head.stop_tracking_face()
        await self.head.reset()

    async def _run_idle_move(self):
        if self._idle_move_duration >= self._time_to_boring:
            await self.switch_to(BoringState.NAME)
            raise QuitIdleMove

        if self._eventbus:
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
                await self._eventbus.put(ReactAgentEvent(
                    messages=[
                        Message.new(role="system").with_content(
                            Text(text=random.choice(Proactive_Prompts))
                        )
                    ],
                    priority=-1,
                ).to_agent_event())
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

    async def context_messages(self):
        msg = Message.new(role="system").with_content(
            Text(text="你现在处于Waken状态"),
        )
        vision_message = await self.vision.context_messages()
        head_msg = await self.head.context_messages()
        antenna_msg = await self.antennas.context_messages()
        return [msg] + vision_message + head_msg + antenna_msg

    def as_channel(self):
        waken_chan = PyChannel(name=WakenState.NAME, description=f"current state is waken", block=True)
        waken_chan.build.command(doc=self.body.dance_docstring)(self.body.dance)
        waken_chan.build.command(doc=self.body.emotion_docstring)(self.body.emotion)
        waken_chan.build.command(name="head_move")(self.head.move)
        waken_chan.build.command(name="head_reset")(self.head.reset)
        waken_chan.build.command()(self.head.start_tracking_face)
        waken_chan.build.command()(self.head.stop_tracking_face)
        waken_chan.build.command()(self.head.start_breathing)
        waken_chan.build.command()(self.head.stop_breathing)
        waken_chan.build.command(name="antennas_move")(self.antennas.move)
        waken_chan.build.command(name="antennas_reset")(self.antennas.reset)
        waken_chan.build.command()(self.antennas.set_idle_flapping)
        waken_chan.build.command()(self.antennas.enable_flapping)
        waken_chan.build.command()(self.vision.look)
        waken_chan.build.with_context_messages(self.context_messages)
        return waken_chan


class BoringState(MiniStateHook):

    NAME = "boring"

    def __init__(self, mini: ReachyMini, body: Body, vision: Vision, switch_to, container: IoCContainer):
        super().__init__()
        self.mini = mini
        self.body = body
        self.vision = vision

        self.switch_to = switch_to
        self.container = container
        self.logger = container.get(LoggerItf) or logging.getLogger("BoringState")

        self._time_to_sleep = 30 # 30秒
        self._emotion_prob = 0.03 # 目标：每秒有3%的概率触发函数

    async def on_self_enter(self):
        # Boring只能靠自己来触发idle move
        await self.start_idle_move()

    async def on_self_exit(self):
        pass

    async def _run_idle_move(self):
        if self._idle_move_duration >= self._time_to_sleep:
            await self.switch_to(AsleepState.NAME)
            raise QuitIdleMove

        loop_times_per_second = 1 / self._idle_move_elapsed  # 每秒循环的次数
        per_loop_prob = 1 - math.pow(1 - self._emotion_prob, 1 / loop_times_per_second)  # 每次循环的概率
        if per_loop_prob >= 0 and  random.random() < per_loop_prob:
            emotion = random.choice(["sleep1", "boredom1", "boredom2"])
            await self.body.emotion(emotion)
            self._emotion_prob -= 0.003  # 每次触发后降低概率

    async def cancel_idle_move(self):
        await super().cancel_idle_move()
        await self.switch_to(WakenState.NAME)  # cancel_idle_move由Agent触发
        self._emotion_prob = 0.03  # 重置触发概率

    async def context_messages(self):
        msg = Message.new(role="system").with_content(
            Text(text="你现在处于Boring状态"),
        )
        vision_message = await self.vision.context_messages()
        return [msg] + vision_message

    def as_channel(self):
        boring_chan = PyChannel(name=BoringState.NAME, description=f"current state is boring", block=True)
        boring_chan.build.command(doc=self.body.emotion_docstring)(self.body.emotion)
        boring_chan.build.command()(self.vision.look)
        boring_chan.build.with_context_messages(self.context_messages)
        return boring_chan

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

class LiveState(MiniStateHook):

    NAME = "live"
    LIVE_ID = "892335786371"

    def __init__(self, mini: ReachyMini, body: Body, head: Head, antennas: Antennas, vision: Vision, switch_to, container: IoCContainer):
        super().__init__()
        self.mini = mini
        self.body = body
        self.head = head
        self.antennas = antennas
        self.vision = vision
        self.switch_to = switch_to
        self.container = container
        self.logger = container.get(LoggerItf) or logging.getLogger("WakenState")
        self._eventbus: Optional[EventBus] = container.get(EventBus)

        self.douyin_live_channel = DouyinLiveChannel(self.LIVE_ID, self.container)

        self._time_to_react = 20  # 20秒

    async def on_self_enter(self):
        self.mini.enable_motors()
        self.mini.wake_up()
        await self.head.start_breathing()
        if self.LIVE_ID == "":
            await self._eventbus.put(ReactAgentEvent(
                messages=[Message.new(role="system").with_content(
                    Text(text="切换到live状态失败，未指定直播ID，你需要切换回Waken状态")
                )]
            ).to_agent_event())
            return

        self.douyin_live_channel.start()

    async def on_self_exit(self):
        self.douyin_live_channel.stop()

    async def _run_idle_move(self):
        if self._idle_move_duration <= self._time_to_react:
            return

        if self._eventbus:
            await self._eventbus.put(ReactAgentEvent(
                messages=[Message.new(role="system").with_content(
                    Text(text=random.choice(LiveIdle_Prompts))
                )]
            ).to_agent_event())

    async def start_idle_move(self):
        await super().start_idle_move()
        await self.head.on_policy_run()
        await self.antennas.on_policy_run()

    async def cancel_idle_move(self):
        await super().cancel_idle_move()
        await self.head.on_policy_pause()
        await self.antennas.on_policy_pause()

    def as_channel(self):
        chan = self.douyin_live_channel.as_channel()
        chan.build.with_description()(lambda :"当前状态是直播状态，不可以切换为其他状态")
        chan.build.command(doc=self.body.dance_docstring)(self.body.dance)
        chan.build.command(doc=self.body.emotion_docstring)(self.body.emotion)
        chan.build.command(name="head_move")(self.head.move)
        chan.build.command(name="head_reset")(self.head.reset)
        chan.build.command()(self.head.start_breathing)
        chan.build.command()(self.head.stop_breathing)
        chan.build.command(name="antennas_move")(self.antennas.move)
        chan.build.command(name="antennas_reset")(self.antennas.reset)
        chan.build.command()(self.antennas.set_idle_flapping)
        chan.build.command()(self.antennas.enable_flapping)
        return chan

LiveIdle_Prompts = ["""
话题轮换：避免每次都说同样的话。可以从以下几个方向随机选择：
1. 延续话题：回顾刚才聊的内容，补充一个有趣的细节或抛出新的角度。“刚才我们说到XX，我突然想到一个特别有意思的例子……”
2. 提问互动：向观众抛出一个开放式问题，引导大家参与。“想问下直播间的朋友们，你们周末最喜欢做什么呀？在公屏上告诉我~”
3. 分享日常：聊聊自己（AI）的趣事、最近的热点、冷知识等。“我最近学到一个超实用的生活小技巧，分享给你们……”
4. 感谢与提醒：感谢观众停留，提醒点赞、分享等。“谢谢大家还在直播间，如果觉得内容不错，动动手指点个赞，让我看到你们！”
5. 趣味互动：发起小游戏，如“猜谜语”、“看图猜物”、“听前奏猜歌名”等。“来玩个猜谜游戏：什么东西越洗越脏？答案是水！你们猜对了吗？”
6. 结合历史：参考之前用户评论中提到的兴趣点，延续相关话题。
7. 语气自然：保持口语化，加入表情动作，
""",
"""
 讲一个有趣的笑话，以活跃直播间气氛。请严格遵循以下多样化原则：

**笑话库轮换策略**（每次从以下类别中随机选择一个）：
- **生活糗事**：讲述自己或身边人的搞笑经历，增加真实感。“我昨天煮泡面，把调料包撕开后直接扔进了锅里，结果发现没撕口……最后捞出来的时候，调料包已经鼓成气球了！”
- **谐音梗/双关语**：利用词语歧义制造笑点，适合轻松互动。“为什么数学书总是很忧郁？因为它有太多‘几何’（音似‘几盒’愁）问题。”
- **冷笑话**：一本正经地讲无厘头笑话，配合呆萌的语气。“有一天，绿豆跟女朋友分手了，它很难过，然后它哭了，结果……它变成了豆芽。”
- **动物笑话**：以小动物为主角的趣事。“为什么企鹅的肚子是白色的？因为如果它肚子朝上躺在雪地里，就没人能发现它……（停顿）好吧，其实是方便它摔倒时伪装。”
- **程序员专属笑话**（如果直播间技术类观众多）：“程序员最讨厌康熙的哪个儿子？——四阿哥（Four阿哥，谐音for循环）。”

**讲笑话的流程**：
1.  **开场白**：用一句吸引人的话引入，如“哎，直播间有点安静，我给大家讲个笑话吧～”、“刚才突然想到一个特别搞笑的，你们想听吗？”
2.  **讲述笑话**：口语化，适当加入语气词和停顿，模拟真人讲故事。可以在关键处加入`[神秘一笑]` `[摊手]`等动作提示。
3.  **互动收尾**：讲完后立刻抛出互动，如“这个笑话能打几分？满分10分的话，公屏告诉我！”、“如果不好笑，我自罚一首歌！”

**注意事项**：
- 避免低俗、冒犯性内容，保持正能量。
- 如果连续多次收到 `[空场 讲笑话]`，笑话类型必须轮换，不能重复。
"""
]
