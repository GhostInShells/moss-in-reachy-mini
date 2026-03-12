import asyncio
import enum
import json
import logging
import random
import time
import uuid
from collections import defaultdict, deque
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ghoshell_common.contracts import YamlConfig, Storage, WorkspaceConfigs, FileStorage, Workspace, LoggerItf
from ghoshell_container import INSTANCE, Provider, IoCContainer
from ghoshell_moss import Message, Text, PyChannel
from pydantic import Field, BaseModel

from framework.abcd.agent_event import UserInputAgentEvent
from framework.abcd.agent_hub import EventBus
from framework.apps.live.DouyinLiveWebFetcher.liveMan import DouyinLiveWebFetcher
from framework.apps.live.DouyinLiveWebFetcher.protobuf.douyin import ChatMessage, GiftMessage, LikeMessage, \
    MemberMessage, \
    SocialMessage, RoomUserSeqMessage
from framework.apps.live.barrage_classify.config import Priority
from framework.apps.utils import EnumEncoder
from framework.apps.live.barrage_classify.classifier import BarrageClassifier


class DouyinLiveConfig(YamlConfig):
    relative_path = "douyin_live_config.yaml"

    live_id: str = Field(default="", description="douyin live id")

    p0_prompt: str = Field(default="", description="douyin p0 prompt")
    p0_overdue: int = Field(default=30, description="douyin p0 overdue")
    gift_prompt: str = Field(default="", description="douyin gift prompt")

    max_user_history_size: int = Field(default=10, description="max user history size")

    cues_prompt: str = Field(default="", description="cues prompt")

    # 快照配置
    snapshot_interval: int = Field(default=5, description="快照生成间隔（秒）")
    max_events_in_snapshot: int = Field(default=20, description="每次快照最多包含的事件数")
    event_retention_seconds: int = Field(default=300, description="事件保留时间（秒）")

    # 定时任务配置
    periodic_task_interval: int = Field(default=60, description="定时任务触发间隔（秒）")
    periodic_task_overdue: int = Field(default=30, description="定时任务触发间隔（秒）")
    periodic_task_prompt: str = Field(default="", description="定时任务的提示词")

    # 新增配置：去重和过滤
    deduplication_window: int = Field(default=120, description="去重时间窗口（秒）")
    max_enter_events_per_snapshot: int = Field(default=3, description="每个快照最多包含的进入事件数")
    enter_event_cooldown: int = Field(default=30, description="同一用户进入事件的冷却时间（秒）")

    # 新增配置：智能过滤
    enable_smart_filtering: bool = Field(default=True, description="启用智能过滤")
    min_chat_length_for_response: int = Field(default=3, description="需要响应的弹幕最小长度")
    max_events_per_user: int = Field(default=2, description="同一用户在每个快照中的最大事件数")


class DouyinLiveEventType(enum.Enum):
    super_gift = "super_gift"
    medium_gift = "medium_gift"
    small_gift = "small_gift"
    chat = "chat"
    enter = "enter"
    like = "like"
    social = "social"
    periodic = "periodic"  # 新增：定时任务事件类型

    def to_natural(self):
        return {
            DouyinLiveEventType.super_gift: "送出超级礼物",
            DouyinLiveEventType.medium_gift: "送出中等礼物",
            DouyinLiveEventType.small_gift: "送出小礼物",
            DouyinLiveEventType.chat: "说",
            DouyinLiveEventType.enter: "进入直播间",
            DouyinLiveEventType.social: "点击关注",
            DouyinLiveEventType.periodic: "定时任务触发"
        }.get(self, self.value)


# ========== 直播间的历史记录 ==========
class DouyinLiveEvent(BaseModel):
    # 唯一标识符，用于去重
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="事件唯一ID")

    # 用户id
    user_id: str = Field(default_factory=str, description="douyin user id")
    # 用户名
    user_name: str = Field(default_factory=str, description="douyin user name")
    event_type: DouyinLiveEventType = Field(description="type")
    content: str = Field(default_factory=str, description="user do what")

    # 是否已被处理（用于去重）
    processed: bool = Field(default=False, description="是否已被处理")
    processed_by: Optional[str] = Field(default=None, description="被哪个Agent处理")
    processed_at: Optional[int] = Field(default=None, description="处理时间")

    create_at: int = Field(default_factory=lambda: int(time.time()), description="time")

    # 新增字段
    priority: Optional[Priority] = Field(default=None, description="事件优先级")
    bar_type: Optional[str] = Field(default=None, description="弹幕类型")

    def to_natural(self):
        local_time = time.localtime(self.create_at)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        res = f"{time_str} {self.user_name}#{self.user_id} {self.event_type.to_natural()}"
        if self.content:
            res = f"{res} {self.content}"
        return res

    def mark_processed(self, agent_name: str = "main"):
        """标记为已处理"""
        self.processed = True
        self.processed_by = agent_name
        self.processed_at = int(time.time())

    def is_recent(self, seconds: int = 60) -> bool:
        """判断事件是否在最近seconds秒内发生"""
        return time.time() - self.create_at <= seconds


# 单个用户所有的交互历史
class DouyinLiveUserHistory(BaseModel):
    # 用户id
    user_id: str = Field(default_factory=str, description="douyin user id")
    # 用户名
    user_name: str = Field(default_factory=str, description="douyin user name")
    # 是否为重点用户，评论数多的，送礼多的，进入直播间次数多的等
    is_core_user: bool = Field(default=False, description="is special user")
    # 历史发生过的所有事情
    history: List[DouyinLiveEvent] = Field(default_factory=list, description="douyin user history")
    # 对该用户的综合评价
    assessment: str = Field(default_factory=str, description="assessment")
    # 最后进入时间（用于冷却）
    last_enter_time: Optional[int] = Field(default=None, description="最后进入时间")
    # 最后交互时间
    last_interaction_time: Optional[int] = Field(default=None, description="最后交互时间")
    # 新增送礼统计
    gift_statistics: Dict[str, int] = Field(default_factory=dict, description="送礼统计")
    total_gift_value: int = Field(default=0, description="累计送礼价值（钻石）")
    gift_count: int = Field(default=0, description="送礼总次数")
    last_gift_time: Optional[int] = Field(default=None, description="最近送礼时间")
    consecutive_gift_days: int = Field(default=0, description="连续送礼天数")

    # 新增用户关系标签
    relationship_tags: List[str] = Field(default_factory=list, description="用户关系标签")
    # 例如：["新观众", "常客", "土豪", "真爱粉", "榜一大哥"]

    def update_gift_statistics(self, gift_name: str, diamond_count: int, gift_count: int = 1):
        """更新送礼统计"""
        self.gift_count += gift_count
        self.total_gift_value += diamond_count * gift_count
        self.last_gift_time = int(time.time())

        # 更新具体礼物统计
        if gift_name not in self.gift_statistics:
            self.gift_statistics[gift_name] = 0
        self.gift_statistics[gift_name] += gift_count

        # 更新关系标签
        self._update_relationship_tags()

    def _update_relationship_tags(self):
        """根据送礼情况更新关系标签"""
        tags = []

        if self.gift_count == 1:
            tags.append("新观众")
        elif self.gift_count <= 5:
            tags.append("常客")
        elif self.gift_count <= 20:
            tags.append("活跃粉丝")
        else:
            tags.append("真爱粉")

        if self.total_gift_value >= 1000:
            tags.append("土豪")
        if self.total_gift_value >= 5000:
            tags.append("榜一大哥")

        self.relationship_tags = tags

    def get_gift_summary(self) -> str:
        """获取送礼摘要"""
        if self.gift_count == 0:
            return "这是该用户第一次送礼"

        summary = f"用户已累计送礼{self.gift_count}次，总价值{self.total_gift_value}钻石"

        # 添加最喜欢的礼物
        if self.gift_statistics:
            favorite_gift = max(self.gift_statistics.items(), key=lambda x: x[1])
            summary += f"，最喜欢送{favorite_gift[0]}（{favorite_gift[1]}次）"

        # 添加时间信息
        if self.last_gift_time:
            minutes_ago = (time.time() - self.last_gift_time) // 60
            if minutes_ago < 60:
                summary += f"，{int(minutes_ago)}分钟前刚送过礼"

        return summary

    def get_history_events(self, *event_types: DouyinLiveEventType, max_count=3) -> List[DouyinLiveEvent]:
        res = []
        count = 0
        # 从后往前遍历
        for event in reversed(self.history):
            # 不传event_types默认全部
            if len(event_types) == 0 or event.event_type in event_types:
                res.append(event)
                count += 1
            if count >= max_count:
                break
        # 再重置顺序
        res.reverse()
        return res

    def update_last_enter_time(self):
        """更新最后进入时间"""
        self.last_enter_time = int(time.time())

    def update_last_interaction_time(self):
        """更新最后交互时间"""
        self.last_interaction_time = int(time.time())


@dataclass
class EventBuffer:
    """事件缓冲区，用于存储和管理事件"""
    events: deque = field(default_factory=deque)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # 去重集合：存储最近处理过的事件ID
    processed_events: Set[str] = field(default_factory=set)
    # 用户冷却时间记录
    user_cooldowns: Dict[Tuple[str, str], float] = field(default_factory=dict)
    # 事件指纹集合（用于内容去重）
    event_fingerprints: Dict[str, float] = field(default_factory=dict)

    def add(self, event: DouyinLiveEvent):
        """添加事件到缓冲区"""
        self.events.append(event)

    def get_recent_events(self, max_count: int = 20) -> List[DouyinLiveEvent]:
        """获取最近的事件"""
        count = min(max_count, len(self.events))
        return list(self.events)[-count:]

    def clear_old_events(self, retention_seconds: int):
        """清理过期事件"""
        current_time = time.time()
        cutoff_time = current_time - retention_seconds

        # 找到第一个未过期的事件索引
        first_valid_index = 0
        for i, event in enumerate(self.events):
            if event.create_at >= cutoff_time:
                first_valid_index = i
                break

        # 清理过期事件
        if first_valid_index > 0:
            for _ in range(first_valid_index):
                self.events.popleft()

        # 清理过期的去重记录
        self.processed_events = {
            event.event_id for event in self.events
            if event.event_id in self.processed_events
        }

        # 清理过期的冷却记录
        current_time = time.time()
        self.user_cooldowns = {
            key: cooldown for key, cooldown in self.user_cooldowns.items()
            if cooldown > current_time
        }

        # 清理过期的指纹记录
        self.event_fingerprints = {
            fingerprint: timestamp for fingerprint, timestamp in self.event_fingerprints.items()
            if timestamp > current_time
        }

    def clear_all(self):
        """清空所有事件"""
        self.events.clear()
        self.processed_events.clear()
        self.user_cooldowns.clear()
        self.event_fingerprints.clear()

    def is_processed(self, event_id: str) -> bool:
        """检查事件是否已被处理"""
        return event_id in self.processed_events

    def mark_processed(self, event_id: str):
        """标记事件为已处理"""
        self.processed_events.add(event_id)

    def check_cooldown(self, user_id: str, user_name: str, cooldown_seconds: int) -> bool:
        """检查用户是否在冷却期内"""
        key = (user_id, user_name)
        current_time = time.time()

        if key in self.user_cooldowns:
            if current_time < self.user_cooldowns[key]:
                return False  # 还在冷却期

        # 设置新的冷却时间
        self.user_cooldowns[key] = current_time + cooldown_seconds
        return True

    def check_content_duplicate(self, content: str, deduplication_window: int = 120) -> bool:
        """检查内容是否重复（用于弹幕去重）"""
        if not content:
            return False

        # 创建内容指纹（简单哈希）
        fingerprint = f"content_{hash(content) & 0xFFFFFFFF}"
        current_time = time.time()

        if fingerprint in self.event_fingerprints:
            if current_time - self.event_fingerprints[fingerprint] < deduplication_window:
                return True  # 内容重复

        # 记录新的指纹
        self.event_fingerprints[fingerprint] = current_time
        return False


class DouyinLive(DouyinLiveWebFetcher):

    def __init__(self, eventbus: EventBus, history_storage: Storage, config: DouyinLiveConfig,
                 logger: LoggerItf = None):
        super().__init__(config.live_id, logger=logger)
        self.eventbus = eventbus
        self.history_storage = history_storage.sub_storage(f"live_id_{config.live_id}")
        self.config = config
        self.logger = logger or logging.getLogger("DouyinLive")

        # 分类器
        self.classifier = BarrageClassifier()

        # 事件缓冲区
        self.event_buffer = EventBuffer()

        # 实时处理队列（P0和礼物）
        self.realtime_queue: asyncio.Queue[DouyinLiveEvent] = asyncio.Queue()
        self._realtime_task: Optional[asyncio.Task] = None

        # 快照任务
        self._snapshot_task: Optional[asyncio.Task] = None

        # 保存任务
        self.save_queue: asyncio.Queue[Tuple[str, str, List[DouyinLiveEvent]]] = asyncio.Queue()
        self.save_task: Optional[asyncio.Task] = None

        # 定时任务
        self._periodic_task: Optional[asyncio.Task] = None

        # 当前直播间的人数
        self.current_users = 0
        self.total_users = 0

        self._ws_task: Optional[asyncio.Task] = None

        # 当前快照
        self._current_snapshot: List[Message] = []
        self._snapshot_lock = asyncio.Lock()

        # 生命周期
        self._start_lock = asyncio.Lock()
        self._is_stopped = asyncio.Event()

        # 统计信息
        self.stats = {
            "total_events": 0,
            "processed_events": 0,
            "filtered_events": 0,
            "last_snapshot_time": 0,
            "last_periodic_time": 0
        }

    async def start(self):
        """
        避免重复启动
        """
        async with self._start_lock:
            if not self._ws_task:
                self._ws_task = asyncio.create_task(self._connectWebSocket())
            if not self.save_task:
                self.save_task = asyncio.create_task(self._run_save())
            if not self._realtime_task:
                self._realtime_task = asyncio.create_task(self._run_realtime_processing())
            if not self._snapshot_task:
                self._snapshot_task = asyncio.create_task(self._run_snapshot_generator())
            if not self._periodic_task:
                self._periodic_task = asyncio.create_task(self._run_periodic_task())  # 启动定时任务

    async def stop(self):
        if self._is_stopped.is_set():
            return
        self._is_stopped.set()

        if hasattr(self, "ws"):
            self.ws.close()
        if self._ws_task:
            self._ws_task.cancel()
        if self.save_task:
            self.save_task.cancel()
        if self._realtime_task:
            self._realtime_task.cancel()
        if self._snapshot_task:
            self._snapshot_task.cancel()
        if self._periodic_task:
            self._periodic_task.cancel()

    def _should_filter_enter_event(self, user_id: str, user_name: str) -> bool:
        """判断是否应该过滤进入事件"""
        if not self.config.enable_smart_filtering:
            return False

        # 检查冷却时间
        if not self.event_buffer.check_cooldown(user_id, user_name, self.config.enter_event_cooldown):
            self.logger.debug(f"过滤进入事件：{user_name} 在冷却期内")
            return True

        return False

    def _parseChatMsg(self, payload):
        message = ChatMessage().parse(payload)

        # 内容去重检查
        if self.event_buffer.check_content_duplicate(message.content, self.config.deduplication_window):
            self.logger.debug(f"过滤重复弹幕：{message.content}")
            self.stats["filtered_events"] += 1
            return

        event = DouyinLiveEvent(
            user_id=str(message.user.id),
            user_name=message.user.nick_name,
            event_type=DouyinLiveEventType.chat,
            content=message.content,
        )

        # 分类弹幕
        bar_type, priority = self.classifier.classify(message.content)
        event.priority = priority

        # 智能过滤：过短的弹幕不实时处理
        if self.config.enable_smart_filtering and len(
                message.content.strip()) < self.config.min_chat_length_for_response:
            event.priority = Priority.P3
            self.stats["filtered_events"] += 1

        # 添加到事件缓冲区
        self.event_buffer.add(event)
        self.stats["total_events"] += 1

        # P0事件放入实时处理队列
        if priority == Priority.P0:
            self.realtime_queue.put_nowait(event)

    def _parseGiftMsg(self, payload):
        message = GiftMessage().parse(payload)
        self.logger.info(f"收到礼物：{message}")
        gift_name = message.gift.name
        gift_cnt = message.combo_count
        diamond_count = message.gift.diamond_count

        if diamond_count <= 20:  # 小礼物
            event_type = DouyinLiveEventType.small_gift
        elif diamond_count <= 100:  # 中礼物
            event_type = DouyinLiveEventType.medium_gift
        else:  # 大礼物
            event_type = DouyinLiveEventType.super_gift

        event = DouyinLiveEvent(
            user_id=str(message.user.id),
            user_name=message.user.nick_name,
            event_type=event_type,
            content=f"{gift_name}（钻石={diamond_count}）x{gift_cnt}",
            priority=Priority.P0  # 礼物都作为P0处理
        )

        # 添加到事件缓冲区
        self.event_buffer.add(event)
        self.stats["total_events"] += 1

        # 放入实时处理队列
        self.realtime_queue.put_nowait(event)

    def _parseLikeMsg(self, payload):
        message = LikeMessage().parse(payload)
        event = DouyinLiveEvent(
            user_id=str(message.user.id),
            user_name=message.user.nick_name,
            event_type=DouyinLiveEventType.like,
            content=f"{message.count}次",
            priority=Priority.P3  # 点赞作为P3
        )

        # 添加到事件缓冲区
        self.event_buffer.add(event)
        self.stats["total_events"] += 1

    def _parseMemberMsg(self, payload):
        message = MemberMessage().parse(payload)

        # 智能过滤：检查是否应该过滤进入事件
        if self._should_filter_enter_event(str(message.user.id), message.user.nick_name):
            self.stats["filtered_events"] += 1
            return

        event = DouyinLiveEvent(
            user_id=str(message.user.id),
            user_name=message.user.nick_name,
            event_type=DouyinLiveEventType.enter,
            priority=Priority.P3  # 进入作为P3
        )

        # 添加到事件缓冲区
        self.event_buffer.add(event)
        self.stats["total_events"] += 1

    def _parseSocialMsg(self, payload):
        message = SocialMessage().parse(payload)
        event = DouyinLiveEvent(
            user_id=str(message.user.id),
            user_name=message.user.nick_name,
            event_type=DouyinLiveEventType.social,
            priority=Priority.P1  # 关注作为P1
        )

        # 添加到事件缓冲区
        self.event_buffer.add(event)
        self.stats["total_events"] += 1

    def _parseRoomUserSeqMsg(self, payload):
        message = RoomUserSeqMessage().parse(payload)
        self.current_users = message.total
        self.total_users = message.total_pv_for_anchor

    @staticmethod
    def get_history_storage_filename(user_id: str, user_name: str) -> str:
        return f"live_user_history_{user_name}#{user_id}.json"

    async def get_user_history(self, user_id, user_name) -> DouyinLiveUserHistory:
        filename = self.get_history_storage_filename(user_id, user_name)
        if not self.history_storage.exists(filename):
            return DouyinLiveUserHistory(user_id=user_id, user_name=user_name)
        history_str = self.history_storage.get(filename)
        history_dict = json.loads(history_str)
        return DouyinLiveUserHistory.model_validate(history_dict)

    async def save_user_history(self, history: DouyinLiveUserHistory):
        history_dict = history.model_dump()
        history_str = json.dumps(history_dict, indent=4, ensure_ascii=False, cls=EnumEncoder)
        self.history_storage.put(
            self.get_history_storage_filename(user_id=history.user_id, user_name=history.user_name),
            history_str.encode("utf8"),
        )

    # ========= 实时处理 =========
    async def _run_realtime_processing(self):
        """处理实时事件（P0和礼物）"""
        try:
            while True:
                event = await self.realtime_queue.get()

                # 检查是否已被处理
                if self.event_buffer.is_processed(event.event_id):
                    self.logger.debug(f"跳过已处理事件: {event.event_id}")
                    self.realtime_queue.task_done()
                    continue

                # 标记为已处理（由MainAgent处理）
                event.mark_processed("main")
                self.event_buffer.mark_processed(event.event_id)
                self.stats["processed_events"] += 1

                # 获取用户历史
                user_history = await self.get_user_history(event.user_id, event.user_name)

                # 构建消息内容
                message_content = []

                # 根据事件类型选择不同的prompt
                if event.event_type in [DouyinLiveEventType.super_gift,
                                        DouyinLiveEventType.medium_gift,
                                        DouyinLiveEventType.small_gift]:
                    prompt = self.config.gift_prompt
                    priority = 99  # 礼物最高优先级
                    message_content.append(Text(text=prompt))
                else:
                    prompt = self.config.p0_prompt
                    priority = 0  # P0事件优先级
                    message_content.append(Text(text=prompt))

                # 添加当前事件
                message_content.append(Text(text=event.to_natural()))

                # 添加用户历史信息（如果存在）
                if user_history.history:
                    # 获取用户最近的历史事件（排除当前事件，最多3个）
                    recent_history = user_history.get_history_events(max_count=3)
                    if recent_history:
                        message_content.append(Text(text=f"\n用户 {event.user_name} 的历史交互记录："))
                        for hist_event in recent_history:
                            message_content.append(Text(text=f"  - {hist_event.to_natural()}"))

                # 如果是核心用户，添加综合评价
                if user_history.assessment:
                    message_content.append(Text(text=f"\n用户 {event.user_name} 的综合评价：{user_history.assessment}"))

                # 发送到事件总线
                await self.eventbus.put(UserInputAgentEvent(
                    message=Message.new(
                        role="user",
                        name=f"__douyin_live_{event.event_type.value}__"
                    ).with_content(*message_content),
                    priority=priority,
                    overdue=self.config.p0_overdue,
                    agent_id="main",  # 指定由MainAgent处理
                ))

                # 保存用户历史
                self.save_queue.put_nowait((event.user_id, event.user_name, [event]))

                self.realtime_queue.task_done()
        except asyncio.CancelledError:
            pass

    # ========= 定时任务 =========
    async def _run_periodic_task(self):
        """定时触发事件总线（由LiveAgent处理）"""
        try:
            while True:
                # 等待定时任务间隔
                await asyncio.sleep(self.config.periodic_task_interval)

                # 获取最近60秒内未处理的事件
                current_time = time.time()
                recent_unprocessed_events = []

                for event in self.event_buffer.get_recent_events(self.config.max_events_in_snapshot):
                    # 只考虑最近60秒内且未处理的事件
                    if event.is_recent(60) and not event.processed:
                        recent_unprocessed_events.append(event)

                if not recent_unprocessed_events:
                    # 如果没有未处理事件，LiveAgent可以分析整体情况
                    recent_events = [
                        event for event in self.event_buffer.get_recent_events(self.config.max_events_in_snapshot)
                        if event.is_recent(60)
                    ]
                    event_summary = self._summarize_events(recent_events)

                    await self.eventbus.put(UserInputAgentEvent(
                        message=Message.new(
                            role="user",
                            name="__douyin_live_periodic__"
                        ).with_content(
                            Text(text=self.config.periodic_task_prompt),
                            Text(text=f"\n最近60秒没有新的未处理事件，以下是整体情况："),
                            Text(text=f"当前在线人数：{self.current_users}"),
                            Text(
                                text=f"事件统计：{event_summary['chat_count']}条弹幕，{event_summary['enter_count']}人进入"),
                            Text(text=f"请分析当前直播状态并提供互动建议。")
                        ),
                        priority=0,
                        overdue=self.config.periodic_task_overdue,
                        agent_id="live",
                    ))
                else:
                    # 有未处理事件，LiveAgent分析这些事件
                    # 标记这些事件为"已分析"（但不一定是"已处理"）
                    for event in recent_unprocessed_events:
                        event.mark_processed("live")  # 或者用一个新的方法 mark_analyzed
                        self.event_buffer.mark_processed(event.event_id)

                    event_summary = self._summarize_events(recent_unprocessed_events)

                    await self.eventbus.put(UserInputAgentEvent(
                        message=Message.new(
                            role="user",
                            name="__douyin_live_periodic__"
                        ).with_content(
                            Text(text=self.config.periodic_task_prompt),
                            Text(text=f"\n发现{len(recent_unprocessed_events)}个未处理事件，请分析："),
                            Text(text=f"当前在线人数：{self.current_users}"),
                            Text(text=f"事件类型分布："),
                            *[Text(
                                text=f"- {event.event_type.to_natural()}: {event_summary.get(event.event_type.value, 0)}个")
                              for event in recent_unprocessed_events[:3]],
                            Text(text=f"\n请分析这些事件并提供互动建议。")
                        ),
                        priority=0,
                        overdue=self.config.periodic_task_overdue,
                        agent_id="live",
                    ))

                self.stats["last_periodic_time"] = int(current_time)
                self.logger.info(f"定时任务触发，当前观看人数：{self.current_users}")

        except asyncio.CancelledError:
            pass

    # ========= 快照生成 =========
    async def _run_snapshot_generator(self):
        """定期生成直播间快照（仅提供上下文，不处理事件）"""
        try:
            while True:
                await asyncio.sleep(self.config.snapshot_interval)

                # 清理过期事件
                self.event_buffer.clear_old_events(self.config.event_retention_seconds)

                # 生成新的快照
                await self._generate_snapshot()
        except asyncio.CancelledError:
            pass

    # ========= 快照生成 =========
    async def _generate_snapshot(self):
        """生成直播间快照（包含事件列表和统计信息）"""
        # 获取最近的事件
        recent_events = self.event_buffer.get_recent_events(self.config.max_events_in_snapshot)

        # 生成统计信息
        event_summary = self._summarize_events(recent_events)

        # 创建快照消息
        messages = []

        # 1. 概览消息（包含统计信息）
        overview_msg = Message.new(role="user", name="__douyin_live_overview__").with_content(
            Text(text="=== 直播间状态概览 ==="),
            Text(text=f"当前在线人数：{self.current_users}，累计观看：{self.total_users}"),
            Text(text=f"最近{self.config.max_events_in_snapshot}个事件统计："),
            Text(text=f"• 弹幕：{event_summary['chat_count']}条"),
            Text(text=f"• 进入：{event_summary['enter_count']}人"),
            Text(text=f"• 点赞：{event_summary['like_count']}次"),
            Text(text=f"• 礼物：{event_summary['gift_count']}个"),
            Text(text=f"• 关注：{event_summary['social_count']}人"),
            Text(text=f"• 未处理事件：{event_summary['unprocessed_count']}个"),
        )

        # 如果有活跃用户，添加活跃用户信息
        if event_summary['active_users']:
            overview_msg.with_content(
                Text(text=f"\n活跃用户（最近{len(recent_events)}个事件）："),
                *[Text(text=f"• {user}（{count}次互动）") for user, count in event_summary['active_users'][:3]]
            )

        messages.append(overview_msg)

        # 2. 事件详情消息（按用户分组）
        if recent_events:
            # 按用户分组
            group_by_user = defaultdict(list)
            for event in recent_events:
                group_by_user[(event.user_id, event.user_name)].append(event)

            # 为每个活跃用户创建消息
            for (user_id, user_name), events in group_by_user.items():
                # 智能过滤：限制每个用户的事件数量
                if self.config.enable_smart_filtering and len(events) > self.config.max_events_per_user:
                    events = events[-self.config.max_events_per_user:]

                # 创建用户交互消息
                user_msg = Message.new(role="user", name=f"__douyin_live_{user_name}#{user_id}_interaction__")

                # 添加事件列表
                event_contents = []
                for event in events:
                    status = "✓" if event.processed else "○"
                    processor = f"({event.processed_by})" if event.processed_by else ""
                    event_contents.append(Text(text=f"[{status}{processor}] {event.to_natural()}"))

                user_msg.with_content(
                    Text(text=f"用户 {user_name} 的最近互动："),
                    *event_contents
                )

                messages.append(user_msg)

                # 异步保存到本地文件
                self.save_queue.put_nowait((user_id, user_name, events))

        # 更新当前快照
        async with self._snapshot_lock:
            self._current_snapshot = messages

        self.stats["last_snapshot_time"] = int(time.time())

    def _summarize_events(self, events: List[DouyinLiveEvent]) -> Dict:
        """汇总事件统计信息"""
        summary = {
            "chat_count": 0,
            "enter_count": 0,
            "like_count": 0,
            "gift_count": 0,
            "social_count": 0,
            "unprocessed_count": 0,
            "active_users": []  # (用户名, 互动次数)
        }

        user_activity = defaultdict(int)

        for event in events:
            # 统计事件类型
            if event.event_type == DouyinLiveEventType.chat:
                summary["chat_count"] += 1
            elif event.event_type == DouyinLiveEventType.enter:
                summary["enter_count"] += 1
            elif event.event_type == DouyinLiveEventType.like:
                summary["like_count"] += 1
            elif event.event_type in [DouyinLiveEventType.super_gift,
                                      DouyinLiveEventType.medium_gift,
                                      DouyinLiveEventType.small_gift]:
                summary["gift_count"] += 1
            elif event.event_type == DouyinLiveEventType.social:
                summary["social_count"] += 1

            # 统计未处理事件
            if not event.processed:
                summary["unprocessed_count"] += 1

            # 统计用户活跃度
            user_activity[event.user_name] += 1

        # 按活跃度排序
        summary["active_users"] = sorted(
            user_activity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return summary

    # ========= 保存任务 =========
    async def _run_save(self):
        """保存用户历史"""
        try:
            while True:
                user_id, user_name, events = await self.save_queue.get()

                history = await self.get_user_history(user_id, user_name)
                history.history.extend(events)

                # 更新最后交互时间
                if events:
                    history.update_last_interaction_time()

                    # 如果有进入事件，更新最后进入时间
                    for event in events:
                        if event.event_type == DouyinLiveEventType.enter:
                            history.update_last_enter_time()
                            break

                # 判断是否为核心用户
                if not history.is_core_user:
                    # 检查互动次数
                    chat_like_events = history.get_history_events(
                        DouyinLiveEventType.chat,
                        DouyinLiveEventType.like,
                        max_count=3,
                    )
                    if len(chat_like_events) >= 3:
                        history.is_core_user = True

                    # 检查是否有重要行为
                    important_events = history.get_history_events(
                        DouyinLiveEventType.super_gift,
                        DouyinLiveEventType.medium_gift,
                        DouyinLiveEventType.small_gift,
                        DouyinLiveEventType.social,
                        max_count=1
                    )
                    if len(important_events) >= 1:
                        history.is_core_user = True

                await self.save_user_history(history)
                self.save_queue.task_done()
        except asyncio.CancelledError:
            pass

    # ========= 上下文消息 =========
    async def context_messages(self) -> List[Message]:
        """
        获取当前直播间的上下文消息（快照）
        """
        async with self._snapshot_lock:
            return self._current_snapshot.copy()

    async def give_cues(self, text__):
        """
        给主Agent提供直播话术建议
        :param text__:
        """
        await self.eventbus.put(UserInputAgentEvent(
            message=Message.new(role="user", name="__douyin_live_cues__").with_content(
                Text(text=self.config.cues_prompt),
                Text(text=text__)
            ),
            agent_id="main",  # 默认主agent
            priority=0,
            overdue=30,
        ))

    def as_channel(self, is_live_agent: bool = False):
        chan = PyChannel(name="douyin_live", description="抖音直播间内可操作的command和上下文获取通道", blocking=True)

        chan.build.context_messages(self.context_messages)

        if is_live_agent:
            chan.build.command()(self.give_cues)

        # 生命周期
        chan.build.start_up(self.start)
        chan.build.close(self.stop)

        return chan


class DouyinLiveProvider(Provider[DouyinLive]):

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        eventbus = con.force_fetch(EventBus)
        ws = con.force_fetch(Workspace)
        _storage: FileStorage | Storage = ws.configs().sub_storage("douyin_live")
        config = WorkspaceConfigs(_storage).get_or_create(DouyinLiveConfig())

        return DouyinLive(
            eventbus=eventbus,
            history_storage=ws.runtime().sub_storage("live_memory"),
            config=config
        )