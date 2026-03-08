import asyncio
import enum
import json
import logging
import random
import threading
import time
from collections import defaultdict
from typing import List, Optional, Tuple, Dict

from ghoshell_common.contracts import YamlConfig, Storage, WorkspaceConfigs, FileStorage, Workspace, LoggerItf
from ghoshell_container import INSTANCE, Provider, IoCContainer
from ghoshell_moss import Message, Text, PyChannel
from pydantic import Field, BaseModel

from framework.abcd.agent import EventBus
from framework.abcd.agent_event import UserInputAgentEvent
from framework.apps.live.DouyinLiveWebFetcher.liveMan import DouyinLiveWebFetcher
from framework.apps.live.DouyinLiveWebFetcher.protobuf.douyin import ChatMessage, GiftMessage, LikeMessage, \
    MemberMessage, \
    SocialMessage, RoomUserSeqMessage
from framework.apps.utils import EnumEncoder


class DouyinLiveConfig(YamlConfig):
    relative_path = "douyin_live_config.yaml"

    live_id: str = Field(default="", description="douyin live id")

    gift_prompt: str = Field(default="", description="douyin gift prompt")
    idle_react_threshold: int = Field(default=10, description="idle threshold")
    idle_prompts: List[str] = Field(default_factory=list, description="idle prompts")

    max_user_history_size: int = Field(default=10, description="max user history size")


class DouyinLiveEventType(enum.Enum):
    super_gift = "super_gift"
    medium_gift = "medium_gift"
    small_gift = "small_gift"
    chat = "chat"
    enter = "enter"
    like = "like"
    social = "social"

    def to_natural(self):
        return {
            DouyinLiveEventType.super_gift: "送出超级礼物",
            DouyinLiveEventType.medium_gift: "送出中等礼物",
            DouyinLiveEventType.small_gift: "送出小礼物",
            DouyinLiveEventType.chat: "说",
            DouyinLiveEventType.enter: "进入直播间",
            DouyinLiveEventType.social: "点击关注"
        }.get(self)


# ========== 直播间的历史记录 ==========
# 单个用户的单条交互历史
class DouyinLiveEvent(BaseModel):
    # 用户id
    user_id: str = Field(default_factory=str, description="douyin user id")
    # 用户名
    user_name: str = Field(default_factory=str, description="douyin user name")
    event_type: DouyinLiveEventType = Field(description="type")
    content: str = Field(default_factory=str, description="user do what")
    # assistant: str = Field(default_factory=str, description="assistant response")

    create_at: int = Field(default_factory=lambda :int(time.time()), description="time")

    def to_natural(self):
        local_time = time.localtime(self.create_at)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        res = f"{time_str} {self.user_name}#{self.user_id} {self.event_type.to_natural()}"
        if self.content:
            res = f"{res} {self.content}"
        return res

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


class DouyinLive(DouyinLiveWebFetcher):

    def __init__(self, eventbus: EventBus, history_storage: Storage, config: DouyinLiveConfig, logger: LoggerItf=None):
        super().__init__(config.live_id)
        self.eventbus = eventbus
        self.history_storage = history_storage.sub_storage(f"live_id_{config.live_id}")
        self.config = config
        self.logger = logger or logging.getLogger("DouyinLive")

        # 抖音直播间最新的事件队列，礼物队列需要特殊处理
        self.gift_queue: asyncio.Queue[DouyinLiveEvent] = asyncio.Queue()
        self.event_queue: asyncio.Queue[DouyinLiveEvent] = asyncio.Queue()

        # 已经看过的数据需要塞到save队列里保存到本地文件
        self.save_queue: asyncio.Queue[Tuple[str, str, List[DouyinLiveEvent]]] = asyncio.Queue()
        self.save_task: Optional[asyncio.Task] = None

        # 当前直播间人数
        self.current_users = 0
        self.total_users = 0

        self._thread: Optional[threading.Thread] = None
        self._ws_task: Optional[asyncio.Task] = None

    async def start(self):
        self._ws_task = asyncio.create_task(self._connectWebSocket())
        self.save_task = asyncio.create_task(self._run_save())

    async def stop(self):
        if hasattr(self, "ws"):
            self.ws.close()
        if self._ws_task:
            self._ws_task.cancel()
        if self.save_task:
            self.save_task.cancel()

    def _parseChatMsg(self, payload):
        message = ChatMessage().parse(payload)
        self.event_queue.put_nowait(DouyinLiveEvent(
            user_id=str(message.user.id),
            user_name= message.user.nick_name,
            event_type=DouyinLiveEventType.chat,
            content=message.content,
        ))

    def _parseGiftMsg(self, payload):
        message = GiftMessage().parse(payload)
        gift_name = message.gift.name
        gift_cnt = message.combo_count
        diamond_count = message.diamond_count

        if diamond_count <= 20: # 小礼物
            event_type = DouyinLiveEventType.small_gift
        elif diamond_count <= 100: # 中礼物
            event_type = DouyinLiveEventType.medium_gift
        else: # 大礼物
            event_type = DouyinLiveEventType.super_gift

        self.gift_queue.put_nowait(DouyinLiveEvent(
            user_id=str(message.user.id),
            user_name=message.user.nick_name,
            event_type=event_type,
            content=f"{gift_name}（钻石={diamond_count}）x{gift_cnt}"
        ))

    def _parseLikeMsg(self, payload):
        message = LikeMessage().parse(payload)
        self.event_queue.put_nowait(DouyinLiveEvent(
            user_id=str(message.user.id),
            user_name=message.user.nick_name,
            event_type=DouyinLiveEventType.like,
            content=f"{message.count}次",
        ))

    def _parseMemberMsg(self, payload):
        message = MemberMessage().parse(payload)
        self.event_queue.put_nowait(DouyinLiveEvent(
            user_id=str(message.user.id),
            user_name=message.user.nick_name,
            event_type=DouyinLiveEventType.enter,
        ))

    def _parseSocialMsg(self, payload):
        message = SocialMessage().parse(payload)
        self.event_queue.put_nowait(DouyinLiveEvent(
            user_id=str(message.user.id),
            user_name=message.user.nick_name,
            event_type=DouyinLiveEventType.social,
        ))

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

    # ========= ModelContext =========
    async def idle(self):
        try:
            while True:
                try:
                    gift_event = await asyncio.wait_for(self.gift_queue.get(), timeout=5)
                    await self.eventbus.put(UserInputAgentEvent(
                        message=Message.new(
                            role="user", name="__douyin_live_gift__"
                        ).with_content(
                            Text(text=self.config.gift_prompt),
                            Text(text=gift_event.to_natural())
                        ),
                        priority=1, # 高优队列
                    ).to_agent_event())
                    self.gift_queue.task_done()
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass

    async def _run_save(self):
        try:
            while True:
                user_id, user_name, events = await self.save_queue.get()
                history = await self.get_user_history(user_id, user_name)
                history.history.extend(events)

                # 判断是否为核心用户
                if not history.is_core_user:
                    if len(history.get_history_events(
                        DouyinLiveEventType.chat,
                        DouyinLiveEventType.like,
                        max_count=3,
                    )) >= 3:
                        history.is_core_user = True
                    elif len(history.get_history_events(
                        DouyinLiveEventType.super_gift,
                        DouyinLiveEventType.medium_gift,
                        DouyinLiveEventType.small_gift,
                        DouyinLiveEventType.social,
                        max_count=1
                    )) >= 1:
                        history.is_core_user = True

                await self.save_user_history(history)
                self.save_queue.task_done()
        except asyncio.CancelledError:
            pass

    async def context_messages(self):
        """
        当前直播间所有发生的事情
        """
        all_events: List[DouyinLiveEvent] = []
        try:
            while not self.event_queue.empty():
                all_events.append(self.event_queue.get_nowait())
                self.event_queue.task_done()
        except asyncio.QueueEmpty:
            pass

        messages = [
            Message.new(role="user", name="__douyin_live_overview__").with_content(
                Text(text=f"当前观看人数：{self.current_users}，累计观看人数：{self.total_users}")
            )
        ]

        # 按用户做一次聚合
        group_by_user: Dict[tuple, List[DouyinLiveEvent]] = defaultdict(list)
        for event in all_events:
            group_by_user[(event.user_id, event.user_name)].append(event)

        # 取出用户的历史信息
        for key, events in group_by_user.items():
            user_id = key[0]
            user_name = key[1]

            # 交互的用户超过20人的话 直接忽略掉进入直播间进入的事件
            if len(group_by_user) > 20 and len(events) == 1 and events[0].event_type == DouyinLiveEventType.enter:
                # 90%的概率忽略
                if random.random() < 0.9:
                    continue

            # 当前最新的直播间发生的交互行为
            message = Message.new(role="user", name=f"__douyin_live_{user_name}#{user_id}_interaction__")
            current_contents = []
            for event in events:
                current_contents.append(Text(text=event.to_natural()))
            message.with_content(
                Text(text=f"以下是当前用户{user_name}在直播间最新的交互行为"),
                *current_contents,
            )
            messages.append(message)

            history_contents = []
            history = await self.get_user_history(user_id, user_name)
            # 只有特殊用户再拿历史对话
            if history.is_core_user:
                # 最近 N 条数据，配置读取，默认最多10条
                for event in history.get_history_events(max_count=self.config.max_user_history_size):
                    history_contents.append(Text(text=event.to_natural()))

                message.with_content(
                    Text(text=f"以下是当前用户在直播间的历史近{self.config.max_user_history_size}次的交互行为"),
                    *history_contents,
                )
                # 对用户的评价，可以由旁路Agent分析并写入
                if history.assessment:
                    message.with_content(
                        Text(text=f"你的视角下对当前用户的综合评价为{history.assessment}")
                    )

                messages.append(message)
            # 异步存储到本地文件
            self.save_queue.put_nowait((user_id, user_name, events))

        return messages

    def as_channel(self):
        chan = PyChannel(name="douyin_live", description="抖音直播间内可操作的command和上下文获取通道", blocking=True)

        chan.build.idle(self.idle)
        chan.build.context_messages(self.context_messages)

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
