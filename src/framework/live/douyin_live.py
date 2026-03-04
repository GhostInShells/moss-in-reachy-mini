import asyncio
import threading
from typing import List, Optional

from ghoshell_common.contracts import YamlConfig
from ghoshell_container import INSTANCE
from ghoshell_moss import Message, ContentModel, Text
from pydantic import Field, BaseModel

from framework.abcd.agent_event import UserInputAgentEvent
from framework.live.DouyinLiveWebFetcher.liveMan import DouyinLiveWebFetcher
from framework.live.DouyinLiveWebFetcher.protobuf.douyin import ChatMessage, GiftMessage, LikeMessage, MemberMessage, \
    SocialMessage, RoomUserSeqMessage


class BatchAsyncQueue(asyncio.Queue[INSTANCE]):
    async def get_batch(self, max_count: int) -> List[INSTANCE]:
        """
        消费者接口：批量获取多条数据
        :param max_count: 最多获取的数量
        :return: 获取到的数据集（数量<=max_count）
        """
        items = []
        try:
            # 先获取第一条（确保有数据，无数据则等待）
            # 超时逻辑通过wait_for实现
            item = await self.get()
            items.append(item)

            # 循环获取剩余数据（非阻塞，有多少拿多少，直到达到max_count）
            while len(items) < max_count and not self.empty():
                # get_nowait() 非阻塞获取，不会等待新数据
                items.append(self.get_nowait())
        except asyncio.QueueEmpty:
            pass

        return items

    def get_batch_nowait(self, max_count: int) -> List[INSTANCE]:
        items = []
        try:
            while len(items) < max_count and not self.empty():
                items.append(self.get_nowait())
        except asyncio.QueueEmpty:
            pass
        return items

class DouyinLiveEventConfig(BaseModel):
    max_count: int = Field(default=1, description="max count")
    prompt: str = Field(default="", description="prompt")
    priority: int = Field(default=0, description="priority")
    overdue: int = Field(default=0, description="overdue")


class DouyinLiveConfig(YamlConfig):
    relative_path = "douyin_live_config.yaml"

    live_id: str = Field(default="", description="douyin live id")
    super_gift_event: DouyinLiveEventConfig = Field(default_factory=DouyinLiveEventConfig, description="super gift queue")
    medium_gift_event: DouyinLiveEventConfig = Field(default_factory=DouyinLiveEventConfig, description="medium gift queue")
    small_gift_event: DouyinLiveEventConfig = Field(default_factory=DouyinLiveEventConfig, description="small gift queue")
    chat_event: DouyinLiveEventConfig = Field(default_factory=DouyinLiveEventConfig, description="chat queue")
    like_event: DouyinLiveEventConfig = Field(default_factory=DouyinLiveEventConfig, description="like queue")
    enter_event: DouyinLiveEventConfig = Field(default_factory=DouyinLiveEventConfig, description="enter queue")
    social_event: DouyinLiveEventConfig = Field(default_factory=DouyinLiveEventConfig, description="social queue")

    idle_react_threshold: int = Field(default=10, description="idle threshold")
    idle_prompts: List[str] = Field(default_factory=list, description="idle prompts")


class DouyinLive(DouyinLiveWebFetcher):

    def __init__(self, config: DouyinLiveConfig):
        super().__init__(config.live_id)
        self.stopped = asyncio.Event()

        self.chat_queue: BatchAsyncQueue[ContentModel] = BatchAsyncQueue()
        self.small_git_queue: BatchAsyncQueue[ContentModel] = BatchAsyncQueue()
        self.medium_git_queue: BatchAsyncQueue[ContentModel] = BatchAsyncQueue()
        self.super_gift_queue: BatchAsyncQueue[ContentModel] = BatchAsyncQueue()
        self.like_queue: BatchAsyncQueue[ContentModel] = BatchAsyncQueue()
        self.enter_queue: BatchAsyncQueue[ContentModel] = BatchAsyncQueue()
        self.social_queue: BatchAsyncQueue[ContentModel] = BatchAsyncQueue()

        self.config = config

        self.current_users = 0
        self.total_users = 0

        self._thread: Optional[threading.Thread] = None

    def get_agent_events(self):
        high_gifts = self.super_gift_queue.get_batch_nowait(max_count=self.config.super_gift_event.max_count)
        if high_gifts:
            return [UserInputAgentEvent(
                message=Message.new(role="user").with_content(
                    Text(
                        text=self.config.super_gift_event.prompt
                    ),
                    *high_gifts
                ),
                priority=2,  # 高优先级，直接返回
            )]
        events = []
        event_queues = [
            (self.chat_queue, self.config.chat_event),
            (self.small_git_queue, self.config.small_gift_event),
            (self.medium_git_queue, self.config.medium_gift_event),
            (self.like_queue, self.config.like_event),
            (self.enter_queue, self.config.enter_event),
            (self.social_queue, self.config.social_event),
        ]
        for queue, config in event_queues:
            contents = queue.get_batch_nowait(max_count=config.max_count)
            if contents:
                events.append(UserInputAgentEvent(
                    message=Message.new(role="user").with_content(
                        Text(
                            text=config.prompt
                        ),
                        *contents
                    ),
                    priority=config.priority,
                    overdue=config.overdue
                ))
        return events

    def start(self):
        self._thread = threading.Thread(target=self._connectWebSocket)
        self._thread.start()

    def stop(self):
        if hasattr(self, "ws"):
            self.ws.close()
        if self._thread:
            self._thread.join()

    def _parseChatMsg(self, payload):
        message = ChatMessage().parse(payload)
        user_name = message.user.nick_name
        content = message.content
        self.chat_queue.put_nowait(Text(text=f"用户[{user_name}]说：{content}"))

    def _parseGiftMsg(self, payload):
        message = GiftMessage().parse(payload)
        user_name = message.user.nick_name
        gift_name = message.gift.name
        gift_cnt = message.combo_count
        diamond_count = message.diamond_count
        if diamond_count <= 20: # 小礼物
            self.small_git_queue.put_nowait(Text(text=f"用户[{user_name}]送出了{gift_name}x{gift_cnt}"))
        elif diamond_count <= 100: # 中礼物
            self.medium_git_queue.put_nowait(Text(text=f"用户[{user_name}]送出了{gift_name}x{gift_cnt}"))
        else: # 大礼物
            self.super_gift_queue.put_nowait(Text(text=f"用户[{user_name}]送出了{gift_name}x{gift_cnt}"))

    def _parseLikeMsg(self, payload):
        message = LikeMessage().parse(payload)
        user_name = message.user.nick_name
        count = message.count
        self.like_queue.put_nowait(Text(text=f"用户[{user_name}]点了{count}个赞"))

    def _parseMemberMsg(self, payload):
        message = MemberMessage().parse(payload)
        user_name = message.user.nick_name
        self.enter_queue.put_nowait(Text(text=f"用户[{user_name}]进入了直播间"))

    def _parseSocialMsg(self, payload):
        message = SocialMessage().parse(payload)
        user_name = message.user.nick_name
        self.social_queue.put_nowait(Text(text=f"用户[{user_name}]关注了你"))

    def _parseRoomUserSeqMsg(self, payload):
        message = RoomUserSeqMessage().parse(payload)
        self.current_users = message.total
        self.total_users = message.total_pv_for_anchor
