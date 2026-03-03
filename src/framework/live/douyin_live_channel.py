import asyncio
import random
import threading
from typing import List, Any, Optional

from ghoshell_container import IoCContainer, INSTANCE
from ghoshell_moss import Message, ContentModel, Text, PyChannel

from framework.abcd.agent import EventBus
from framework.abcd.agent_event import AgentEventModel, UserInputAgentEvent, ReactAgentEvent
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


class DouyinLiveChannel(DouyinLiveWebFetcher):

    def __init__(self, live_id, container: IoCContainer):
        super().__init__(live_id)
        self.container = container
        self.event_bus = self.container.get(EventBus)

        self.stopped = asyncio.Event()
        self.queue: BatchAsyncQueue[ContentModel] = BatchAsyncQueue()

        self._main_loop_task: Optional[asyncio.Task] = None


        self._current_users = 0
        self._total_users = 0

    async def main_loop(self):
        while not self.stopped.is_set():
            await asyncio.sleep(0.5)
            contents = await self.queue.get_batch(max_count=10)
            await self.event_bus.put(UserInputAgentEvent(
                message=Message.new(role="user").with_content(
                    Text(text="你需要做用简短的语言加上匹配的动作来响应用户，以下是用户间的互动消息，回复最好带上用户的名字哦"),
                    *contents
                ),
                priority=-1,
            ).to_agent_event())

    def start(self):
        self._main_loop_task = asyncio.create_task(self.main_loop())
        threading.Thread(target=self._connectWebSocket).start()

    def close(self):
        self.stopped.set()

    def _parseChatMsg(self, payload):
        message = ChatMessage().parse(payload)
        user_name = message.user.nick_name
        content = message.content
        self.queue.put_nowait(Text(text=f"用户[{user_name}]说: {content}"))

    def _parseGiftMsg(self, payload):
        message = GiftMessage().parse(payload)
        user_name = message.user.nick_name
        gift_name = message.gift.name
        gift_cnt = message.combo_count
        self.queue.put_nowait(Text(text=f"用户[{user_name}]送出了 {gift_name}x{gift_cnt}"))

    def _parseLikeMsg(self, payload):
        message = LikeMessage().parse(payload)
        user_name = message.user.nick_name
        count = message.count
        self.queue.put_nowait(Text(text=f"用户[{user_name}]点了{count}个赞"))

    def _parseMemberMsg(self, payload):
        message = MemberMessage().parse(payload)
        user_name = message.user.nick_name
        # 50% 概率忽略用户进入直播间的消息
        if random.random() < 0.5:
            return
        self.queue.put_nowait(Text(text=f"用户[{user_name}] 进入了直播间"))

    def _parseSocialMsg(self, payload):
        message = SocialMessage().parse(payload)
        user_name = message.user.nick_name
        self.queue.put_nowait(Text(text=f"用户[{user_name}]关注了你"))

    def _parseRoomUserSeqMsg(self, payload):
        message = RoomUserSeqMessage().parse(payload)
        self._current_users = message.total
        self._total_users = message.total_pv_for_anchor

    async def context_messages(self):
        msg = Message.new(role="system").with_content(
            Text(text=f"你正在抖音里进行直播，当前观看人数: {self._current_users}, 累计观看人数: {self._total_users}")
        )
        return [msg]

    def as_channel(self):
        chan = PyChannel(name="douyin_live")
        chan.build.with_context_messages(self.context_messages)
        return chan
        