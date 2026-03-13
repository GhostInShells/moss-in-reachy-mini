import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from framework.apps.live.abcd.live_event import EventTransformer, LiveEvent


class LivePlatform(ABC):
    """
    直播平台抽象接口
    不再使用callback，改为事件队列模式
    """

    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        self._event_queue = asyncio.Queue()
        self._is_connected = False
        self._event_transformer: Optional[EventTransformer] = None

    @abstractmethod
    async def connect(self) -> None:
        """
        连接直播平台
        实现应该建立连接并开始接收事件
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    async def fetch_room_info(self) -> Dict[str, Any]:
        """
        获取房间信息
        返回包含在线人数、累计观看等信息的字典
        """
        pass

    def get_event_queue(self) -> asyncio.Queue:
        """
        获取事件队列
        LiveAssistant从这个队列中获取事件
        """
        return self._event_queue

    async def put_event(self, event: LiveEvent) -> None:
        """
        将事件放入队列
        平台实现应该在收到事件后调用此方法
        """
        await self._event_queue.put(event)

    def set_event_transformer(self, transformer: EventTransformer) -> None:
        """设置事件转换器"""
        self._event_transformer = transformer

    async def transform_and_put_event(self, raw_event: Dict[str, Any]) -> None:
        """
        转换并放入事件
        平台实现调用此方法将原始事件转换为通用事件
        """
        if self._event_transformer:
            event = self._event_transformer.transform(raw_event)
            if event:
                await self.put_event(event)

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._is_connected

    def get_platform_config_template(self) -> Dict[str, Any]:
        """
        获取平台配置模板
        用于生成配置UI
        """
        return {
            "required_fields": [],
            "optional_fields": [],
            "description": f"{self.platform_name} 直播平台配置"
        }