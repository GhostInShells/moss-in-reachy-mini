from typing import Dict, Optional, Any, List

from framework.apps.live.abcd.event import EventTransformer, LiveEvent, ChatEvent, GiftEvent, EnterEvent, LikeEvent, \
    FollowEvent, LiveEventType


class DouyinEventTransformer(EventTransformer):
    """抖音事件转换器"""

    def __init__(self):
        super().__init__("douyin")

    def transform(self, raw_data: Dict[str, Any]) -> Optional[LiveEvent]:
        """转换抖音原始数据"""
        data_type = raw_data.get("type", "")

        if data_type == "chat":
            return self._transform_chat(raw_data)
        elif data_type == "gift":
            return self._transform_gift(raw_data)
        elif data_type == "enter":
            return self._transform_enter(raw_data)
        elif data_type == "like":
            return self._transform_like(raw_data)
        elif data_type == "follow":
            return self._transform_follow(raw_data)

        return None

    def _transform_chat(self, raw_data: Dict[str, Any]) -> LiveEvent:
        """转换弹幕"""
        user = raw_data.get("user", {})

        event = ChatEvent(
            platform=self.platform,
            user_id=user.get("id", ""),
            user_name=user.get("name", ""),
            user_avatar=user.get("avatar", ""),
            content=raw_data.get("content", ""),
            metadata={"raw_data": raw_data}
        )

        # 设置优先级
        content_length = len(event.content.strip())
        if content_length < 3:
            event.priority = 1
        elif content_length < 10:
            event.priority = 10
        else:
            event.priority = 30

        return event.to_live_event()

    def _transform_gift(self, raw_data: Dict[str, Any]) -> LiveEvent:
        """转换礼物"""
        user = raw_data.get("user", {})
        gift = raw_data.get("gift", {})

        gift_value = gift.get("value", 0)
        is_super_gift = gift_value > 100

        event = GiftEvent(
            platform=self.platform,
            user_id=user.get("id", ""),
            user_name=user.get("name", ""),
            user_avatar=user.get("avatar", ""),
            gift_id=gift.get("id", ""),
            gift_name=gift.get("name", ""),
            gift_count=gift.get("count", 1),
            gift_value=gift_value,
            is_super_gift=is_super_gift,
            metadata={"raw_data": raw_data}
        )

        # 礼物优先级较高
        if is_super_gift:
            event.priority = 100
        elif gift_value > 50:
            event.priority = 80
        else:
            event.priority = 60

        return event.to_live_event()

    def _transform_enter(self, raw_data: Dict[str, Any]) -> LiveEvent:
        """转换进入事件"""
        user = raw_data.get("user", {})

        event = EnterEvent(
            platform=self.platform,
            user_id=user.get("id", ""),
            user_name=user.get("name", ""),
            user_avatar=user.get("avatar", ""),
            metadata={"raw_data": raw_data}
        )

        # 进入事件优先级较低
        event.priority = 5

        return event.to_live_event()

    def _transform_like(self, raw_data: Dict[str, Any]) -> LiveEvent:
        """转换点赞事件"""
        user = raw_data.get("user", {})

        event = LikeEvent(
            platform=self.platform,
            user_id=user.get("id", ""),
            user_name=user.get("name", ""),
            user_avatar=user.get("avatar", ""),
            like_count=raw_data.get("count", 1),
            metadata={"raw_data": raw_data}
        )

        # 点赞事件优先级较低
        event.priority = 3

        return event.to_live_event()

    def _transform_follow(self, raw_data: Dict[str, Any]) -> LiveEvent:
        """转换关注事件"""
        user = raw_data.get("user", {})

        event = FollowEvent(
            platform=self.platform,
            user_id=user.get("id", ""),
            user_name=user.get("name", ""),
            user_avatar=user.get("avatar", ""),
            metadata={"raw_data": raw_data}
        )

        # 关注事件优先级中等
        event.priority = 40

        return event.to_live_event()

    def get_supported_event_types(self) -> List[str]:
        """获取支持的事件类型"""
        return [
            LiveEventType.CHAT.value,
            LiveEventType.GIFT.value,
            LiveEventType.ENTER.value,
            LiveEventType.LIKE.value,
            LiveEventType.FOLLOW.value,
        ]

