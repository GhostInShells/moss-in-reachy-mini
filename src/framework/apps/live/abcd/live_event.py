from abc import ABC, abstractmethod
from typing import TypedDict, Optional, Dict, Any, List, Type, ClassVar, Self
from pydantic import BaseModel, Field
from enum import Enum
import time
import uuid


# ========== 事件类型定义 ==========

class LiveEventType(Enum):
    """直播事件类型枚举"""
    # 基础事件
    CHAT = "chat"  # 弹幕/评论
    GIFT = "gift"  # 送礼
    ENTER = "enter"  # 进入直播间
    LIKE = "like"  # 点赞
    FOLLOW = "follow"  # 关注
    SHARE = "share"  # 分享

    # 直播相关
    LIVE_START = "live_start"  # 直播开始
    LIVE_END = "live_end"  # 直播结束
    ROOM_INFO_UPDATE = "room_info_update"  # 房间信息更新

    # 系统事件
    SYSTEM = "system"  # 系统事件
    CUSTOM = "custom"  # 自定义事件

    # 平台特定事件（前缀区分）
    DOUYIN_SUPER_CHAT = "douyin_super_chat"  # 抖音超级弹幕
    BILIBILI_GUARD = "bilibili_guard"  # B站舰长
    WEIBO_VIP_ENTER = "weibo_vip_enter"  # 微博VIP进入


# ========== 事件数据结构 ==========

class LiveEvent(TypedDict):
    """直播事件数据结构"""
    event_id: str
    event_type: str
    platform: str
    priority: Optional[int]
    issuer: Optional[str]
    overdue: Optional[float]
    created: float

    # 事件数据
    data: Optional[Dict[str, Any]]


class LiveEventModel(BaseModel):
    """直播事件模型基类"""
    event_type: ClassVar[str] = ""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="事件唯一ID")
    platform: str = Field(default="", description="直播平台名称")
    priority: int = Field(
        default=1,
        description=(
            "事件优先级: "
            "1-10: 低优先级（普通弹幕、进入）"
            "11-50: 中优先级（关注、分享）"
            "51-99: 高优先级（礼物、重要弹幕）"
            "100+: 实时处理（P0弹幕、超级礼物）"
        )
    )
    issuer: str = Field(default="live_system", description="事件发起方")
    overdue: float = Field(
        default=0,
        description="过期时间（秒），0表示永不过期"
    )
    created: float = Field(
        default_factory=lambda: round(time.time(), 4),
        description="事件创建时间"
    )

    # 通用字段
    user_id: str = Field(default="", description="用户ID")
    user_name: str = Field(default="", description="用户名")
    user_avatar: str = Field(default="", description="用户头像")
    content: str = Field(default="", description="事件内容")

    # 处理状态
    processed: bool = Field(default=False, description="是否已处理")
    processed_by: Optional[str] = Field(default=None, description="处理者")
    processed_at: Optional[float] = Field(default=None, description="处理时间")

    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="事件元数据")

    def is_overdue(self, now: float = None) -> bool:
        """事件是否已过期"""
        if now is None:
            now = round(time.time(), 4)
        return self.overdue > 0 and ((now - self.created) > self.overdue)

    def mark_processed(self, processor: str = "main") -> None:
        """标记为已处理"""
        self.processed = True
        self.processed_by = processor
        self.processed_at = round(time.time(), 4)

    def to_live_event(self) -> LiveEvent:
        """转换为LiveEvent字典"""
        # 提取基础字段
        base_data = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "platform": self.platform,
            "priority": self.priority,
            "issuer": self.issuer,
            "overdue": self.overdue,
            "created": self.created,
        }

        # 提取自定义数据字段
        exclude_fields = {
            "event_id", "event_type", "platform", "priority",
            "issuer", "overdue", "created", "processed",
            "processed_by", "processed_at"
        }

        data = self.model_dump(
            exclude=exclude_fields,
            exclude_none=True
        )

        # 添加处理状态
        if self.processed:
            data["processed"] = True
            data["processed_by"] = self.processed_by
            data["processed_at"] = self.processed_at

        return LiveEvent(
            **base_data,
            data=data
        )

    @classmethod
    def from_live_event(cls, live_event: LiveEvent) -> Optional[Self]:
        """从LiveEvent字典创建实例"""
        if cls.event_type != live_event["event_type"]:
            return None

        # 合并基础字段和数据字段
        event_data = live_event.get("data", {})
        event_data.update({
            "event_id": live_event["event_id"],
            "platform": live_event["platform"],
            "priority": live_event["priority"],
            "issuer": live_event["issuer"],
            "overdue": live_event["overdue"],
            "created": live_event["created"],
        })

        return cls(**event_data)

    def to_natural_language(self) -> str:
        """转换为自然语言描述"""
        raise NotImplementedError


# ========== 具体事件类型 ==========

class ChatEvent(LiveEventModel):
    """弹幕事件"""
    event_type: ClassVar[str] = LiveEventType.CHAT.value

    # 弹幕特有字段
    bar_type: Optional[str] = Field(default=None, description="弹幕类型")
    is_super_chat: bool = Field(default=False, description="是否超级弹幕")
    price: float = Field(default=0.0, description="价格（如果有）")

    def to_natural_language(self) -> str:
        prefix = "【超级弹幕】" if self.is_super_chat else ""
        return f"{prefix}{self.user_name} 说：{self.content}"


class GiftEvent(LiveEventModel):
    """礼物事件"""
    event_type: ClassVar[str] = LiveEventType.GIFT.value

    # 礼物特有字段
    gift_id: str = Field(default="", description="礼物ID")
    gift_name: str = Field(default="", description="礼物名称")
    gift_count: int = Field(default=1, description="礼物数量")
    gift_value: int = Field(default=0, description="礼物价值（平台货币）")
    combo_count: int = Field(default=1, description="连击次数")
    is_super_gift: bool = Field(default=False, description="是否超级礼物")

    def to_natural_language(self) -> str:
        prefix = "【超级礼物】" if self.is_super_gift else ""
        return f"{prefix}{self.user_name} 送出了 {self.gift_name} x{self.gift_count}"


class EnterEvent(LiveEventModel):
    """进入直播间事件"""
    event_type: ClassVar[str] = LiveEventType.ENTER.value

    # 进入特有字段
    user_level: int = Field(default=0, description="用户等级")
    is_vip: bool = Field(default=False, description="是否VIP")
    is_manager: bool = Field(default=False, description="是否房管")

    def to_natural_language(self) -> str:
        prefix = ""
        if self.is_manager:
            prefix = "【房管】"
        elif self.is_vip:
            prefix = "【VIP】"
        return f"{prefix}{self.user_name} 进入了直播间"


class FollowEvent(LiveEventModel):
    """关注事件"""
    event_type: ClassVar[str] = LiveEventType.FOLLOW.value

    # 关注特有字段
    follow_count: int = Field(default=1, description="关注次数")

    def to_natural_language(self) -> str:
        return f"{self.user_name} 关注了直播间"


class LikeEvent(LiveEventModel):
    """点赞事件"""
    event_type: ClassVar[str] = LiveEventType.LIKE.value

    # 点赞特有字段
    like_count: int = Field(default=1, description="点赞数量")

    def to_natural_language(self) -> str:
        return f"{self.user_name} 点赞了 {self.like_count} 次"


class ShareEvent(LiveEventModel):
    """分享事件"""
    event_type: ClassVar[str] = LiveEventType.SHARE.value

    # 分享特有字段
    share_platform: str = Field(default="", description="分享到的平台")
    share_type: str = Field(default="", description="分享类型")

    def to_natural_language(self) -> str:
        return f"{self.user_name} 分享了直播间到{self.share_platform}"


class LiveStartEvent(LiveEventModel):
    """直播开始事件"""
    event_type: ClassVar[str] = LiveEventType.LIVE_START.value

    def to_natural_language(self) -> str:
        return "直播开始了！"


class LiveEndEvent(LiveEventModel):
    """直播结束事件"""
    event_type: ClassVar[str] = LiveEventType.LIVE_END.value

    def to_natural_language(self) -> str:
        return "直播结束了"


class RoomInfoUpdateEvent(LiveEventModel):
    """房间信息更新事件"""
    event_type: ClassVar[str] = LiveEventType.ROOM_INFO_UPDATE.value

    # 房间信息字段
    current_users: int = Field(default=0, description="当前在线人数")
    total_users: int = Field(default=0, description="累计观看人数")
    room_title: str = Field(default="", description="房间标题")
    room_status: str = Field(default="", description="房间状态")

    def to_natural_language(self) -> str:
        return f"房间信息更新：当前{self.current_users}人在线"


# ========== 事件工厂 ==========

class EventFactory:
    """事件工厂，负责创建和管理事件类型"""

    def __init__(self):
        self._event_types: Dict[str, Type[LiveEventModel]] = {}
        self._register_base_event_types()

    def _register_base_event_types(self) -> None:
        """注册基础事件类型"""
        self.register_event_type(ChatEvent)
        self.register_event_type(GiftEvent)
        self.register_event_type(EnterEvent)
        self.register_event_type(FollowEvent)
        self.register_event_type(LikeEvent)
        self.register_event_type(ShareEvent)
        self.register_event_type(LiveStartEvent)
        self.register_event_type(LiveEndEvent)
        self.register_event_type(RoomInfoUpdateEvent)

    def register_event_type(self, event_class: Type[LiveEventModel]) -> None:
        """注册事件类型"""
        self._event_types[event_class.event_type] = event_class

    def create_event(self, event_type: str, **kwargs) -> Optional[LiveEventModel]:
        """创建事件实例"""
        if event_type not in self._event_types:
            return None

        event_class = self._event_types[event_type]
        return event_class(**kwargs)

    def from_live_event(self, live_event: LiveEvent) -> Optional[LiveEventModel]:
        """从LiveEvent字典创建事件实例"""
        event_type = live_event["event_type"]
        if event_type not in self._event_types:
            return None

        event_class = self._event_types[event_type]
        return event_class.from_live_event(live_event)

    def get_event_class(self, event_type: str) -> Optional[Type[LiveEventModel]]:
        """获取事件类"""
        return self._event_types.get(event_type)

    def get_all_event_types(self) -> List[str]:
        """获取所有已注册的事件类型"""
        return list(self._event_types.keys())


# ========== 事件转换器 ==========

class EventTransformer(ABC):
    """事件转换器抽象，将平台原始数据转换为通用事件"""

    def __init__(self, platform: str):
        self.platform = platform

    @abstractmethod
    def transform(self, raw_data: Dict[str, Any]) -> Optional[LiveEvent]:
        """
        转换平台原始数据为LiveEvent
        返回None表示无法转换或应该过滤
        """
        pass

    @abstractmethod
    def get_supported_event_types(self) -> List[str]:
        """获取支持的事件类型"""
        pass

