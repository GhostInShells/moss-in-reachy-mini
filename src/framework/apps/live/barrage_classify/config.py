# config.py
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class BarrageType(Enum):
    """弹幕类型枚举"""
    QUESTION = "question"  # 提问
    COMMAND = "command"  # 指令/请求
    BUSINESS = "business"  # 业务/产品相关
    PRAISE = "praise"  # 赞美/评价
    GREET = "greet"  # 问候/社交
    OTHER = "other"  # 其他


class Priority(Enum):
    """优先级枚举"""
    P0 = 0  # 最高优先级 - 立即处理
    P1 = 1  # 高优先级 - 短时窗内处理
    P2 = 2  # 中优先级 - 可聚合处理
    P3 = 3  # 低优先级 - 可忽略或批量回应


@dataclass
class KeywordGroupConfig:
    """关键词组配置"""
    keywords: List[str]  # 关键词列表
    weight: float = 1.0  # 权重（用于计算匹配强度）
    require_exact: bool = False  # 是否需要精确匹配


@dataclass
class BarrageClassifierConfig:
    """弹幕分类器完整配置"""
    # 类型关键词映射
    type_keywords: Dict[BarrageType, KeywordGroupConfig]

    # 优先级关键词映射（覆盖类型优先级）
    priority_keywords: Dict[Priority, KeywordGroupConfig]

    # 停用词（包含这些词的弹幕直接降为P3）
    stop_words: List[str]

    # 简单问候词（用于快速识别）
    simple_greetings: List[str]

    # 默认优先级映射
    default_priority_map: Dict[BarrageType, Priority]

    # 重复检测时间窗口（秒）
    repeat_detection_window: int = 30

    # 历史记录清理时间（秒）
    history_cleanup_interval: int = 60