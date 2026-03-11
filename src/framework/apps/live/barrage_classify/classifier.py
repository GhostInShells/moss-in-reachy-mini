# classifier.py
import re
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from framework.apps.live.barrage_classify.config import BarrageClassifierConfig, BarrageType, Priority
from framework.apps.live.barrage_classify.default_config import DEFAULT_CONFIG


class BarrageClassifier:
    """配置驱动的弹幕分类器"""

    def __init__(self, config: BarrageClassifierConfig=DEFAULT_CONFIG):
        """
        初始化分类器

        Args:
            config: 分类器配置对象
        """
        self.config = config
        self._build_keyword_patterns()

        # 用户发言历史（用于检测重复）
        self.user_history: Dict[str, Tuple[float, str]] = {}
        self.last_cleanup_time = time.time()

    def _build_keyword_patterns(self):
        """构建关键词正则表达式模式（用于精确匹配）"""
        self.patterns = {}

        # 构建类型关键词模式
        for bar_type, group_config in self.config.type_keywords.items():
            if group_config.require_exact:
                # 将关键词转换为正则表达式，匹配单词边界
                patterns = [rf'\b{re.escape(kw)}\b' for kw in group_config.keywords]
                self.patterns[bar_type] = re.compile('|'.join(patterns))

        # 构建优先级关键词模式
        for priority, group_config in self.config.priority_keywords.items():
            if group_config.require_exact:
                patterns = [rf'\b{re.escape(kw)}\b' for kw in group_config.keywords]
                self.patterns[priority] = re.compile('|'.join(patterns))

    def classify(self, text: str, user_id: Optional[str] = None) -> Tuple[BarrageType, Priority]:
        """
        分类弹幕

        Args:
            text: 弹幕文本
            user_id: 用户ID（可选，用于重复检测）

        Returns:
            (弹幕类型, 优先级)
        """
        # 清理文本
        cleaned_text = self._clean_text(text)

        # 检查停用词
        if self._contains_stop_words(cleaned_text):
            return BarrageType.OTHER, Priority.P3

        # 1. 检查是否为P0优先级（最高优先级）
        if self._is_priority(cleaned_text, Priority.P0):
            # 确定类型
            bar_type = self._get_barrage_type(cleaned_text)
            return bar_type, Priority.P0

        # 2. 检查是否为P1优先级（高优先级）
        if self._is_priority(cleaned_text, Priority.P1):
            bar_type = self._get_barrage_type(cleaned_text)
            return bar_type, Priority.P1

        # 3. 检查重复发言（可视为P1优先级）
        if user_id and self._is_repeat_message(cleaned_text, user_id):
            bar_type = self._get_barrage_type(cleaned_text)
            return bar_type, Priority.P1

        # 4. 检查是否为简单问候（P3优先级）
        if self._is_simple_greeting(cleaned_text):
            return BarrageType.GREET, Priority.P3

        # 5. 普通分类
        bar_type = self._get_barrage_type(cleaned_text)
        priority = self.config.default_priority_map.get(bar_type, Priority.P3)

        return bar_type, priority

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 转换为小写（中文不受影响，英文会变小写）
        text = text.lower()
        # 移除首尾空白
        text = text.strip()
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        return text

    def _contains_stop_words(self, text: str) -> bool:
        """检查是否包含停用词"""
        for stop_word in self.config.stop_words:
            if stop_word in text:
                return True
        return False

    def _is_priority(self, text: str, priority: Priority) -> bool:
        """检查是否匹配指定优先级的关键词"""
        if priority not in self.config.priority_keywords:
            return False

        group_config = self.config.priority_keywords[priority]

        if group_config.require_exact and priority in self.patterns:
            # 使用正则表达式精确匹配
            return bool(self.patterns[priority].search(text))
        else:
            # 使用包含匹配
            for keyword in group_config.keywords:
                if keyword in text:
                    return True
        return False

    def _is_simple_greeting(self, text: str) -> bool:
        """检查是否为简单问候"""
        for greeting in self.config.simple_greetings:
            if greeting in text:
                return True
        return False

    def _get_barrage_type(self, text: str) -> BarrageType:
        """获取弹幕类型"""
        # 计算每个类型的匹配分数
        scores = {}

        for bar_type, group_config in self.config.type_keywords.items():
            score = 0.0

            if group_config.require_exact and bar_type in self.patterns:
                # 精确匹配，计算匹配次数
                matches = self.patterns[bar_type].findall(text)
                score = len(matches) * group_config.weight
            else:
                # 包含匹配，计算匹配关键词数量
                for keyword in group_config.keywords:
                    if keyword in text:
                        score += group_config.weight

            if score > 0:
                scores[bar_type] = score

        # 如果没有匹配到任何类型，返回OTHER
        if not scores:
            return BarrageType.OTHER

        # 返回分数最高的类型
        return max(scores.items(), key=lambda x: x[1])[0]

    def _is_repeat_message(self, text: str, user_id: str) -> bool:
        """检查是否为重复发言"""
        current_time = time.time()

        # 定期清理历史记录
        if current_time - self.last_cleanup_time > self.config.history_cleanup_interval:
            self._clean_history(current_time)
            self.last_cleanup_time = current_time

        # 检查用户历史
        if user_id in self.user_history:
            last_time, last_text = self.user_history[user_id]

            # 在时间窗口内发送相同内容
            if current_time - last_time < self.config.repeat_detection_window and text == last_text:
                return True

        # 更新历史
        self.user_history[user_id] = (current_time, text)
        return False

    def _clean_history(self, current_time: float):
        """清理过期的用户历史"""
        expired_users = [
            user_id for user_id, (last_time, _) in self.user_history.items()
            if current_time - last_time > self.config.repeat_detection_window * 2
        ]

        for user_id in expired_users:
            del self.user_history[user_id]

    def update_config(self, config: BarrageClassifierConfig):
        """更新配置（热更新）"""
        self.config = config
        self._build_keyword_patterns()