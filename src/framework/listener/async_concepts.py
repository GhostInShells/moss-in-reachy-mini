"""
异步 Listener 模块的核心概念和协议定义。

完全基于 asyncio 设计，弃用线程模型。
"""

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union, Callable, Protocol, runtime_checkable

import numpy as np
from ghoshell_common.helpers import uuid

# 重用现有的 Recognition 类
from .concepts import Recognition


@runtime_checkable
class AsyncAudioInput(Protocol):
    """
    异步音频输入协议。
    从音频输入设备（如麦克风）异步获取音频数据。
    """

    input_id: str
    rate: int
    channels: int
    dtype: np.dtype

    @abstractmethod
    async def read(self, *, rate: Optional[int] = None, duration: Optional[float] = None) -> np.ndarray:
        """
        异步获取一段音频数据。

        Args:
            rate: 目标采样率，如果为None则使用原始采样率
            duration: 音频时长（秒），如果为None则使用默认块大小

        Returns:
            音频数据数组
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """启动音频输入"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止音频输入"""
        pass

    @abstractmethod
    async def close(self, error: Optional[Exception] = None) -> None:
        """关闭音频输入，释放资源"""
        pass

    @abstractmethod
    async def closed(self) -> bool:
        """检查是否已关闭"""
        pass

    async def __aenter__(self) -> "AsyncAudioInput":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close(exc_val)


# Recognition 类从 concepts 模块导入


class AsyncRecognitionCallback(Protocol):
    """
    异步识别回调接口。
    """

    @abstractmethod
    async def on_recognition(self, result: Recognition) -> None:
        """收到识别结果时的回调"""
        pass

    @abstractmethod
    async def on_error(self, error: str) -> None:
        """收到错误时的回调"""
        pass

    async def save_batch(self, rec: Recognition, audio: np.ndarray) -> None:
        """完成时回调保存批次（可选实现）"""
        pass


@runtime_checkable
class AsyncRecognitionBatch(Protocol):
    """
    异步语音识别批次。
    代表一次完整的识别流程。
    """
    batch_id: str

    @abstractmethod
    async def start(self) -> None:
        """启动识别批次"""
        pass

    @abstractmethod
    async def close(self, error: Optional[Exception] = None) -> None:
        """关闭识别批次"""
        pass

    @abstractmethod
    async def buffer(self, audio: np.ndarray) -> None:
        """异步追加音频数据"""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """提交批次，表示音频输入结束"""
        pass

    @abstractmethod
    async def get_last_recognition(self) -> Optional[Recognition]:
        """获取最后一次识别结果"""
        pass

    @abstractmethod
    async def get_buffer(self) -> np.ndarray:
        """获取当前批次的完整音频缓冲"""
        pass

    @abstractmethod
    async def is_done(self) -> bool:
        """检查批次是否已完成"""
        pass

    async def wait_until_done(self, timeout: Optional[float] = None) -> None:
        """等待批次完成"""
        pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            await self.close(exc_val)
        else:
            await self.commit()
            await self.wait_until_done()


@runtime_checkable
class AsyncRecognizer(Protocol):
    """
    异步语音识别引擎。
    """
    frame_duration: float  # 每帧的采样时间（秒）
    sample_rate: int  # 采样率

    @abstractmethod
    async def start(self) -> None:
        """启动识别引擎"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭识别引擎"""
        pass

    @abstractmethod
    async def is_closed(self) -> bool:
        """检查引擎是否已关闭"""
        pass

    @abstractmethod
    async def new_batch(
        self,
        callback: AsyncRecognitionCallback,
        batch_id: str = "",
        vad: Optional[int] = None,
        stop_on_sentence: bool = False,
    ) -> AsyncRecognitionBatch:
        """
        创建新的识别批次。

        Args:
            callback: 识别回调
            batch_id: 批次ID
            vad: VAD时间（毫秒）
            stop_on_sentence: 是否在拿到分句时结束批次
        """
        pass


class AsyncListenerCallback(AsyncRecognitionCallback, Protocol):
    """
    异步Listener回调接口。
    """

    @abstractmethod
    async def on_state_change(self, state: str) -> None:
        """状态变化回调"""
        pass

    async def on_waken(self) -> None:
        """唤醒回调（可选，唤醒词功能跳过）"""
        pass


class AsyncListenerStateName(str, Enum):
    """异步Listener状态枚举"""
    LISTENING = "listening"
    DEAF = "deaf"
    ASLEEP = "asleep"  # 注意：唤醒词功能跳过，此状态可能退化为DEAF
    PDT_LISTENING = "pdt_listening"
    PDT_WAITING = "pdt_waiting"


@runtime_checkable
class AsyncListenerState(Protocol):
    """
    异步Listener状态协议。
    """

    @abstractmethod
    def name(self) -> AsyncListenerStateName:
        """获取状态名称"""
        pass

    @abstractmethod
    async def start(self) -> None:
        """启动状态"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭状态"""
        pass

    @abstractmethod
    async def clear_buffer(self) -> None:
        """清空缓冲"""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """提交状态（如PTT模式中的松手操作）"""
        pass

    @abstractmethod
    async def set_vad(self, vad_time: int) -> None:
        """设置VAD时间"""
        pass

    @abstractmethod
    async def next_state(self) -> Optional[tuple[str, Optional[np.ndarray]]]:
        """
        获取下一个状态。

        Returns:
            (state_name, audio_buffer) 或 None
        """
        pass


@runtime_checkable
class AsyncListenerService(Protocol):
    """
    异步音频输入服务。
    集成音频采集、事件检测和语音识别功能。
    """

    @abstractmethod
    async def bootstrap(self) -> None:
        """启动服务"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """关闭服务"""
        pass

    @abstractmethod
    async def audio_input(self) -> AsyncAudioInput:
        """获取音频输入"""
        pass

    @abstractmethod
    async def recognizer(self) -> AsyncRecognizer:
        """获取识别引擎"""
        pass

    @abstractmethod
    async def set_callback(self, callback: AsyncListenerCallback) -> None:
        """设置回调"""
        pass

    @abstractmethod
    async def clear_buffer(self) -> None:
        """清空所有缓冲"""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """主动提交会话，获取最终结果"""
        pass

    @abstractmethod
    async def set_vad(self, vad_time: int) -> None:
        """设置VAD时间"""
        pass

    @abstractmethod
    async def all_states(self) -> list[str]:
        """获取所有支持的状态"""
        pass

    @abstractmethod
    async def set_state(self, state: str) -> None:
        """设置当前状态"""
        pass

    @abstractmethod
    async def current_state(self) -> AsyncListenerState:
        """获取当前状态"""
        pass


# 类型别名，便于迁移期间使用
AudioInput = AsyncAudioInput
RecognitionCallback = AsyncRecognitionCallback
RecognitionBatch = AsyncRecognitionBatch
Recognizer = AsyncRecognizer
ListenerCallback = AsyncListenerCallback
ListenerState = AsyncListenerState
ListenerService = AsyncListenerService