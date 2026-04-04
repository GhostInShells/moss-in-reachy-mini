"""
异步 PyAudio 音频输入实现。

基于现有 PyAudioInput，但完全异步化。
使用 asyncio.to_thread 执行阻塞的音频读取操作。
"""

import asyncio
import logging
from typing import Optional, Union
import numpy as np
import pyaudio
from ghoshell_common.contracts import LoggerItf

from .async_concepts import AsyncAudioInput
from .pyaudio_input_impl import (
    PyAudioInput as SyncPyAudioInput,
    from_np_type_to_pyaudio_type,
    mse_denoise_advanced,
)


class AsyncPyAudioInput(AsyncAudioInput):
    """
    异步 PyAudio 音频输入。

    包装同步的 PyAudioInput，通过 asyncio.to_thread 提供异步接口。
    """

    def __init__(
        self,
        pa: pyaudio.PyAudio,
        *,
        input_id: str = "",
        rate: int = 44100,
        channels: int = 1,
        device_index: Optional[int] = None,
        logger: Optional[LoggerItf] = None,
        dtype: np.dtype = np.int16,
        read_interval: float = 0.128,
        chunk_size: int = 2048,
    ):
        """
        初始化异步 PyAudio 输入。

        参数与同步版本相同。
        """
        self._sync_input = SyncPyAudioInput(
            pa=pa,
            input_id=input_id,
            rate=rate,
            channels=channels,
            device_index=device_index,
            logger=logger,
            dtype=dtype,
            read_interval=read_interval,
            chunk_size=chunk_size,
        )

        # 属性代理
        self.input_id = input_id
        self.rate = rate
        self.channels = channels
        self.dtype = dtype

        self._logger = logger or logging.getLogger("AsyncPyAudioInput")
        self._closed = False
        self._started = False

    async def read(
        self, *, rate: Optional[int] = None, duration: Optional[float] = None
    ) -> np.ndarray:
        """
        异步读取音频数据。

        Args:
            rate: 目标采样率
            duration: 音频时长（秒）

        Returns:
            音频数据数组
        """
        if not self._started:
            raise RuntimeError("AsyncPyAudioInput is not started")
        if self._closed:
            raise RuntimeError("AsyncPyAudioInput is closed")

        try:
            # 在单独的线程中执行阻塞的读取和降噪操作
            audio_data = await asyncio.to_thread(
                self._sync_input.read,
                rate=rate,
                duration=duration,
            )
            return audio_data
        except Exception as e:
            self._logger.exception(f"Error reading audio: {e}")
            raise

    async def start(self) -> None:
        """启动音频输入"""
        if self._closed:
            raise RuntimeError("AsyncPyAudioInput is closed")
        if self._started:
            return

        # 在单独的线程中启动同步输入
        await asyncio.to_thread(self._sync_input.start)
        self._started = True
        self._logger.info("AsyncPyAudioInput started")

    async def stop(self) -> None:
        """停止音频输入"""
        if not self._started or self._closed:
            return

        await asyncio.to_thread(self._sync_input.stop)
        self._started = False
        self._logger.info("AsyncPyAudioInput stopped")

    async def close(self, error: Optional[Exception] = None) -> None:
        """关闭音频输入，释放资源"""
        if self._closed:
            return

        self._closed = True
        self._started = False

        try:
            await asyncio.to_thread(self._sync_input.close, error)
        except Exception as e:
            self._logger.exception(f"Error closing audio input: {e}")

        self._logger.info("AsyncPyAudioInput closed")

    async def closed(self) -> bool:
        """检查是否已关闭"""
        return self._closed

    def __repr__(self) -> str:
        return (
            f"AsyncPyAudioInput(input_id={self.input_id}, "
            f"rate={self.rate}, channels={self.channels}, "
            f"started={self._started}, closed={self._closed})"
        )


class AsyncPyAudioInputConfig:
    """
    异步 PyAudio 输入配置。
    包装同步配置，提供异步工厂方法。
    """

    def __init__(self, sync_config):
        """
        初始化异步配置。

        Args:
            sync_config: 同步的 PyAudioInputConfig 实例
        """
        self._sync_config = sync_config

    async def new_audio_input(
        self,
        *,
        pa: Optional[pyaudio.PyAudio] = None,
        logger: Optional[LoggerItf] = None,
        dtype: np.dtype = np.int16,
    ) -> AsyncPyAudioInput:
        """
        异步创建新的音频输入实例。

        Args:
            pa: PyAudio 实例
            logger: 日志记录器
            dtype: 数据类型

        Returns:
            AsyncPyAudioInput 实例
        """
        # 在单独的线程中创建同步输入（可能涉及设备查询等阻塞操作）
        sync_input = await asyncio.to_thread(
            self._sync_config.new_audio_input,
            pa=pa,
            logger=logger,
            dtype=dtype,
        )

        # 包装为异步输入
        return AsyncPyAudioInput(
            pa=sync_input.pa,
            input_id=sync_input.input_id,
            rate=sync_input.rate,
            channels=sync_input.channels,
            device_index=sync_input._device_index,
            logger=sync_input.logger,
            dtype=sync_input.dtype,
            read_interval=sync_input._read_interval,
            chunk_size=sync_input._chunk_size,
        )