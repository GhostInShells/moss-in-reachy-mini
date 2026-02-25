import logging
from collections import deque
from typing import Optional, Union

import numpy as np
import scipy.signal as signal
from ghoshell_common.contracts import LoggerItf
from numpy.typing import NDArray
from reachy_mini import ReachyMini

from moss_in_reachy_mini.listener.concepts.listener import AudioInput


def mse_denoise_advanced(signal: np.ndarray,
                         window_size: int = 2048,
                         hop_size: int = 512,
                         fixed_noise_mse: float = 0.01,
                         attenuation_factor: float = 0.3,
                         min_voice_length_ms: int = 50,
                         neighbor_check_range: int = 3,  # 实时优化：3帧约35ms延时，平衡效果和实时性
                         sample_rate: int = 44100) -> np.ndarray:
    # 数据类型处理：如果是int16，先归一化到float32范围
    original_dtype = signal.dtype
    if original_dtype == np.int16:
        # 归一化到 [-1, 1] 范围，和test.py的float32保持一致
        signal_normalized = signal.astype(np.float32) / 32768.0
    else:
        signal_normalized = signal.astype(np.float32)

    # 检查输入数据长度是否足够
    if len(signal_normalized) < window_size:
        # 数据长度不足，直接返回原始信号
        if original_dtype == np.int16:
            return (signal_normalized * 32768.0).astype(np.int16)
        else:
            return signal_normalized.astype(original_dtype)

    denoised = np.copy(signal_normalized)
    n_frames = (len(signal_normalized) - window_size) // hop_size + 1
    mse_history = deque(maxlen=neighbor_check_range * 2 + 1)
    voice_region_mask = np.zeros(n_frames, dtype=bool)

    # 第一遍：计算所有窗口的MSE并标记疑似语音段
    for i in range(n_frames):
        start = i * hop_size
        window = signal_normalized[start:start + window_size]
        mse = np.mean((window - np.mean(window)) ** 2)
        mse_history.append(mse)

        # 相邻帧联合判断
        if len(mse_history) == neighbor_check_range * 2 + 1:
            avg_mse = np.mean(mse_history)
            if avg_mse > fixed_noise_mse * 1.5:
                voice_region_mask[i] = True

    # 第二遍：应用降噪
    for i in range(n_frames):
        start = i * hop_size
        end = start + window_size

        # 检查当前帧是否在语音保护区域内
        is_protected = False
        protection_range = int(min_voice_length_ms * sample_rate / 1000 / hop_size)
        for j in range(max(0, i - protection_range), min(n_frames, i + protection_range + 1)):
            if voice_region_mask[j]:
                is_protected = True
                break

        # 仅对非保护区域且低MSE的帧降噪
        if not is_protected:
            window = signal_normalized[start:end]
            mse = np.mean((window - np.mean(window)) ** 2)
            if mse < fixed_noise_mse:
                fade_window = np.linspace(1, attenuation_factor, window_size)
                denoised[start:end] *= fade_window

    # 转回原始数据类型
    if original_dtype == np.int16:
        # 从 [-1, 1] 范围转回 int16
        denoised = (denoised * 32768.0).astype(np.int16)
    else:
        denoised = denoised.astype(original_dtype)

    return denoised

class ReachyMiniInput(AudioInput):

    def __init__(
            self,
            mini: ReachyMini,
            *,
            input_id: str = "",
            rate: int = 16000,
            channels: int = 2,
            dtype: np.dtype = np.int16,
            logger: LoggerItf = None,
    ) -> None:
        # 初始化基本参数
        self.mini = mini
        self.input_id: str = input_id
        self.rate: int = rate
        self.channels: int = channels
        self.logger = logger if logger is not None else logging.getLogger("PyAudioInput")
        self.dtype = dtype

        self._started = False
        self._closed: bool = False

    def start(self) -> None:
        """
        启动录音状态?
        可以不断用 read 接口拉取数据.
        """
        if self._closed:
            raise OSError('PyAudio already closed')
        self._started = True
        self.mini.media.start_recording()
        self.logger.info("start audio input")

    def stop(self) -> None:
        if self._closed:
            return
        self.mini.media.stop_recording()

    def stopped(self) -> bool:
        return self._closed

    def closed(self) -> bool:
        return self._closed

    def _return_zero(self, duration: float) -> np.ndarray:
        return np.zeros(int(self.rate * duration), dtype=self.dtype)

    def read(self, *, rate: Optional[int] = None, duration: Optional[float] = None) -> np.ndarray:
        if not self._started:
            raise RuntimeError(f"PyAudioInput is not running")

        try:
            # 直接读取数据
            data = self.mini.media.get_audio_sample()
            if data is None:
                return self._return_zero(duration)

            # 转换为numpy数组
            np_data = np.frombuffer(data, dtype=self.dtype)

            # 调用降噪函数，使用实时优化参数
            denoised_data = mse_denoise_advanced(
                np_data,
                sample_rate=self.rate,
                neighbor_check_range=2,  # 实时模式：仅2帧延时约23ms
                min_voice_length_ms=30  # 减少语音保护长度，提高响应速度
            )

            # 如果需要，重采样
            return self._resample(denoised_data, rate)
        except Exception as e:
            # 出错了的话不中断，而是继续运行
            self.logger.exception(e)
            raise e

    def _resample(self, audio_data: NDArray, rate: Union[int, None] = None) -> NDArray:
        """
        使用 scipy.signal.resample 进行采样率转换
        Args:
            audio_data: 原始音频数据
        Returns:
            np.ndarray: 重采样后的音频数据
        """
        if rate is None or rate == self.rate:
            return audio_data

        number_of_samples = int(len(audio_data) * float(rate) / self.rate)
        resampled_audio_data = signal.resample(audio_data, number_of_samples)
        return resampled_audio_data.astype(self.dtype)

    def close(self, error: Optional[Exception] = None) -> None:
        """
        关闭音频流并释放资源
        """
        if self._closed:
            return
        self._closed = True
        if error is not None:
            self.logger.exception(error)
        self.stop()