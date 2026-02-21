# import wave  # 用于保存音频文件
import logging
from collections import deque
from typing import Optional, Union

import numpy as np
import pyaudio
import scipy.signal as signal
from ghoshell_common.contracts import LoggerItf
from numpy.typing import NDArray

from listener.concepts.listener import AudioInput

__all__ = ['PyAudioInput']


def from_np_type_to_pyaudio_type(np_type: np.dtype) -> int:
    if np_type is np.int16:
        return pyaudio.paInt16
    elif np_type is np.int32:
        return pyaudio.paInt32
    else:
        # todo: 支持各种格式.
        raise NotImplementedError(f'np_type {np_type} is not supported yet')


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


class PyAudioInput(AudioInput):

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
            read_interval: float = 0.128,  # 修改为默认128ms的块大小
            chunk_size: int = 2048,  # 修改为2048，在16kHz时能提供更好的块大小
            # save_path: str = "./audio_data",      # 音频文件保存路径
    ) -> None:
        """
        初始化PyAudio输入流
        :param rate: 采样率
        :param channels: 通道
        :param device_index: 设备 id.
        :param logger: 日志
        :param dtype: 音频的位深.
        :param chunk_size: 拉取数据的 frames_per_buffer
        """
        # 初始化基本参数
        self.pa = pa
        self.input_id: str = input_id
        self.rate: int = rate
        self.channels: int = channels
        self.dtype = dtype
        self._pyaudio_format: int = from_np_type_to_pyaudio_type(dtype)
        self._device_index: Optional[int] = device_index
        self._read_interval: float = read_interval
        self._chunk_size: int = chunk_size
        self.logger = logger if logger is not None else logging.getLogger("PyAudioInput")
        # self.save_path = save_path
        params = dict(
            format=self._pyaudio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self._device_index,
            stream_callback=None
        )
        self._started = False
        self.logger.info("PyAudio initialized stream with params: %s", params)
        try:
            self._stream = self.pa.open(**params)
        except OSError as e:
            raise ValueError(f'fail to create input stream of params: %r', params) from e
        self._closed: bool = False
        
        # 音频保存相关初始化
        # self._chunk_counter = 0  # 用于文件命名
        # self._raw_wav_file = None
        # self._denoised_wav_file = None
        # self._init_audio_files()

    # def _init_audio_files(self) -> None:
    #     """初始化音频保存文件"""
    #     try:
    #         import os
    #         import time
    #         
    #         # 确保保存目录存在
    #         os.makedirs(self.save_path, exist_ok=True)
    #         
    #         # 创建带时间戳的文件名
    #         timestamp = int(time.time())
    #         raw_filename = f"raw_audio_{timestamp}.wav"
    #         denoised_filename = f"denoised_audio_{timestamp}.wav"
    #         
    #         raw_path = os.path.join(self.save_path, raw_filename)
    #         denoised_path = os.path.join(self.save_path, denoised_filename)
    #         
    #         # 初始化原始音频文件
    #         self._raw_wav_file = wave.open(raw_path, 'wb')
    #         self._raw_wav_file.setnchannels(self.channels)
    #         self._raw_wav_file.setsampwidth(np.dtype(self.dtype).itemsize)
    #         self._raw_wav_file.setframerate(self.rate)
    #         
    #         # 初始化降噪后音频文件
    #         self._denoised_wav_file = wave.open(denoised_path, 'wb')
    #         self._denoised_wav_file.setnchannels(self.channels)
    #         self._denoised_wav_file.setsampwidth(np.dtype(self.dtype).itemsize)
    #         self._denoised_wav_file.setframerate(self.rate)
    #         
    #         self.logger.info(f"音频保存文件已初始化: {raw_path}, {denoised_path}")
    #         
    #     except Exception as e:
    #         self.logger.exception(f"初始化音频文件失败: {e}")
    #         raise e

    def start(self) -> None:
        """
        启动录音状态?
        可以不断用 read 接口拉取数据.
        """
        if self._closed:
            raise OSError('PyAudio already closed')
        self._started = True
        self._stream.start_stream()
        self.logger.info("start audio input")

    def stop(self) -> None:
        if self._closed:
            return
        self._stream.stop_stream()

    def stopped(self) -> bool:
        return self._closed or self._stream.is_stopped()

    def closed(self) -> bool:
        return self._closed

    def _return_zero(self, duration: float) -> np.ndarray:
        return np.zeros(int(self.rate * duration), dtype=self.dtype)

    def read(self, *, rate: Optional[int] = None, duration: Optional[float] = None) -> np.ndarray:
        """
        直接读取固定大小的音频数据块（约128ms）
        """
        if not self._started:
            raise RuntimeError(f"PyAudioInput is not running")

        try:
            read_interval = duration
            if read_interval is None:
                read_interval = self._read_interval
            # 计算帧数，约为128ms的数据
            frames_to_read = int(self.rate * read_interval)

            # 直接读取数据
            data = self._stream.read(frames_to_read, exception_on_overflow=False)

            # 转换为numpy数组
            np_data = np.frombuffer(data, dtype=self.dtype)

            # 保存原始音频数据
            # if self._raw_wav_file is not None:
            #     try:
            #         self._raw_wav_file.writeframes(np_data.tobytes())
            #     except Exception as e:
            #         self.logger.exception(f"保存原始音频失败: {e}")

            # 调用降噪函数，使用实时优化参数
            denoised_data = mse_denoise_advanced(
                np_data, 
                sample_rate=self.rate,
                neighbor_check_range=2,  # 实时模式：仅2帧延时约23ms
                min_voice_length_ms=30   # 减少语音保护长度，提高响应速度
            )

            # 保存降噪后音频数据
            # if self._denoised_wav_file is not None:
            #     try:
            #         self._denoised_wav_file.writeframes(denoised_data.astype(self.dtype).tobytes())
            #     except Exception as e:
            #         self.logger.exception(f"保存降噪音频失败: {e}")

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
        self._stream.close()
        
        # 关闭音频保存文件
        # try:
        #     if self._raw_wav_file is not None:
        #         self._raw_wav_file.close()
        #         self._raw_wav_file = None
        #         self.logger.info("原始音频文件已关闭")
        #     
        #     if self._denoised_wav_file is not None:
        #         self._denoised_wav_file.close()
        #         self._denoised_wav_file = None
        #         self.logger.info("降噪音频文件已关闭")
        # except Exception as e:
        #     self.logger.exception(f"关闭音频文件失败: {e}")
