import time

import numpy as np
from ghoshell_common.contracts import LoggerItf
from ghoshell_moss import AudioFormat
from ghoshell_moss.speech.player.base_player import BaseAudioStreamPlayer
from reachy_mini import ReachyMini

from moss_in_reachy_mini.video.recorder_worker import VideoRecorderWorker


class ReachyMiniStreamPlayer(BaseAudioStreamPlayer):
    def __init__(
        self,
        mini: ReachyMini,
        *,
        logger: LoggerItf | None = None,
        safety_delay: float = 0.5,
        recorder: VideoRecorderWorker | None = None,
    ):
        """
        基于 PyAudio 的异步音频播放器实现
        使用单独的线程处理阻塞的音频输出操作
        """
        self.mini = mini
        self._recorder = recorder
        self._logger = logger
        self._estimated_end_time = 0.0
        super().__init__(
            sample_rate=self.mini.media.get_output_audio_samplerate(),
            channels=self.mini.media.get_output_channels(),
            logger=logger,
            safety_delay=safety_delay,
        )

    def add(
        self,
        chunk: np.ndarray,
        *,
        audio_type: AudioFormat,
        rate: int,
        channels: int = 1,
    ) -> float:
        """添加音频片段到播放队列"""
        if self._closed:
            return time.time()

        # 格式转换
        if audio_type == AudioFormat.PCM_F32LE:
            # float32 [-1, 1] -> int16
            audio_data = (chunk * 32767).astype(np.int16)
        else:
            # 假设已经是 int16
            audio_data = chunk.astype(np.int16)

        # 2. 核心转换：int16 → float32（归一化到音频标准范围[-1.0, 1.0]）
        audio_f32 = audio_data.astype(np.float32) / 32768.0

        # 3. 单声道转双通道（复制左声道数据到右声道，硬件要求双通道）
        audio_data = np.column_stack((audio_f32, audio_f32))

        # 计算持续时间
        duration = audio_data.shape[0] / self.sample_rate

        resampled_audio_data = self.resample(audio_data, origin_rate=rate, target_rate=self.sample_rate)

        # 添加到线程安全队列
        self._audio_queue.put_nowait(resampled_audio_data)
        self._play_done_event.clear()

        # 更新预计结束时间
        current_time = time.time()
        if current_time > self._estimated_end_time:
            self._estimated_end_time = current_time + duration
        else:
            self._estimated_end_time += duration
        return self._estimated_end_time

    def _audio_stream_start(self):
        self.mini.media.start_playing()

    def _audio_stream_stop(self):
        self.mini.media.stop_playing()

    def _audio_stream_write(self, data: np.ndarray):
        # 1. 校验输入数据的合法性（避免格式错误）
        # if data.dtype != np.int16:
        #     raise ValueError(f"输入数据类型错误，需要int16，实际是{data.dtype}")
        # if len(data.shape) != 1:
        #     raise ValueError(f"输入数据维度错误，需要一维数组，实际是{data.shape}")
        # Tap output audio for recording (int16 PCM interleaved)
        if self._recorder is not None:
            try:
                out_channels = int(self.mini.media.get_output_channels())
                if out_channels <= 0:
                    out_channels = 2

                arr = np.asarray(data)
                if arr.ndim == 1:
                    if out_channels > 1:
                        arr = np.column_stack((arr,) * out_channels)
                elif arr.ndim == 2:
                    # Normalize to (frames, channels)
                    if arr.shape[1] > arr.shape[0]:
                        arr = arr.T

                    if arr.shape[1] < out_channels:
                        arr = np.column_stack((arr[:, 0],) * out_channels)
                    elif arr.shape[1] > out_channels:
                        arr = arr[:, :out_channels]
                else:
                    # Unsupported shape
                    arr = None

                if arr is not None:
                    if arr.dtype == np.int16:
                        audio_i16 = arr
                    else:
                        # Assume float PCM in [-1, 1]
                        audio_f32 = arr.astype(np.float32, copy=False)
                        audio_f32 = np.clip(audio_f32, -1.0, 1.0)
                        audio_i16 = (audio_f32 * 32767.0).astype(np.int16)

                    audio_i16 = np.ascontiguousarray(audio_i16.astype("<i2", copy=False))
                    self._recorder.push_output_audio(audio_i16.tobytes())
            except Exception:
                self._logger.warning("Failed to record output audio")

        self.mini.media.push_audio_sample(data)
