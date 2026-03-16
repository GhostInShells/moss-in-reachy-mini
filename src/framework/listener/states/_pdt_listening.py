from typing import Callable, Optional

from framework.listener.concepts import (
    ListenerStateName,
    Recognizer,
    ListenerCallback,
    AudioInput,
    Recognition,
)
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid
from ._listening import ListeningState
import numpy as np

VAD = Callable[[np.ndarray, Optional[int]], bool]


class PdtListeningState(ListeningState):
    """
    Push to Talk 听取模式
    在此模式下，VAD时间被设置为很长的时间，防止ASR引擎自动断句
    完全依赖用户的松手操作来触发completion
    :param recognizer: asr
    :param audio_input: 音频输入
    :param callback: 事件回调.
    :param logger: 日志
    """

    def __init__(
            self, *,
            recognizer: Recognizer,
            audio_input: AudioInput,
            callback: ListenerCallback,
            logger: LoggerItf,
    ):
        super().__init__(
            recognizer=recognizer,
            audio_input=audio_input,
            callback=callback,
            logger=logger,
            on_complete_state=ListenerStateName.pdt_waiting.value,
            stop_on_sentence=False,
            # 只运行一个 batch 就退出循环.
            allow_batch=1,
        )
        self._batch_id = uuid()
        # 幂等开关. pdt 只运行一次 commit.
        self._committed = False
        self._seq: int = 0
        self._logger.info(f"PTT mode initialized with VAD time: {self._vad_time} ms")

    def name(self) -> ListenerStateName:
        return ListenerStateName.pdt_listening

    def on_recognition(self, result: Recognition) -> None:
        result.batch_id = self._batch_id
        self._seq += 1
        result.seq = self._seq
        super().on_recognition(result)

    def commit(self) -> None:
        """PTT模式：松手时立即结束当前批次并发送识别结果"""
        if self._committed:
            return
        # 幂等操作开关.
        self._committed = True
        # 提交时还没有任何输入数据.
        if self._callback_recognition is None or self._callback_recognition.is_last:
            # 直接关闭主循环.
            self._closed_event.set()
            return
        super().commit()

    def _main_loop(self) -> None:
        self._callback.on_recognition(Recognition(
            batch_id=self._batch_id,
            text="",
        ))
        super()._main_loop()
