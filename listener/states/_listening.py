import time
from typing import Union, Callable, Optional

from listener.concepts.listener import (
    ListenerState, ListenerStateName,
    Recognizer, RecognitionBatch, Recognition,
    ListenerCallback,
    AudioInput, AudioInputLoop,
    RecognitionCallback
)
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import Timeleft, uuid
from threading import Thread, Event
from collections import deque
import numpy as np
import queue

VAD = Callable[[np.ndarray, Optional[int]], bool]


class ListeningState(ListenerState, RecognitionCallback):
    """
    认真的听.
    """

    def __init__(
            self, *,
            recognizer: Recognizer,
            audio_input: AudioInput,
            callback: ListenerCallback,
            logger: LoggerItf,
            vad: Optional[VAD] = None,

            # 内部配置项.
            allow_batch: int = 0,
            stop_on_sentence: bool = True,
            on_complete_state: Optional[str] = None,
            on_max_idle_state: Optional[str] = None,
            default_next_state: Optional[str] = None,
            max_idle_time: float = 10,
            max_retry_time: int = -1,
    ):
        """
        :param recognizer: asr
        :param audio_input: 音频输入
        :param callback: 事件回调.
        :param logger: 日志
        :param vad: 是否有 vad 检查的抽象.
        :param allow_batch: 运行运行多少个 asr batch, < 1 表示一直运行.
        :param stop_on_sentence: 是否在每个分句结束 batch 的运行.
        :param on_complete_state: vad 后是否要进入别的 state, 如果没有的话, 立刻开始下一轮 asr.
        :param on_max_idle_state: 如果超过了闲置时间, 是否要进入别的状态.
        :param default_next_state: 如果不设置的话, 默认用 on_complete_state 替代.
        :param max_idle_time: 如果超过了最大闲置 (没有识别到输入) 时间, 就会清空这一轮 asr.
        :param max_retry_time: 如果连接失败, 或 asr 异常, 能够容忍多少次.  <0 表示无限次.
        """
        self._recognizer = recognizer
        self._current_recognition_batch: Union[RecognitionBatch, None] = None
        self._audio_input = audio_input
        self._logger = logger
        self._audio_input_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._callback = callback
        self._closed_event = Event()
        self._vad_local_detector = vad
        self._allow_batch = allow_batch
        self._ran_batch = 0

        # 设置当前 state 结束时, 用哪个状态返回 .
        self._stop_on_sentence = stop_on_sentence
        self._on_complete_state = on_complete_state
        default_next_state = default_next_state or on_complete_state
        self._default_next_state = default_next_state

        self._clear_buffer_event = Event()
        self._started = False
        self._commit_event = Event()
        self._max_idle_time = max_idle_time
        self._on_max_idle_state = on_max_idle_state
        self._max_retry_time = max_retry_time
        self._fail_and_retry_time = 0
        self._main_loop_done_event = Event()
        self._wait_audio_interval = 0.05
        self._next_state: Union[ListeningState.NextState, None] = None
        self._vad_time: Optional[float] = None
        self._callback_recognition: Optional[Recognition] = None

    def on_recognition(self, result: Recognition) -> None:
        if self._closed_event.is_set() or self._main_loop_done_event.is_set():
            return
        if self._callback_recognition == result:
            return
        self._callback_recognition = result
        self._callback.on_recognition(result)

    def on_error(self, error: str) -> None:
        self._callback.on_error(error)

    def save_batch(self, rec: Recognition, audio: np.ndarray) -> None:
        self._callback.save_batch(rec, audio)

    def name(self) -> ListenerStateName:
        return ListenerStateName.listening

    def clear_buffer(self) -> None:
        self._clear_buffer_event.set()

    def commit(self) -> None:
        """
        主动变更状态. 有两个含义:
        1. 结束音频输入.
        2. 等待音频输出结束.
        """
        self._commit_event.set()
        self._logger.info("Committing changes.")

    def set_vad(self, vad_time: int) -> None:
        self._vad_time = vad_time

    def next(self) -> Union[ListenerState.NextState, None]:
        return self._next_state

    def _main_loop(self) -> None:
        try:
            self._run_main_loop()
        except Exception as e:
            self._logger.exception(e)
        finally:
            # 中断主循环会自动进入下一个状态.
            if self._next_state is None:
                self._try_close_on_next_state(None)
            # 保底逻辑.
            if self._callback_recognition and not self._callback_recognition.is_last:
                r = self._callback_recognition.model_copy()
                r.is_last = True
                self._callback.on_recognition(r)
                self._logger.info("main_loop finished with last recognition %s", r)
            self._logger.info("main_loop finished")
            self._main_loop_done_event.set()

    def _run_main_loop(self) -> None:
        while not self._closed_event.is_set():
            try:
                if self._next_state is not None:
                    # 退出.
                    self._closed_event.set()
                    self._logger.info("quit main loop on next state %s", self._next_state)
                    break
                batch_id = uuid()
                self._logger.info("start to run new audio asr batch %s", batch_id)
                self._run_audio_asr_batch(batch_id)
                # 默认继续进行 asr batch.
                self._ran_batch += 1
                # 判断是否要终结循环.
                self._logger.info("run audio asr batch times %s to %s", self._ran_batch, self._allow_batch)
                if 0 < self._allow_batch <= self._ran_batch:
                    self._logger.info("stop state on batch %s", self._ran_batch)
                    break
            except Exception as e:
                self._logger.exception(e)

    def _run_audio_asr_batch(self, batch_id: str) -> None:
        """
        运行一轮 asr. 这一轮 asr 结束时, 音频输入也会有一个中断.
        """
        self._clear_buffer_event.clear()
        _queue: deque[np.ndarray] = deque()
        audio_loop = AudioInputLoop(
            _queue.append,
            self._audio_input,
            resample_rate=self._recognizer.sample_rate,
            frame=self._recognizer.frame_duration,
        )
        asr = self._recognizer.new_batch(
            # 用自己来作为 callback.
            callback=self,
            batch_id=batch_id,
            vad=self._vad_time,
            stop_on_sentence=self._stop_on_sentence,
        )
        # 统一赋值. 通过这个对象可以拿到 batch 的信息.
        self._current_recognition_batch = asr
        try:
            with audio_loop:
                asr.start()
                self._run_asr_batch(asr=asr, _input=audio_loop, _queue=_queue)
        finally:
            self._logger.info("audio asr batch %s finally stopped", batch_id)
            # 关闭 asr batch.
            asr.close()
            _queue.clear()
            audio_loop.stop()
            audio_loop.join()
            self._commit_event.clear()
            self._logger.info("audio asr batch %s done", batch_id)

    def _run_asr_batch(self, asr: RecognitionBatch, _input: AudioInputLoop, _queue: deque[np.ndarray]) -> None:
        self._clear_buffer_event.clear()
        # 每次开始前, 先保证 commit 没有提交.
        self._commit_event.clear()
        committed = False
        left = Timeleft(0.3)
        # asr & audio input loop 在别的线程里独立运行中. 这个循环仅仅用来做响应逻辑.
        while not self._closed_event.is_set():
            if asr.is_done():
                # asr 结束, 可以正常退出.
                self._logger.info("audio asr %s is done", asr.batch_id)
                break
            elif self._clear_buffer_event.is_set():
                # 提交了清空历史事件. 直接退出这一轮.
                self._logger.info("clear buffer event set, quit asr batch %s", asr.batch_id)
                break

            # commit 判断逻辑. 只执行一次.
            if self._commit_event.is_set() and not committed:
                committed = True
                left = Timeleft(0.3)
                # 告知 asr 结束. 不过只告知一次.
                asr.commit()
                self._commit_event.clear()
                self._logger.info("commit event set, set committed to asr  %s", asr.batch_id)
                # 只提交一次.
                continue

            # 暂时没有输入数据, 但 asr 也没有结束. 继续等待.
            if len(_queue) == 0:
                # 等待下一轮检查.
                time.sleep(self._wait_audio_interval)
                continue

            audio_data = _queue.popleft()
            asr.buffer(audio_data)
            # 如果有识别结果, 尝试走 vad (如果定义了本地 vad)
            # 否则的话只走 asr 默认的 vad 逻辑.
            if self._vad_local_detector is not None:
                if self._vad_local_detector(audio_data, self._vad_time):
                    # vad 发生了, 主动 commit. 这样就只等待 asr 结束了.
                    self._commit_event.set()
                    self._logger.info("vad event happened at %s", asr.batch_id)
                    # 结束当前 batch.
                    continue

            rec = asr.get_last_recognition()
            if rec and rec.is_last:
                self._logger.info("receive asr last recognition %s", rec.batch_id)
                break
            if committed and not left.alive():
                self._logger.info("finish asr cause timeout after committed %s", rec.batch_id)
                break

    def _try_close_on_next_state(self, state_name: Optional[str]) -> None:
        if not state_name:
            state_name = self._default_next_state
        self._next_state = ListeningState.NextState(str(state_name), None)
        self._logger.info("close listening state and set next state %s", self._next_state)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        main_loop = Thread(target=self._main_loop, daemon=True)
        main_loop.start()

    def close(self) -> None:
        if self._closed_event.is_set():
            return
        # 设置退出.
        self._closed_event.set()
        self._logger.info("close listening state")
        self._main_loop_done_event.wait()
